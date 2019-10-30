//! Path expression resolution.

use hir_def::path::PathSegment;

use super::{ExprOrPatId, InferenceContext, TraitRef};
use crate::{
    db::HirDatabase,
    resolve::{ResolveValueResult, Resolver, TypeNs, ValueNs},
    ty::{Substs, Ty, TypableDef, TypeWalk},
    AssocItem, HasGenericParams, Namespace, Path,
};

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    pub(super) fn infer_path(
        &mut self,
        resolver: &Resolver,
        path: &Path,
        id: ExprOrPatId,
    ) -> Option<Ty> {
        let ty = self.resolve_value_path(resolver, path, id)?;
        let ty = self.insert_type_vars(ty);
        let ty = self.normalize_associated_types_in(ty);
        Some(ty)
    }

    fn resolve_value_path(
        &mut self,
        resolver: &Resolver,
        path: &Path,
        id: ExprOrPatId,
    ) -> Option<Ty> {
        let (value, self_subst) = if let crate::PathKind::Type(type_ref) = &path.kind {
            if path.segments.is_empty() {
                // This can't actually happen syntax-wise
                return None;
            }
            let ty = self.make_ty(type_ref);
            let remaining_segments_for_ty = &path.segments[..path.segments.len() - 1];
            let ty = Ty::from_type_relative_path(self.db, resolver, ty, remaining_segments_for_ty);
            self.resolve_ty_assoc_item(
                ty,
                path.segments.last().expect("path had at least one segment"),
                id,
            )?
        } else {
            let value_or_partial = resolver.resolve_path_in_value_ns(self.db, &path)?;

            match value_or_partial {
                ResolveValueResult::ValueNs(it) => (it, None),
                ResolveValueResult::Partial(def, remaining_index) => {
                    self.resolve_assoc_item(def, path, remaining_index, id)?
                }
            }
        };

        let typable: TypableDef = match value {
            ValueNs::LocalBinding(pat) => {
                let ty = self.result.type_of_pat.get(pat)?.clone();
                let ty = self.resolve_ty_as_possible(&mut vec![], ty);
                return Some(ty);
            }
            ValueNs::Function(it) => it.into(),
            ValueNs::Const(it) => it.into(),
            ValueNs::Static(it) => it.into(),
            ValueNs::Struct(it) => it.into(),
            ValueNs::EnumVariant(it) => it.into(),
        };

        let mut ty = self.db.type_for_def(typable, Namespace::Values);
        if let Some(self_subst) = self_subst {
            ty = ty.subst(&self_subst);
        }

        let substs = Ty::substs_from_path(self.db, &self.resolver, path, typable);
        let ty = ty.subst(&substs);
        Some(ty)
    }

    fn resolve_assoc_item(
        &mut self,
        def: TypeNs,
        path: &Path,
        remaining_index: usize,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Option<Substs>)> {
        assert!(remaining_index < path.segments.len());
        // there may be more intermediate segments between the resolved one and
        // the end. Only the last segment needs to be resolved to a value; from
        // the segments before that, we need to get either a type or a trait ref.

        let resolved_segment = &path.segments[remaining_index - 1];
        let remaining_segments = &path.segments[remaining_index..];
        let is_before_last = remaining_segments.len() == 1;

        match (def, is_before_last) {
            (TypeNs::Trait(trait_), true) => {
                let segment =
                    remaining_segments.last().expect("there should be at least one segment here");
                let trait_ref = TraitRef::from_resolved_path(
                    self.db,
                    &self.resolver,
                    trait_,
                    resolved_segment,
                    None,
                );
                self.resolve_trait_assoc_item(trait_ref, segment, id)
            }
            (def, _) => {
                // Either we already have a type (e.g. `Vec::new`), or we have a
                // trait but it's not the last segment, so the next segment
                // should resolve to an associated type of that trait (e.g. `<T
                // as Iterator>::Item::default`)
                let remaining_segments_for_ty = &remaining_segments[..remaining_segments.len() - 1];
                let ty = Ty::from_partly_resolved_hir_path(
                    self.db,
                    &self.resolver,
                    def,
                    resolved_segment,
                    remaining_segments_for_ty,
                );
                if let Ty::Unknown = ty {
                    return None;
                }

                let segment =
                    remaining_segments.last().expect("there should be at least one segment here");

                self.resolve_ty_assoc_item(ty, segment, id)
            }
        }
    }

    fn resolve_trait_assoc_item(
        &mut self,
        trait_ref: TraitRef,
        segment: &PathSegment,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Option<Substs>)> {
        let trait_ = trait_ref.trait_;
        let item = trait_.items(self.db).iter().copied().find_map(|item| match item {
            AssocItem::Function(func) => {
                if segment.name == func.name(self.db) {
                    Some(AssocItem::Function(func))
                } else {
                    None
                }
            }

            AssocItem::Const(konst) => {
                if konst.name(self.db).map_or(false, |n| n == segment.name) {
                    Some(AssocItem::Const(konst))
                } else {
                    None
                }
            }
            AssocItem::TypeAlias(_) => None,
        })?;
        let def = match item {
            AssocItem::Function(f) => ValueNs::Function(f),
            AssocItem::Const(c) => ValueNs::Const(c),
            AssocItem::TypeAlias(_) => unreachable!(),
        };
        let substs = Substs::build_for_def(self.db, item)
            .use_parent_substs(&trait_ref.substs)
            .fill_with_unknown()
            .build();

        self.write_assoc_resolution(id, item);
        Some((def, Some(substs)))
    }

    fn resolve_ty_assoc_item(
        &mut self,
        ty: Ty,
        segment: &PathSegment,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Option<Substs>)> {
        if let Ty::Unknown = ty {
            return None;
        }

        let krate = self.resolver.krate()?;

        // Find impl
        // FIXME: consider trait candidates
        let item = ty.clone().iterate_impl_items(self.db, krate, |item| match item {
            AssocItem::Function(func) => {
                if segment.name == func.name(self.db) {
                    Some(AssocItem::Function(func))
                } else {
                    None
                }
            }

            AssocItem::Const(konst) => {
                if konst.name(self.db).map_or(false, |n| n == segment.name) {
                    Some(AssocItem::Const(konst))
                } else {
                    None
                }
            }
            AssocItem::TypeAlias(_) => None,
        })?;
        let def = match item {
            AssocItem::Function(f) => ValueNs::Function(f),
            AssocItem::Const(c) => ValueNs::Const(c),
            AssocItem::TypeAlias(_) => unreachable!(),
        };
        let substs = self.find_self_types(&def, ty);

        self.write_assoc_resolution(id, item);
        Some((def, substs))
    }

    fn find_self_types(&self, def: &ValueNs, actual_def_ty: Ty) -> Option<Substs> {
        if let ValueNs::Function(func) = def {
            // We only do the infer if parent has generic params
            let gen = func.generic_params(self.db);
            if gen.count_parent_params() == 0 {
                return None;
            }

            let impl_block = func.impl_block(self.db)?.target_ty(self.db);
            let impl_block_substs = impl_block.substs()?;
            let actual_substs = actual_def_ty.substs()?;

            let mut new_substs = vec![Ty::Unknown; gen.count_parent_params()];

            // The following code *link up* the function actual parma type
            // and impl_block type param index
            impl_block_substs.iter().zip(actual_substs.iter()).for_each(|(param, pty)| {
                if let Ty::Param { idx, .. } = param {
                    if let Some(s) = new_substs.get_mut(*idx as usize) {
                        *s = pty.clone();
                    }
                }
            });

            Some(Substs(new_substs.into()))
        } else {
            None
        }
    }
}
