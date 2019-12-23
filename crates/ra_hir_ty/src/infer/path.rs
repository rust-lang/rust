//! Path expression resolution.

use std::iter;

use hir_def::{
    path::{Path, PathSegment},
    resolver::{ResolveValueResult, Resolver, TypeNs, ValueNs},
    AssocContainerId, AssocItemId, Lookup,
};
use hir_expand::name::Name;

use crate::{db::HirDatabase, method_resolution, Substs, Ty, TypeWalk, ValueTyDefId};

use super::{ExprOrPatId, InferenceContext, TraitRef};

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
        let (value, self_subst) = if let Some(type_ref) = path.type_anchor() {
            if path.segments().is_empty() {
                // This can't actually happen syntax-wise
                return None;
            }
            let ty = self.make_ty(type_ref);
            let remaining_segments_for_ty = path.segments().take(path.segments().len() - 1);
            let ty = Ty::from_type_relative_path(self.db, resolver, ty, remaining_segments_for_ty);
            self.resolve_ty_assoc_item(
                ty,
                &path.segments().last().expect("path had at least one segment").name,
                id,
            )?
        } else {
            let value_or_partial = resolver.resolve_path_in_value_ns(self.db, path.mod_path())?;

            match value_or_partial {
                ResolveValueResult::ValueNs(it) => (it, None),
                ResolveValueResult::Partial(def, remaining_index) => {
                    self.resolve_assoc_item(def, path, remaining_index, id)?
                }
            }
        };

        let typable: ValueTyDefId = match value {
            ValueNs::LocalBinding(pat) => {
                let ty = self.result.type_of_pat.get(pat)?.clone();
                let ty = self.resolve_ty_as_possible(ty);
                return Some(ty);
            }
            ValueNs::FunctionId(it) => it.into(),
            ValueNs::ConstId(it) => it.into(),
            ValueNs::StaticId(it) => it.into(),
            ValueNs::StructId(it) => it.into(),
            ValueNs::EnumVariantId(it) => it.into(),
        };

        let mut ty = self.db.value_ty(typable);
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
        assert!(remaining_index < path.segments().len());
        // there may be more intermediate segments between the resolved one and
        // the end. Only the last segment needs to be resolved to a value; from
        // the segments before that, we need to get either a type or a trait ref.

        let resolved_segment = path.segments().get(remaining_index - 1).unwrap();
        let remaining_segments = path.segments().skip(remaining_index);
        let is_before_last = remaining_segments.len() == 1;

        match (def, is_before_last) {
            (TypeNs::TraitId(trait_), true) => {
                let segment =
                    remaining_segments.last().expect("there should be at least one segment here");
                let trait_ref = TraitRef::from_resolved_path(
                    self.db,
                    &self.resolver,
                    trait_.into(),
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
                let remaining_segments_for_ty =
                    remaining_segments.take(remaining_segments.len() - 1);
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

                let ty = self.insert_type_vars(ty);
                let ty = self.normalize_associated_types_in(ty);

                let segment =
                    remaining_segments.last().expect("there should be at least one segment here");

                self.resolve_ty_assoc_item(ty, &segment.name, id)
            }
        }
    }

    fn resolve_trait_assoc_item(
        &mut self,
        trait_ref: TraitRef,
        segment: PathSegment<'_>,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Option<Substs>)> {
        let trait_ = trait_ref.trait_;
        let item = self
            .db
            .trait_data(trait_)
            .items
            .iter()
            .map(|(_name, id)| (*id).into())
            .find_map(|item| match item {
                AssocItemId::FunctionId(func) => {
                    if segment.name == &self.db.function_data(func).name {
                        Some(AssocItemId::FunctionId(func))
                    } else {
                        None
                    }
                }

                AssocItemId::ConstId(konst) => {
                    if self.db.const_data(konst).name.as_ref().map_or(false, |n| n == segment.name)
                    {
                        Some(AssocItemId::ConstId(konst))
                    } else {
                        None
                    }
                }
                AssocItemId::TypeAliasId(_) => None,
            })?;
        let def = match item {
            AssocItemId::FunctionId(f) => ValueNs::FunctionId(f),
            AssocItemId::ConstId(c) => ValueNs::ConstId(c),
            AssocItemId::TypeAliasId(_) => unreachable!(),
        };
        let substs = Substs::build_for_def(self.db, item)
            .use_parent_substs(&trait_ref.substs)
            .fill_with_params()
            .build();

        self.write_assoc_resolution(id, item);
        Some((def, Some(substs)))
    }

    fn resolve_ty_assoc_item(
        &mut self,
        ty: Ty,
        name: &Name,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Option<Substs>)> {
        if let Ty::Unknown = ty {
            return None;
        }

        let canonical_ty = self.canonicalizer().canonicalize_ty(ty.clone());

        method_resolution::iterate_method_candidates(
            &canonical_ty.value,
            self.db,
            &self.resolver.clone(),
            Some(name),
            method_resolution::LookupMode::Path,
            move |_ty, item| {
                let (def, container) = match item {
                    AssocItemId::FunctionId(f) => {
                        (ValueNs::FunctionId(f), f.lookup(self.db).container)
                    }
                    AssocItemId::ConstId(c) => (ValueNs::ConstId(c), c.lookup(self.db).container),
                    AssocItemId::TypeAliasId(_) => unreachable!(),
                };
                let substs = match container {
                    AssocContainerId::ImplId(impl_id) => {
                        let impl_substs = Substs::build_for_def(self.db, impl_id)
                            .fill(iter::repeat_with(|| self.table.new_type_var()))
                            .build();
                        let impl_self_ty = self.db.impl_self_ty(impl_id).subst(&impl_substs);
                        let substs = Substs::build_for_def(self.db, item)
                            .use_parent_substs(&impl_substs)
                            .fill_with_params()
                            .build();
                        self.unify(&impl_self_ty, &ty);
                        Some(substs)
                    }
                    AssocContainerId::TraitId(trait_) => {
                        // we're picking this method
                        let trait_substs = Substs::build_for_def(self.db, trait_)
                            .push(ty.clone())
                            .fill(std::iter::repeat_with(|| self.table.new_type_var()))
                            .build();
                        let substs = Substs::build_for_def(self.db, item)
                            .use_parent_substs(&trait_substs)
                            .fill_with_params()
                            .build();
                        self.obligations.push(super::Obligation::Trait(TraitRef {
                            trait_,
                            substs: trait_substs,
                        }));
                        Some(substs)
                    }
                    AssocContainerId::ContainerId(_) => None,
                };

                self.write_assoc_resolution(id, item.into());
                Some((def, substs))
            },
        )
    }
}
