//! Path expression resolution.

use std::iter;

use hir_def::{
    path::{Path, PathSegment},
    resolver::{ResolveValueResult, Resolver, TypeNs, ValueNs},
    AdtId, AssocContainerId, AssocItemId, EnumVariantId, Lookup,
};
use hir_expand::name::Name;

use crate::{method_resolution, Interner, Substs, Ty, TyKind, ValueTyDefId};

use super::{ExprOrPatId, InferenceContext, TraitRef};

impl<'a> InferenceContext<'a> {
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
            let ctx = crate::lower::TyLoweringContext::new(self.db, &resolver);
            let (ty, _) = Ty::from_type_relative_path(&ctx, ty, None, remaining_segments_for_ty);
            self.resolve_ty_assoc_item(
                ty,
                &path.segments().last().expect("path had at least one segment").name,
                id,
            )?
        } else {
            let value_or_partial =
                resolver.resolve_path_in_value_ns(self.db.upcast(), path.mod_path())?;

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
            ValueNs::StructId(it) => {
                self.write_variant_resolution(id, it.into());

                it.into()
            }
            ValueNs::EnumVariantId(it) => {
                self.write_variant_resolution(id, it.into());

                it.into()
            }
            ValueNs::ImplSelf(impl_id) => {
                let generics = crate::utils::generics(self.db.upcast(), impl_id.into());
                let substs = Substs::type_params_for_generics(self.db, &generics);
                let ty = self.db.impl_self_ty(impl_id).subst(&substs);
                if let Some((AdtId::StructId(struct_id), substs)) = ty.as_adt() {
                    let ty = self.db.value_ty(struct_id.into()).subst(&substs);
                    return Some(ty);
                } else {
                    // FIXME: diagnostic, invalid Self reference
                    return None;
                }
            }
            ValueNs::GenericParam(it) => return Some(self.db.const_param_ty(it)),
        };

        let ty = self.db.value_ty(typable);
        // self_subst is just for the parent
        let parent_substs = self_subst.unwrap_or_else(Substs::empty);
        let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver);
        let substs = Ty::substs_from_path(&ctx, path, typable, true);
        let full_substs = Substs::builder(substs.len())
            .use_parent_substs(&parent_substs)
            .fill(substs.0[parent_substs.len()..].iter().cloned())
            .build();
        let ty = ty.subst(&full_substs);
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
                let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver);
                let trait_ref = TraitRef::from_resolved_path(&ctx, trait_, resolved_segment, None);
                self.resolve_trait_assoc_item(trait_ref, segment, id)
            }
            (def, _) => {
                // Either we already have a type (e.g. `Vec::new`), or we have a
                // trait but it's not the last segment, so the next segment
                // should resolve to an associated type of that trait (e.g. `<T
                // as Iterator>::Item::default`)
                let remaining_segments_for_ty =
                    remaining_segments.take(remaining_segments.len() - 1);
                let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver);
                let (ty, _) = Ty::from_partly_resolved_hir_path(
                    &ctx,
                    def,
                    resolved_segment,
                    remaining_segments_for_ty,
                    true,
                );
                if let TyKind::Unknown = ty.interned(&Interner) {
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
        let item =
            self.db.trait_data(trait_).items.iter().map(|(_name, id)| (*id)).find_map(|item| {
                match item {
                    AssocItemId::FunctionId(func) => {
                        if segment.name == &self.db.function_data(func).name {
                            Some(AssocItemId::FunctionId(func))
                        } else {
                            None
                        }
                    }

                    AssocItemId::ConstId(konst) => {
                        if self
                            .db
                            .const_data(konst)
                            .name
                            .as_ref()
                            .map_or(false, |n| n == segment.name)
                        {
                            Some(AssocItemId::ConstId(konst))
                        } else {
                            None
                        }
                    }
                    AssocItemId::TypeAliasId(_) => None,
                }
            })?;
        let def = match item {
            AssocItemId::FunctionId(f) => ValueNs::FunctionId(f),
            AssocItemId::ConstId(c) => ValueNs::ConstId(c),
            AssocItemId::TypeAliasId(_) => unreachable!(),
        };

        self.write_assoc_resolution(id, item);
        Some((def, Some(trait_ref.substs)))
    }

    fn resolve_ty_assoc_item(
        &mut self,
        ty: Ty,
        name: &Name,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Option<Substs>)> {
        if let TyKind::Unknown = ty.interned(&Interner) {
            return None;
        }

        if let Some(result) = self.resolve_enum_variant_on_ty(&ty, name, id) {
            return Some(result);
        }

        let canonical_ty = self.canonicalizer().canonicalize_ty(ty.clone());
        let krate = self.resolver.krate()?;
        let traits_in_scope = self.resolver.traits_in_scope(self.db.upcast());

        method_resolution::iterate_method_candidates(
            &canonical_ty.value,
            self.db,
            self.trait_env.clone(),
            krate,
            &traits_in_scope,
            Some(name),
            method_resolution::LookupMode::Path,
            move |_ty, item| {
                let (def, container) = match item {
                    AssocItemId::FunctionId(f) => {
                        (ValueNs::FunctionId(f), f.lookup(self.db.upcast()).container)
                    }
                    AssocItemId::ConstId(c) => {
                        (ValueNs::ConstId(c), c.lookup(self.db.upcast()).container)
                    }
                    AssocItemId::TypeAliasId(_) => unreachable!(),
                };
                let substs = match container {
                    AssocContainerId::ImplId(impl_id) => {
                        let impl_substs = Substs::build_for_def(self.db, impl_id)
                            .fill(iter::repeat_with(|| self.table.new_type_var()))
                            .build();
                        let impl_self_ty = self.db.impl_self_ty(impl_id).subst(&impl_substs);
                        self.unify(&impl_self_ty, &ty);
                        Some(impl_substs)
                    }
                    AssocContainerId::TraitId(trait_) => {
                        // we're picking this method
                        let trait_substs = Substs::build_for_def(self.db, trait_)
                            .push(ty.clone())
                            .fill(std::iter::repeat_with(|| self.table.new_type_var()))
                            .build();
                        self.obligations.push(super::Obligation::Trait(TraitRef {
                            trait_,
                            substs: trait_substs.clone(),
                        }));
                        Some(trait_substs)
                    }
                    AssocContainerId::ModuleId(_) => None,
                };

                self.write_assoc_resolution(id, item);
                Some((def, substs))
            },
        )
    }

    fn resolve_enum_variant_on_ty(
        &mut self,
        ty: &Ty,
        name: &Name,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Option<Substs>)> {
        let (enum_id, subst) = match ty.as_adt() {
            Some((AdtId::EnumId(e), subst)) => (e, subst),
            _ => return None,
        };
        let enum_data = self.db.enum_data(enum_id);
        let local_id = enum_data.variant(name)?;
        let variant = EnumVariantId { parent: enum_id, local_id };
        self.write_variant_resolution(id, variant.into());
        Some((ValueNs::EnumVariantId(variant), Some(subst.clone())))
    }
}
