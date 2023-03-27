//! Path expression resolution.

use chalk_ir::cast::Cast;
use hir_def::{
    path::{Path, PathSegment},
    resolver::{ResolveValueResult, TypeNs, ValueNs},
    AdtId, AssocItemId, EnumVariantId, ItemContainerId, Lookup,
};
use hir_expand::name::Name;
use stdx::never;

use crate::{
    builder::ParamKind,
    consteval,
    method_resolution::{self, VisibleFromModule},
    utils::generics,
    InferenceDiagnostic, Interner, Substitution, TraitRefExt, Ty, TyBuilder, TyExt, TyKind,
    ValueTyDefId,
};

use super::{ExprOrPatId, InferenceContext, TraitRef};

impl<'a> InferenceContext<'a> {
    pub(super) fn infer_path(&mut self, path: &Path, id: ExprOrPatId) -> Option<Ty> {
        let ty = self.resolve_value_path(path, id)?;
        let ty = self.insert_type_vars(ty);
        let ty = self.normalize_associated_types_in(ty);
        Some(ty)
    }

    fn resolve_value_path(&mut self, path: &Path, id: ExprOrPatId) -> Option<Ty> {
        let (value, self_subst) = if let Some(type_ref) = path.type_anchor() {
            let Some(last) = path.segments().last() else { return None };
            let ty = self.make_ty(type_ref);
            let remaining_segments_for_ty = path.segments().take(path.segments().len() - 1);
            let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver);
            let (ty, _) = ctx.lower_ty_relative_path(ty, None, remaining_segments_for_ty);
            self.resolve_ty_assoc_item(ty, last.name, id).map(|(it, substs)| (it, Some(substs)))?
        } else {
            // FIXME: report error, unresolved first path segment
            let value_or_partial =
                self.resolver.resolve_path_in_value_ns(self.db.upcast(), path.mod_path())?;

            match value_or_partial {
                ResolveValueResult::ValueNs(it) => (it, None),
                ResolveValueResult::Partial(def, remaining_index) => self
                    .resolve_assoc_item(def, path, remaining_index, id)
                    .map(|(it, substs)| (it, Some(substs)))?,
            }
        };

        let typable: ValueTyDefId = match value {
            ValueNs::LocalBinding(pat) => match self.result.type_of_binding.get(pat) {
                Some(ty) => return Some(ty.clone()),
                None => {
                    never!("uninferred pattern?");
                    return None;
                }
            },
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
                let substs = generics.placeholder_subst(self.db);
                let ty = self.db.impl_self_ty(impl_id).substitute(Interner, &substs);
                if let Some((AdtId::StructId(struct_id), substs)) = ty.as_adt() {
                    let ty = self.db.value_ty(struct_id.into()).substitute(Interner, &substs);
                    return Some(ty);
                } else {
                    // FIXME: report error, invalid Self reference
                    return None;
                }
            }
            ValueNs::GenericParam(it) => return Some(self.db.const_param_ty(it)),
        };

        let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver);
        let substs = ctx.substs_from_path(path, typable, true);
        let substs = substs.as_slice(Interner);
        let parent_substs = self_subst.or_else(|| {
            let generics = generics(self.db.upcast(), typable.to_generic_def_id()?);
            let parent_params_len = generics.parent_generics()?.len();
            let parent_args = &substs[substs.len() - parent_params_len..];
            Some(Substitution::from_iter(Interner, parent_args))
        });
        let parent_substs_len = parent_substs.as_ref().map_or(0, |s| s.len(Interner));
        let mut it = substs.iter().take(substs.len() - parent_substs_len).cloned();
        let ty = TyBuilder::value_ty(self.db, typable, parent_substs)
            .fill(|x| {
                it.next().unwrap_or_else(|| match x {
                    ParamKind::Type => self.result.standard_types.unknown.clone().cast(Interner),
                    ParamKind::Const(ty) => consteval::unknown_const_as_generic(ty.clone()),
                })
            })
            .build();
        Some(ty)
    }

    fn resolve_assoc_item(
        &mut self,
        def: TypeNs,
        path: &Path,
        remaining_index: usize,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Substitution)> {
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
                let trait_ref =
                    ctx.lower_trait_ref_from_resolved_path(trait_, resolved_segment, None);
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
                let (ty, _) = ctx.lower_partly_resolved_path(
                    def,
                    resolved_segment,
                    remaining_segments_for_ty,
                    true,
                );
                if ty.is_unknown() {
                    return None;
                }

                let ty = self.insert_type_vars(ty);
                let ty = self.normalize_associated_types_in(ty);

                let segment =
                    remaining_segments.last().expect("there should be at least one segment here");

                self.resolve_ty_assoc_item(ty, segment.name, id)
            }
        }
    }

    fn resolve_trait_assoc_item(
        &mut self,
        trait_ref: TraitRef,
        segment: PathSegment<'_>,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Substitution)> {
        let trait_ = trait_ref.hir_trait_id();
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

        self.write_assoc_resolution(id, item, trait_ref.substitution.clone());
        Some((def, trait_ref.substitution))
    }

    fn resolve_ty_assoc_item(
        &mut self,
        ty: Ty,
        name: &Name,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Substitution)> {
        if let TyKind::Error = ty.kind(Interner) {
            return None;
        }

        if let Some(result) = self.resolve_enum_variant_on_ty(&ty, name, id) {
            return Some(result);
        }

        let canonical_ty = self.canonicalize(ty.clone());

        let mut not_visible = None;
        let res = method_resolution::iterate_method_candidates(
            &canonical_ty.value,
            self.db,
            self.table.trait_env.clone(),
            self.get_traits_in_scope().as_ref().left_or_else(|&it| it),
            VisibleFromModule::Filter(self.resolver.module()),
            Some(name),
            method_resolution::LookupMode::Path,
            |_ty, item, visible| {
                if visible {
                    Some((item, true))
                } else {
                    if not_visible.is_none() {
                        not_visible = Some((item, false));
                    }
                    None
                }
            },
        );
        let res = res.or(not_visible);
        let (item, visible) = res?;

        let (def, container) = match item {
            AssocItemId::FunctionId(f) => {
                (ValueNs::FunctionId(f), f.lookup(self.db.upcast()).container)
            }
            AssocItemId::ConstId(c) => (ValueNs::ConstId(c), c.lookup(self.db.upcast()).container),
            AssocItemId::TypeAliasId(_) => unreachable!(),
        };
        let substs = match container {
            ItemContainerId::ImplId(impl_id) => {
                let impl_substs = TyBuilder::subst_for_def(self.db, impl_id, None)
                    .fill_with_inference_vars(&mut self.table)
                    .build();
                let impl_self_ty = self.db.impl_self_ty(impl_id).substitute(Interner, &impl_substs);
                self.unify(&impl_self_ty, &ty);
                impl_substs
            }
            ItemContainerId::TraitId(trait_) => {
                // we're picking this method
                let trait_ref = TyBuilder::trait_ref(self.db, trait_)
                    .push(ty.clone())
                    .fill_with_inference_vars(&mut self.table)
                    .build();
                self.push_obligation(trait_ref.clone().cast(Interner));
                trait_ref.substitution
            }
            ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => {
                never!("assoc item contained in module/extern block");
                return None;
            }
        };

        self.write_assoc_resolution(id, item, substs.clone());
        if !visible {
            self.push_diagnostic(InferenceDiagnostic::PrivateAssocItem { id, item });
        }
        Some((def, substs))
    }

    fn resolve_enum_variant_on_ty(
        &mut self,
        ty: &Ty,
        name: &Name,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Substitution)> {
        let ty = self.resolve_ty_shallow(ty);
        let (enum_id, subst) = match ty.as_adt() {
            Some((AdtId::EnumId(e), subst)) => (e, subst),
            _ => return None,
        };
        let enum_data = self.db.enum_data(enum_id);
        let local_id = enum_data.variant(name)?;
        let variant = EnumVariantId { parent: enum_id, local_id };
        self.write_variant_resolution(id, variant.into());
        Some((ValueNs::EnumVariantId(variant), subst.clone()))
    }
}
