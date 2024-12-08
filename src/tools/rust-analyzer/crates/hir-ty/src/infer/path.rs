//! Path expression resolution.

use chalk_ir::cast::Cast;
use hir_def::{
    path::{Path, PathSegment},
    resolver::{ResolveValueResult, TypeNs, ValueNs},
    AdtId, AssocItemId, GenericDefId, ItemContainerId, Lookup,
};
use hir_expand::name::Name;
use intern::sym;
use stdx::never;

use crate::{
    builder::ParamKind,
    consteval, error_lifetime,
    generics::generics,
    method_resolution::{self, VisibleFromModule},
    to_chalk_trait_id, InferenceDiagnostic, Interner, Substitution, TraitRef, TraitRefExt, Ty,
    TyBuilder, TyExt, TyKind, ValueTyDefId,
};

use super::{ExprOrPatId, InferenceContext};

impl InferenceContext<'_> {
    pub(super) fn infer_path(&mut self, path: &Path, id: ExprOrPatId) -> Option<Ty> {
        let (value_def, generic_def, substs) = match self.resolve_value_path(path, id)? {
            ValuePathResolution::GenericDef(value_def, generic_def, substs) => {
                (value_def, generic_def, substs)
            }
            ValuePathResolution::NonGeneric(ty) => return Some(ty),
        };
        let substs = self.insert_type_vars(substs);
        let substs = self.normalize_associated_types_in(substs);

        self.add_required_obligations_for_value_path(generic_def, &substs);

        let ty = self.db.value_ty(value_def)?.substitute(Interner, &substs);
        let ty = self.normalize_associated_types_in(ty);
        Some(ty)
    }

    fn resolve_value_path(&mut self, path: &Path, id: ExprOrPatId) -> Option<ValuePathResolution> {
        let (value, self_subst) = self.resolve_value_path_inner(path, id)?;

        let value_def: ValueTyDefId = match value {
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
            ValueNs::LocalBinding(pat) => {
                return match self.result.type_of_binding.get(pat) {
                    Some(ty) => Some(ValuePathResolution::NonGeneric(ty.clone())),
                    None => {
                        never!("uninferred pattern?");
                        None
                    }
                }
            }
            ValueNs::ImplSelf(impl_id) => {
                let generics = crate::generics::generics(self.db.upcast(), impl_id.into());
                let substs = generics.placeholder_subst(self.db);
                let ty = self.db.impl_self_ty(impl_id).substitute(Interner, &substs);
                return if let Some((AdtId::StructId(struct_id), substs)) = ty.as_adt() {
                    Some(ValuePathResolution::GenericDef(
                        struct_id.into(),
                        struct_id.into(),
                        substs.clone(),
                    ))
                } else {
                    // FIXME: report error, invalid Self reference
                    None
                };
            }
            ValueNs::GenericParam(it) => {
                return Some(ValuePathResolution::NonGeneric(self.db.const_param_ty(it)))
            }
        };

        let generic_def_id = value_def.to_generic_def_id(self.db);
        let Some(generic_def) = generic_def_id else {
            // `value_def` is the kind of item that can never be generic (i.e. statics, at least
            // currently). We can just skip the binders to get its type.
            let (ty, binders) = self.db.value_ty(value_def)?.into_value_and_skipped_binders();
            stdx::always!(binders.is_empty(Interner), "non-empty binders for non-generic def",);
            return Some(ValuePathResolution::NonGeneric(ty));
        };

        let substs = self.with_body_ty_lowering(|ctx| ctx.substs_from_path(path, value_def, true));
        let substs = substs.as_slice(Interner);

        if let ValueNs::EnumVariantId(_) = value {
            let mut it = self_subst
                .as_ref()
                .map_or(&[][..], |s| s.as_slice(Interner))
                .iter()
                .chain(substs)
                .cloned();
            let builder = TyBuilder::subst_for_def(self.db, generic_def, None);
            let substs = builder
                .fill(|x| {
                    it.next().unwrap_or_else(|| match x {
                        ParamKind::Type => {
                            self.result.standard_types.unknown.clone().cast(Interner)
                        }
                        ParamKind::Const(ty) => consteval::unknown_const_as_generic(ty.clone()),
                        ParamKind::Lifetime => error_lifetime().cast(Interner),
                    })
                })
                .build();

            return Some(ValuePathResolution::GenericDef(value_def, generic_def, substs));
        }

        let parent_substs = self_subst.or_else(|| {
            let generics = generics(self.db.upcast(), generic_def_id?);
            let parent_params_len = generics.parent_generics()?.len();
            let parent_args = &substs[substs.len() - parent_params_len..];
            Some(Substitution::from_iter(Interner, parent_args))
        });
        let parent_substs_len = parent_substs.as_ref().map_or(0, |s| s.len(Interner));
        let mut it = substs.iter().take(substs.len() - parent_substs_len).cloned();
        let builder = TyBuilder::subst_for_def(self.db, generic_def, parent_substs);
        let substs = builder
            .fill(|x| {
                it.next().unwrap_or_else(|| match x {
                    ParamKind::Type => self.result.standard_types.unknown.clone().cast(Interner),
                    ParamKind::Const(ty) => consteval::unknown_const_as_generic(ty.clone()),
                    ParamKind::Lifetime => error_lifetime().cast(Interner),
                })
            })
            .build();

        Some(ValuePathResolution::GenericDef(value_def, generic_def, substs))
    }

    pub(super) fn resolve_value_path_inner(
        &mut self,
        path: &Path,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Option<chalk_ir::Substitution<Interner>>)> {
        let (value, self_subst) = if let Some(type_ref) = path.type_anchor() {
            let last = path.segments().last()?;

            // Don't use `self.make_ty()` here as we need `orig_ns`.
            let mut ctx = crate::lower::TyLoweringContext::new(
                self.db,
                &self.resolver,
                &self.body.types,
                self.owner.into(),
            );
            let (ty, orig_ns) = ctx.lower_ty_ext(type_ref);
            let ty = self.table.insert_type_vars(ty);
            let ty = self.table.normalize_associated_types_in(ty);

            let remaining_segments_for_ty = path.segments().take(path.segments().len() - 1);
            let (ty, _) = ctx.lower_ty_relative_path(ty, orig_ns, remaining_segments_for_ty);
            let ty = self.table.insert_type_vars(ty);
            let ty = self.table.normalize_associated_types_in(ty);
            self.resolve_ty_assoc_item(ty, last.name, id).map(|(it, substs)| (it, Some(substs)))?
        } else {
            let hygiene = self.body.expr_or_pat_path_hygiene(id);
            // FIXME: report error, unresolved first path segment
            let value_or_partial =
                self.resolver.resolve_path_in_value_ns(self.db.upcast(), path, hygiene)?;

            match value_or_partial {
                ResolveValueResult::ValueNs(it, _) => (it, None),
                ResolveValueResult::Partial(def, remaining_index, _) => self
                    .resolve_assoc_item(def, path, remaining_index, id)
                    .map(|(it, substs)| (it, Some(substs)))?,
            }
        };
        Some((value, self_subst))
    }

    fn add_required_obligations_for_value_path(&mut self, def: GenericDefId, subst: &Substitution) {
        let predicates = self.db.generic_predicates(def);
        for predicate in predicates.iter() {
            let (predicate, binders) =
                predicate.clone().substitute(Interner, &subst).into_value_and_skipped_binders();
            // Quantified where clauses are not yet handled.
            stdx::always!(binders.is_empty(Interner));
            self.push_obligation(predicate.cast(Interner));
        }

        // We need to add `Self: Trait` obligation when `def` is a trait assoc item.
        let container = match def {
            GenericDefId::FunctionId(id) => id.lookup(self.db.upcast()).container,
            GenericDefId::ConstId(id) => id.lookup(self.db.upcast()).container,
            _ => return,
        };

        if let ItemContainerId::TraitId(trait_) = container {
            let param_len = generics(self.db.upcast(), def).len_self();
            let parent_subst =
                Substitution::from_iter(Interner, subst.iter(Interner).skip(param_len));
            let trait_ref =
                TraitRef { trait_id: to_chalk_trait_id(trait_), substitution: parent_subst };
            self.push_obligation(trait_ref.cast(Interner));
        }
    }

    fn resolve_assoc_item(
        &mut self,
        def: TypeNs,
        path: &Path,
        remaining_index: usize,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, Substitution)> {
        // there may be more intermediate segments between the resolved one and
        // the end. Only the last segment needs to be resolved to a value; from
        // the segments before that, we need to get either a type or a trait ref.

        let _d;
        let (resolved_segment, remaining_segments) = match path {
            Path::Normal { .. } | Path::BarePath(_) => {
                assert!(remaining_index < path.segments().len());
                (
                    path.segments().get(remaining_index - 1).unwrap(),
                    path.segments().skip(remaining_index),
                )
            }
            Path::LangItem(..) => (
                PathSegment {
                    name: {
                        _d = Name::new_symbol_root(sym::Unknown.clone());
                        &_d
                    },
                    args_and_bindings: None,
                },
                path.segments(),
            ),
        };
        let is_before_last = remaining_segments.len() == 1;

        match (def, is_before_last) {
            (TypeNs::TraitId(trait_), true) => {
                let segment =
                    remaining_segments.last().expect("there should be at least one segment here");
                let self_ty = self.table.new_type_var();
                let trait_ref = self.with_body_ty_lowering(|ctx| {
                    ctx.lower_trait_ref_from_resolved_path(trait_, resolved_segment, self_ty)
                });
                self.resolve_trait_assoc_item(trait_ref, segment, id)
            }
            (def, _) => {
                // Either we already have a type (e.g. `Vec::new`), or we have a
                // trait but it's not the last segment, so the next segment
                // should resolve to an associated type of that trait (e.g. `<T
                // as Iterator>::Item::default`)
                let remaining_segments_for_ty =
                    remaining_segments.take(remaining_segments.len() - 1);
                let (ty, _) = self.with_body_ty_lowering(|ctx| {
                    ctx.lower_partly_resolved_path(
                        def,
                        resolved_segment,
                        remaining_segments_for_ty,
                        true,
                    )
                });
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
            self.db.trait_data(trait_).items.iter().map(|(_name, id)| *id).find_map(|item| {
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
            &canonical_ty,
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
        if res.is_none() {
            self.push_diagnostic(InferenceDiagnostic::UnresolvedAssocItem { id });
        }
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
        let variant = enum_data.variant(name)?;
        self.write_variant_resolution(id, variant.into());
        Some((ValueNs::EnumVariantId(variant), subst.clone()))
    }
}

#[derive(Debug)]
enum ValuePathResolution {
    // It's awkward to wrap a single ID in two enums, but we need both and this saves fallible
    // conversion between them + `unwrap()`.
    GenericDef(ValueTyDefId, GenericDefId, Substitution),
    NonGeneric(Ty),
}
