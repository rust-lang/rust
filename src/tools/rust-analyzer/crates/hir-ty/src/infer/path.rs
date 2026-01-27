//! Path expression resolution.

use hir_def::{
    AdtId, AssocItemId, GenericDefId, ItemContainerId, Lookup,
    expr_store::path::{Path, PathSegment},
    resolver::{ResolveValueResult, TypeNs, ValueNs},
};
use hir_expand::name::Name;
use rustc_type_ir::inherent::{SliceLike, Ty as _};
use stdx::never;

use crate::{
    InferenceDiagnostic, ValueTyDefId,
    generics::generics,
    infer::diagnostics::InferenceTyLoweringContext as TyLoweringContext,
    lower::{GenericPredicates, LifetimeElisionKind},
    method_resolution::{self, CandidateId, MethodError},
    next_solver::{
        GenericArg, GenericArgs, TraitRef, Ty,
        infer::traits::{Obligation, ObligationCause},
        util::clauses_as_obligations,
    },
};

use super::{ExprOrPatId, InferenceContext, InferenceTyDiagnosticSource};

impl<'db> InferenceContext<'_, 'db> {
    pub(super) fn infer_path(&mut self, path: &Path, id: ExprOrPatId) -> Option<Ty<'db>> {
        let (value_def, generic_def, substs) = match self.resolve_value_path(path, id)? {
            ValuePathResolution::GenericDef(value_def, generic_def, substs) => {
                (value_def, generic_def, substs)
            }
            ValuePathResolution::NonGeneric(ty) => return Some(ty),
        };
        let args = self.insert_type_vars(substs);

        self.add_required_obligations_for_value_path(generic_def, args);

        let ty = self.db.value_ty(value_def)?.instantiate(self.interner(), args);
        let ty = self.process_remote_user_written_ty(ty);
        Some(ty)
    }

    fn resolve_value_path(
        &mut self,
        path: &Path,
        id: ExprOrPatId,
    ) -> Option<ValuePathResolution<'db>> {
        let (value, self_subst) = self.resolve_value_path_inner(path, id, false)?;

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
                    Some(ty) => Some(ValuePathResolution::NonGeneric(ty.as_ref())),
                    None => {
                        never!("uninferred pattern?");
                        None
                    }
                };
            }
            ValueNs::ImplSelf(impl_id) => {
                let ty = self.db.impl_self_ty(impl_id).instantiate_identity();
                return if let Some((AdtId::StructId(struct_id), substs)) = ty.as_adt() {
                    Some(ValuePathResolution::GenericDef(
                        struct_id.into(),
                        struct_id.into(),
                        substs,
                    ))
                } else {
                    // FIXME: report error, invalid Self reference
                    None
                };
            }
            ValueNs::GenericParam(it) => {
                return Some(ValuePathResolution::NonGeneric(self.db.const_param_ty_ns(it)));
            }
        };

        let generic_def = value_def.to_generic_def_id(self.db);
        if let GenericDefId::StaticId(_) = generic_def {
            // `Static` is the kind of item that can never be generic currently. We can just skip the binders to get its type.
            let ty = self.db.value_ty(value_def)?.skip_binder();
            let ty = self.process_remote_user_written_ty(ty);
            return Some(ValuePathResolution::NonGeneric(ty));
        };

        let substs = if self_subst.is_some_and(|it| !it.is_empty())
            && matches!(value_def, ValueTyDefId::EnumVariantId(_))
        {
            // This is something like `TypeAlias::<Args>::EnumVariant`. Do not call `substs_from_path()`,
            // as it'll try to re-lower the previous segment assuming it refers to the enum, but it refers
            // to the type alias and they may have different generics.
            self.types.empty.generic_args
        } else {
            self.with_body_ty_lowering(|ctx| {
                let mut path_ctx = ctx.at_path(path, id);
                let last_segment = path.segments().len().checked_sub(1);
                if let Some(last_segment) = last_segment {
                    path_ctx.set_current_segment(last_segment)
                }
                path_ctx.substs_from_path(value_def, true, false)
            })
        };

        let parent_substs_len = self_subst.map_or(0, |it| it.len());
        let substs = GenericArgs::fill_rest(
            self.interner(),
            generic_def.into(),
            self_subst.iter().flat_map(|it| it.iter()).chain(substs.iter().skip(parent_substs_len)),
            |_, id, _| GenericArg::error_from_id(self.interner(), id),
        );

        Some(ValuePathResolution::GenericDef(value_def, generic_def, substs))
    }

    pub(super) fn resolve_value_path_inner(
        &mut self,
        path: &Path,
        id: ExprOrPatId,
        no_diagnostics: bool,
    ) -> Option<(ValueNs, Option<GenericArgs<'db>>)> {
        // Don't use `self.make_ty()` here as we need `orig_ns`.
        let mut ctx = TyLoweringContext::new(
            self.db,
            &self.resolver,
            self.body,
            &self.diagnostics,
            InferenceTyDiagnosticSource::Body,
            self.generic_def,
            LifetimeElisionKind::Infer,
        );
        let mut path_ctx = if no_diagnostics {
            ctx.at_path_forget_diagnostics(path)
        } else {
            ctx.at_path(path, id)
        };
        let (value, self_subst) = if let Some(type_ref) = path.type_anchor() {
            let last = path.segments().last()?;

            let (ty, orig_ns) = path_ctx.ty_ctx().lower_ty_ext(type_ref);
            let ty = self.table.process_user_written_ty(ty);

            path_ctx.ignore_last_segment();
            let (ty, _) = path_ctx.lower_ty_relative_path(ty, orig_ns, true);
            drop_ctx(ctx, no_diagnostics);
            let ty = self.table.process_user_written_ty(ty);
            self.resolve_ty_assoc_item(ty, last.name, id).map(|(it, substs)| (it, Some(substs)))?
        } else {
            let hygiene = self.body.expr_or_pat_path_hygiene(id);
            // FIXME: report error, unresolved first path segment
            let value_or_partial = path_ctx.resolve_path_in_value_ns(hygiene)?;

            match value_or_partial {
                ResolveValueResult::ValueNs(it, _) => {
                    drop_ctx(ctx, no_diagnostics);
                    (it, None)
                }
                ResolveValueResult::Partial(def, remaining_index, _) => {
                    // there may be more intermediate segments between the resolved one and
                    // the end. Only the last segment needs to be resolved to a value; from
                    // the segments before that, we need to get either a type or a trait ref.

                    let remaining_segments = path.segments().skip(remaining_index);
                    let is_before_last = remaining_segments.len() == 1;
                    let last_segment = remaining_segments
                        .last()
                        .expect("there should be at least one segment here");

                    let (resolution, substs) = match (def, is_before_last) {
                        (TypeNs::TraitId(trait_), true) => {
                            let self_ty = self.table.next_ty_var();
                            let trait_ref =
                                path_ctx.lower_trait_ref_from_resolved_path(trait_, self_ty, true);
                            drop_ctx(ctx, no_diagnostics);
                            self.resolve_trait_assoc_item(trait_ref, last_segment, id)
                        }
                        (def, _) => {
                            // Either we already have a type (e.g. `Vec::new`), or we have a
                            // trait but it's not the last segment, so the next segment
                            // should resolve to an associated type of that trait (e.g. `<T
                            // as Iterator>::Item::default`)
                            path_ctx.ignore_last_segment();
                            let (ty, _) = path_ctx.lower_partly_resolved_path(def, true);
                            drop_ctx(ctx, no_diagnostics);
                            if ty.is_ty_error() {
                                return None;
                            }

                            let ty = self.process_user_written_ty(ty);

                            self.resolve_ty_assoc_item(ty, last_segment.name, id)
                        }
                    }?;
                    (resolution, Some(substs))
                }
            }
        };
        return Some((value, self_subst));

        #[inline]
        fn drop_ctx(mut ctx: TyLoweringContext<'_, '_>, no_diagnostics: bool) {
            if no_diagnostics {
                ctx.forget_diagnostics();
            }
        }
    }

    fn add_required_obligations_for_value_path(
        &mut self,
        def: GenericDefId,
        subst: GenericArgs<'db>,
    ) {
        let interner = self.interner();
        let predicates = GenericPredicates::query_all(self.db, def);
        let param_env = self.table.param_env;
        self.table.register_predicates(clauses_as_obligations(
            predicates.iter_instantiated_copied(interner, subst.as_slice()),
            ObligationCause::new(),
            param_env,
        ));

        // We need to add `Self: Trait` obligation when `def` is a trait assoc item.
        let container = match def {
            GenericDefId::FunctionId(id) => id.lookup(self.db).container,
            GenericDefId::ConstId(id) => id.lookup(self.db).container,
            _ => return,
        };

        if let ItemContainerId::TraitId(trait_) = container {
            let parent_len = generics(self.db, def).parent_generics().map_or(0, |g| g.len_self());
            let parent_subst = GenericArgs::new_from_slice(&subst.as_slice()[..parent_len]);
            let trait_ref = TraitRef::new_from_args(interner, trait_.into(), parent_subst);
            self.table.register_predicate(Obligation::new(
                interner,
                ObligationCause::new(),
                param_env,
                trait_ref,
            ));
        }
    }

    fn resolve_trait_assoc_item(
        &mut self,
        trait_ref: TraitRef<'db>,
        segment: PathSegment<'_>,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, GenericArgs<'db>)> {
        let trait_ = trait_ref.def_id.0;
        let item =
            trait_.trait_items(self.db).items.iter().map(|(_name, id)| *id).find_map(|item| {
                match item {
                    AssocItemId::FunctionId(func) => {
                        if segment.name == &self.db.function_signature(func).name {
                            Some(CandidateId::FunctionId(func))
                        } else {
                            None
                        }
                    }

                    AssocItemId::ConstId(konst) => {
                        if self.db.const_signature(konst).name.as_ref() == Some(segment.name) {
                            Some(CandidateId::ConstId(konst))
                        } else {
                            None
                        }
                    }
                    AssocItemId::TypeAliasId(_) => None,
                }
            })?;
        let def = match item {
            CandidateId::FunctionId(f) => ValueNs::FunctionId(f),
            CandidateId::ConstId(c) => ValueNs::ConstId(c),
        };

        self.write_assoc_resolution(id, item, trait_ref.args);
        Some((def, trait_ref.args))
    }

    fn resolve_ty_assoc_item(
        &mut self,
        ty: Ty<'db>,
        name: &Name,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, GenericArgs<'db>)> {
        if ty.is_ty_error() {
            return None;
        }

        if let Some(result) = self.resolve_enum_variant_on_ty(ty, name, id) {
            return Some(result);
        }

        let res = self.with_method_resolution(|ctx| {
            ctx.probe_for_name(method_resolution::Mode::Path, name.clone(), ty)
        });
        let (item, visible) = match res {
            Ok(res) => (res.item, true),
            Err(error) => match error {
                MethodError::PrivateMatch(candidate_id) => (candidate_id.item, false),
                _ => {
                    self.push_diagnostic(InferenceDiagnostic::UnresolvedAssocItem { id });
                    return None;
                }
            },
        };

        let (def, container) = match item {
            CandidateId::FunctionId(f) => (ValueNs::FunctionId(f), f.lookup(self.db).container),
            CandidateId::ConstId(c) => (ValueNs::ConstId(c), c.lookup(self.db).container),
        };
        let substs = match container {
            ItemContainerId::ImplId(impl_id) => {
                let impl_substs = self.table.fresh_args_for_item(impl_id.into());
                let impl_self_ty =
                    self.db.impl_self_ty(impl_id).instantiate(self.interner(), impl_substs);
                self.unify(impl_self_ty, ty);
                impl_substs
            }
            ItemContainerId::TraitId(trait_) => {
                // we're picking this method
                let args = GenericArgs::fill_rest(
                    self.interner(),
                    trait_.into(),
                    [ty.into()],
                    |_, id, _| self.table.next_var_for_param(id),
                );
                let trait_ref = TraitRef::new_from_args(self.interner(), trait_.into(), args);
                self.table.register_predicate(Obligation::new(
                    self.interner(),
                    ObligationCause::new(),
                    self.table.param_env,
                    trait_ref,
                ));
                args
            }
            ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => {
                never!("assoc item contained in module/extern block");
                return None;
            }
        };

        self.write_assoc_resolution(id, item, substs);
        if !visible {
            let item = match item {
                CandidateId::FunctionId(it) => it.into(),
                CandidateId::ConstId(it) => it.into(),
            };
            self.push_diagnostic(InferenceDiagnostic::PrivateAssocItem { id, item });
        }
        Some((def, substs))
    }

    fn resolve_enum_variant_on_ty(
        &mut self,
        ty: Ty<'db>,
        name: &Name,
        id: ExprOrPatId,
    ) -> Option<(ValueNs, GenericArgs<'db>)> {
        let ty = self.table.try_structurally_resolve_type(ty);
        let (enum_id, subst) = match ty.as_adt() {
            Some((AdtId::EnumId(e), subst)) => (e, subst),
            _ => return None,
        };
        let enum_data = enum_id.enum_variants(self.db);
        let variant = enum_data.variant(name)?;
        self.write_variant_resolution(id, variant.into());
        Some((ValueNs::EnumVariantId(variant), subst))
    }
}

#[derive(Debug)]
enum ValuePathResolution<'db> {
    // It's awkward to wrap a single ID in two enums, but we need both and this saves fallible
    // conversion between them + `unwrap()`.
    GenericDef(ValueTyDefId, GenericDefId, GenericArgs<'db>),
    NonGeneric(Ty<'db>),
}
