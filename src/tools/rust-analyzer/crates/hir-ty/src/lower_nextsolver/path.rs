//! A wrapper around [`TyLoweringContext`] specifically for lowering paths.

use std::ops::Deref;

use either::Either;
use hir_def::{
    AssocItemId, GenericDefId, GenericParamId, Lookup, TraitId, TypeAliasId,
    builtin_type::BuiltinType,
    expr_store::{
        ExpressionStore, HygieneId,
        path::{GenericArg, GenericArgs, GenericArgsParentheses, Path, PathSegment, PathSegments},
    },
    hir::generics::{
        GenericParamDataRef, TypeOrConstParamData, TypeParamData, TypeParamProvenance,
    },
    resolver::{ResolveValueResult, TypeNs, ValueNs},
    signatures::TraitFlags,
    type_ref::{TypeRef, TypeRefId},
};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::FxHashSet;
use rustc_type_ir::{
    AliasTerm, AliasTy, AliasTyKind, TypeVisitableExt,
    inherent::{GenericArgs as _, IntoKind, Region as _, SliceLike, Ty as _},
};
use smallvec::{SmallVec, smallvec};
use stdx::never;

use crate::{
    GenericArgsProhibitedReason, IncorrectGenericsLenKind, PathGenericsSource,
    PathLoweringDiagnostic, TyDefId, ValueTyDefId,
    consteval_nextsolver::{unknown_const, unknown_const_as_generic},
    db::HirDatabase,
    generics::{Generics, generics},
    lower::PathDiagnosticCallbackData,
    lower_nextsolver::{
        LifetimeElisionKind, PredicateFilter, generic_predicates_filtered_by,
        named_associated_type_shorthand_candidates,
    },
    next_solver::{
        AdtDef, Binder, Clause, Const, DbInterner, ErrorGuaranteed, Predicate, ProjectionPredicate,
        Region, SolverDefId, TraitRef, Ty,
        mapping::{ChalkToNextSolver, convert_binder_to_early_binder},
    },
    primitive,
};

use super::{
    ImplTraitLoweringMode, TyLoweringContext, associated_type_by_name_including_super_traits,
    const_param_ty_query, ty_query,
};

type CallbackData<'a> =
    Either<PathDiagnosticCallbackData, crate::infer::diagnostics::PathDiagnosticCallbackData<'a>>;

// We cannot use `&mut dyn FnMut()` because of lifetime issues, and we don't want to use `Box<dyn FnMut()>`
// because of the allocation, so we create a lifetime-less callback, tailored for our needs.
pub(crate) struct PathDiagnosticCallback<'a, 'db> {
    pub(crate) data: CallbackData<'a>,
    pub(crate) callback:
        fn(&CallbackData<'_>, &mut TyLoweringContext<'db, '_>, PathLoweringDiagnostic),
}

pub(crate) struct PathLoweringContext<'a, 'b, 'db> {
    ctx: &'a mut TyLoweringContext<'db, 'b>,
    on_diagnostic: PathDiagnosticCallback<'a, 'db>,
    path: &'a Path,
    segments: PathSegments<'a>,
    current_segment_idx: usize,
    /// Contains the previous segment if `current_segment_idx == segments.len()`
    current_or_prev_segment: PathSegment<'a>,
}

impl<'a, 'b, 'db> PathLoweringContext<'a, 'b, 'db> {
    #[inline]
    pub(crate) fn new(
        ctx: &'a mut TyLoweringContext<'db, 'b>,
        on_diagnostic: PathDiagnosticCallback<'a, 'db>,
        path: &'a Path,
    ) -> Self {
        let segments = path.segments();
        let first_segment = segments.first().unwrap_or(PathSegment::MISSING);
        Self {
            ctx,
            on_diagnostic,
            path,
            segments,
            current_segment_idx: 0,
            current_or_prev_segment: first_segment,
        }
    }

    #[inline]
    #[cold]
    fn on_diagnostic(&mut self, diag: PathLoweringDiagnostic) {
        (self.on_diagnostic.callback)(&self.on_diagnostic.data, self.ctx, diag);
    }

    #[inline]
    pub(crate) fn ty_ctx(&mut self) -> &mut TyLoweringContext<'db, 'b> {
        self.ctx
    }

    #[inline]
    fn current_segment_u32(&self) -> u32 {
        self.current_segment_idx as u32
    }

    #[inline]
    fn skip_resolved_segment(&mut self) {
        if !matches!(self.path, Path::LangItem(..)) {
            // In lang items, the resolved "segment" is not one of the segments. Perhaps we should've put it
            // point at -1, but I don't feel this is clearer.
            self.current_segment_idx += 1;
        }
        self.update_current_segment();
    }

    #[inline]
    fn update_current_segment(&mut self) {
        self.current_or_prev_segment =
            self.segments.get(self.current_segment_idx).unwrap_or(self.current_or_prev_segment);
    }

    #[inline]
    pub(crate) fn ignore_last_segment(&mut self) {
        self.segments = self.segments.strip_last();
    }

    #[inline]
    pub(crate) fn set_current_segment(&mut self, segment: usize) {
        self.current_segment_idx = segment;
        self.current_or_prev_segment = self
            .segments
            .get(segment)
            .expect("invalid segment passed to PathLoweringContext::set_current_segment()");
    }

    #[inline]
    fn with_lifetime_elision<T>(
        &mut self,
        lifetime_elision: LifetimeElisionKind<'db>,
        f: impl FnOnce(&mut PathLoweringContext<'_, '_, 'db>) -> T,
    ) -> T {
        let old_lifetime_elision =
            std::mem::replace(&mut self.ctx.lifetime_elision, lifetime_elision);
        let result = f(self);
        self.ctx.lifetime_elision = old_lifetime_elision;
        result
    }

    pub(crate) fn lower_ty_relative_path(
        &mut self,
        ty: Ty<'db>,
        // We need the original resolution to lower `Self::AssocTy` correctly
        res: Option<TypeNs>,
    ) -> (Ty<'db>, Option<TypeNs>) {
        let remaining_segments = self.segments.len() - self.current_segment_idx;
        match remaining_segments {
            0 => (ty, res),
            1 => {
                // resolve unselected assoc types
                (self.select_associated_type(res), None)
            }
            _ => {
                // FIXME report error (ambiguous associated type)
                (Ty::new_error(self.ctx.interner, ErrorGuaranteed), None)
            }
        }
    }

    fn prohibit_parenthesized_generic_args(&mut self) -> bool {
        if let Some(generic_args) = self.current_or_prev_segment.args_and_bindings {
            match generic_args.parenthesized {
                GenericArgsParentheses::No => {}
                GenericArgsParentheses::ReturnTypeNotation | GenericArgsParentheses::ParenSugar => {
                    let segment = self.current_segment_u32();
                    self.on_diagnostic(
                        PathLoweringDiagnostic::ParenthesizedGenericArgsWithoutFnTrait { segment },
                    );
                    return true;
                }
            }
        }
        false
    }

    // When calling this, the current segment is the resolved segment (we don't advance it yet).
    pub(crate) fn lower_partly_resolved_path(
        &mut self,
        resolution: TypeNs,
        infer_args: bool,
    ) -> (Ty<'db>, Option<TypeNs>) {
        let remaining_segments = self.segments.skip(self.current_segment_idx + 1);
        tracing::debug!(?remaining_segments);
        let rem_seg_len = remaining_segments.len();
        tracing::debug!(?rem_seg_len);

        let ty = match resolution {
            TypeNs::TraitId(trait_) => {
                let ty = match remaining_segments.len() {
                    1 => {
                        let trait_ref = self.lower_trait_ref_from_resolved_path(
                            trait_,
                            Ty::new_error(self.ctx.interner, ErrorGuaranteed),
                        );
                        tracing::debug!(?trait_ref);
                        self.skip_resolved_segment();
                        let segment = self.current_or_prev_segment;
                        let trait_id = trait_ref.def_id.0;
                        let found =
                            trait_id.trait_items(self.ctx.db).associated_type_by_name(segment.name);

                        tracing::debug!(?found);
                        match found {
                            Some(associated_ty) => {
                                // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                                // generic params. It's inefficient to splice the `Substitution`s, so we may want
                                // that method to optionally take parent `Substitution` as we already know them at
                                // this point (`trait_ref.substitution`).
                                let substitution = self.substs_from_path_segment(
                                    associated_ty.into(),
                                    false,
                                    None,
                                    true,
                                );
                                let args = crate::next_solver::GenericArgs::new_from_iter(
                                    self.ctx.interner,
                                    trait_ref
                                        .args
                                        .iter()
                                        .chain(substitution.iter().skip(trait_ref.args.len())),
                                );
                                Ty::new_alias(
                                    self.ctx.interner,
                                    AliasTyKind::Projection,
                                    AliasTy::new_from_args(
                                        self.ctx.interner,
                                        associated_ty.into(),
                                        args,
                                    ),
                                )
                            }
                            None => {
                                // FIXME: report error (associated type not found)
                                Ty::new_error(self.ctx.interner, ErrorGuaranteed)
                            }
                        }
                    }
                    0 => {
                        // Trait object type without dyn; this should be handled in upstream. See
                        // `lower_path()`.
                        stdx::never!("unexpected fully resolved trait path");
                        Ty::new_error(self.ctx.interner, ErrorGuaranteed)
                    }
                    _ => {
                        // FIXME report error (ambiguous associated type)
                        Ty::new_error(self.ctx.interner, ErrorGuaranteed)
                    }
                };
                return (ty, None);
            }
            TypeNs::GenericParam(param_id) => {
                let generics = self.ctx.generics();
                let idx = generics.type_or_const_param_idx(param_id.into());
                match idx {
                    None => {
                        never!("no matching generics");
                        Ty::new_error(self.ctx.interner, ErrorGuaranteed)
                    }
                    Some(idx) => {
                        let (pidx, param) = generics.iter().nth(idx).unwrap();
                        assert_eq!(pidx, param_id.into());
                        let p = match param {
                            GenericParamDataRef::TypeParamData(p) => p,
                            _ => unreachable!(),
                        };
                        Ty::new_param(
                            self.ctx.interner,
                            param_id,
                            idx as u32,
                            p.name
                                .as_ref()
                                .map_or_else(|| sym::MISSING_NAME.clone(), |p| p.symbol().clone()),
                        )
                    }
                }
            }
            TypeNs::SelfType(impl_id) => self.ctx.db.impl_self_ty_ns(impl_id).skip_binder(),
            TypeNs::AdtSelfType(adt) => {
                let args = crate::next_solver::GenericArgs::identity_for_item(
                    self.ctx.interner,
                    adt.into(),
                );
                Ty::new_adt(self.ctx.interner, AdtDef::new(adt, self.ctx.interner), args)
            }

            TypeNs::AdtId(it) => self.lower_path_inner(it.into(), infer_args),
            TypeNs::BuiltinType(it) => self.lower_path_inner(it.into(), infer_args),
            TypeNs::TypeAliasId(it) => self.lower_path_inner(it.into(), infer_args),
            // FIXME: report error
            TypeNs::EnumVariantId(_) | TypeNs::ModuleId(_) => {
                return (Ty::new_error(self.ctx.interner, ErrorGuaranteed), None);
            }
        };

        tracing::debug!(?ty);

        self.skip_resolved_segment();
        self.lower_ty_relative_path(ty, Some(resolution))
    }

    fn handle_type_ns_resolution(&mut self, resolution: &TypeNs) {
        let mut prohibit_generics_on_resolved = |reason| {
            if self.current_or_prev_segment.args_and_bindings.is_some() {
                let segment = self.current_segment_u32();
                self.on_diagnostic(PathLoweringDiagnostic::GenericArgsProhibited {
                    segment,
                    reason,
                });
            }
        };

        match resolution {
            TypeNs::SelfType(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::SelfTy)
            }
            TypeNs::GenericParam(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::TyParam)
            }
            TypeNs::AdtSelfType(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::SelfTy)
            }
            TypeNs::BuiltinType(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::PrimitiveTy)
            }
            TypeNs::ModuleId(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::Module)
            }
            TypeNs::AdtId(_)
            | TypeNs::EnumVariantId(_)
            | TypeNs::TypeAliasId(_)
            | TypeNs::TraitId(_) => {}
        }
    }

    pub(crate) fn resolve_path_in_type_ns_fully(&mut self) -> Option<TypeNs> {
        let (res, unresolved) = self.resolve_path_in_type_ns()?;
        if unresolved.is_some() {
            return None;
        }
        Some(res)
    }

    #[tracing::instrument(skip(self), ret)]
    pub(crate) fn resolve_path_in_type_ns(&mut self) -> Option<(TypeNs, Option<usize>)> {
        let (resolution, remaining_index, _, prefix_info) =
            self.ctx.resolver.resolve_path_in_type_ns_with_prefix_info(self.ctx.db, self.path)?;

        let segments = self.segments;
        if segments.is_empty() || matches!(self.path, Path::LangItem(..)) {
            // `segments.is_empty()` can occur with `self`.
            return Some((resolution, remaining_index));
        }

        let (module_segments, resolved_segment_idx, enum_segment) = match remaining_index {
            None if prefix_info.enum_variant => {
                (segments.strip_last_two(), segments.len() - 1, Some(segments.len() - 2))
            }
            None => (segments.strip_last(), segments.len() - 1, None),
            Some(i) => (segments.take(i - 1), i - 1, None),
        };

        self.current_segment_idx = resolved_segment_idx;
        self.current_or_prev_segment =
            segments.get(resolved_segment_idx).expect("should have resolved segment");

        if matches!(self.path, Path::BarePath(..)) {
            // Bare paths cannot have generics, so skip them as an optimization.
            return Some((resolution, remaining_index));
        }

        for (i, mod_segment) in module_segments.iter().enumerate() {
            if mod_segment.args_and_bindings.is_some() {
                self.on_diagnostic(PathLoweringDiagnostic::GenericArgsProhibited {
                    segment: i as u32,
                    reason: GenericArgsProhibitedReason::Module,
                });
            }
        }

        if let Some(enum_segment) = enum_segment
            && segments.get(enum_segment).is_some_and(|it| it.args_and_bindings.is_some())
            && segments.get(enum_segment + 1).is_some_and(|it| it.args_and_bindings.is_some())
        {
            self.on_diagnostic(PathLoweringDiagnostic::GenericArgsProhibited {
                segment: (enum_segment + 1) as u32,
                reason: GenericArgsProhibitedReason::EnumVariant,
            });
        }

        self.handle_type_ns_resolution(&resolution);

        Some((resolution, remaining_index))
    }

    pub(crate) fn resolve_path_in_value_ns(
        &mut self,
        hygiene_id: HygieneId,
    ) -> Option<ResolveValueResult> {
        let (res, prefix_info) = self.ctx.resolver.resolve_path_in_value_ns_with_prefix_info(
            self.ctx.db,
            self.path,
            hygiene_id,
        )?;

        let segments = self.segments;
        if segments.is_empty() || matches!(self.path, Path::LangItem(..)) {
            // `segments.is_empty()` can occur with `self`.
            return Some(res);
        }

        let (mod_segments, enum_segment, resolved_segment_idx) = match res {
            ResolveValueResult::Partial(_, unresolved_segment, _) => {
                (segments.take(unresolved_segment - 1), None, unresolved_segment - 1)
            }
            ResolveValueResult::ValueNs(ValueNs::EnumVariantId(_), _)
                if prefix_info.enum_variant =>
            {
                (segments.strip_last_two(), segments.len().checked_sub(2), segments.len() - 1)
            }
            ResolveValueResult::ValueNs(..) => (segments.strip_last(), None, segments.len() - 1),
        };

        self.current_segment_idx = resolved_segment_idx;
        self.current_or_prev_segment =
            segments.get(resolved_segment_idx).expect("should have resolved segment");

        for (i, mod_segment) in mod_segments.iter().enumerate() {
            if mod_segment.args_and_bindings.is_some() {
                self.on_diagnostic(PathLoweringDiagnostic::GenericArgsProhibited {
                    segment: i as u32,
                    reason: GenericArgsProhibitedReason::Module,
                });
            }
        }

        if let Some(enum_segment) = enum_segment
            && segments.get(enum_segment).is_some_and(|it| it.args_and_bindings.is_some())
            && segments.get(enum_segment + 1).is_some_and(|it| it.args_and_bindings.is_some())
        {
            self.on_diagnostic(PathLoweringDiagnostic::GenericArgsProhibited {
                segment: (enum_segment + 1) as u32,
                reason: GenericArgsProhibitedReason::EnumVariant,
            });
        }

        match &res {
            ResolveValueResult::ValueNs(resolution, _) => {
                let resolved_segment_idx = self.current_segment_u32();
                let resolved_segment = self.current_or_prev_segment;

                let mut prohibit_generics_on_resolved = |reason| {
                    if resolved_segment.args_and_bindings.is_some() {
                        self.on_diagnostic(PathLoweringDiagnostic::GenericArgsProhibited {
                            segment: resolved_segment_idx,
                            reason,
                        });
                    }
                };

                match resolution {
                    ValueNs::ImplSelf(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::SelfTy)
                    }
                    // FIXME: rustc generates E0107 (incorrect number of generic arguments) and not
                    // E0109 (generic arguments provided for a type that doesn't accept them) for
                    // consts and statics, presumably as a defense against future in which consts
                    // and statics can be generic, or just because it was easier for rustc implementors.
                    // That means we'll show the wrong error code. Because of us it's easier to do it
                    // this way :)
                    ValueNs::GenericParam(_) | ValueNs::ConstId(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::Const)
                    }
                    ValueNs::StaticId(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::Static)
                    }
                    ValueNs::FunctionId(_) | ValueNs::StructId(_) | ValueNs::EnumVariantId(_) => {}
                    ValueNs::LocalBinding(_) => {}
                }
            }
            ResolveValueResult::Partial(resolution, _, _) => {
                self.handle_type_ns_resolution(resolution);
            }
        };
        Some(res)
    }

    #[tracing::instrument(skip(self), ret)]
    fn select_associated_type(&mut self, res: Option<TypeNs>) -> Ty<'db> {
        let interner = self.ctx.interner;
        let Some(res) = res else {
            return Ty::new_error(self.ctx.interner, ErrorGuaranteed);
        };
        let db = self.ctx.db;
        let def = self.ctx.def;
        let segment = self.current_or_prev_segment;
        let assoc_name = segment.name;
        let mut check_alias = |name: &Name, t: TraitRef<'db>, associated_ty: TypeAliasId| {
            if name != assoc_name {
                return None;
            }

            // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
            // generic params. It's inefficient to splice the `Substitution`s, so we may want
            // that method to optionally take parent `Substitution` as we already know them at
            // this point (`t.substitution`).
            let substs = self.substs_from_path_segment(associated_ty.into(), false, None, true);

            let substs = crate::next_solver::GenericArgs::new_from_iter(
                interner,
                t.args.iter().chain(substs.iter().skip(t.args.len())),
            );

            Some(Ty::new_alias(
                interner,
                AliasTyKind::Projection,
                AliasTy::new(interner, associated_ty.into(), substs),
            ))
        };
        named_associated_type_shorthand_candidates(
            interner,
            def,
            res,
            Some(assoc_name.clone()),
            check_alias,
        )
        .unwrap_or_else(|| Ty::new_error(interner, ErrorGuaranteed))
    }

    fn lower_path_inner(&mut self, typeable: TyDefId, infer_args: bool) -> Ty<'db> {
        let generic_def = match typeable {
            TyDefId::BuiltinType(builtinty) => return builtin(self.ctx.interner, builtinty),
            TyDefId::AdtId(it) => it.into(),
            TyDefId::TypeAliasId(it) => it.into(),
        };
        let args = self.substs_from_path_segment(generic_def, infer_args, None, false);
        let ty = ty_query(self.ctx.db, typeable);
        ty.instantiate(self.ctx.interner, args)
    }

    /// Collect generic arguments from a path into a `Substs`. See also
    /// `create_substs_for_ast_path` and `def_to_ty` in rustc.
    pub(crate) fn substs_from_path(
        &mut self,
        // Note that we don't call `db.value_type(resolved)` here,
        // `ValueTyDefId` is just a convenient way to pass generics and
        // special-case enum variants
        resolved: ValueTyDefId,
        infer_args: bool,
        lowering_assoc_type_generics: bool,
    ) -> crate::next_solver::GenericArgs<'db> {
        let interner = self.ctx.interner;
        let prev_current_segment_idx = self.current_segment_idx;
        let prev_current_segment = self.current_or_prev_segment;

        let generic_def = match resolved {
            ValueTyDefId::FunctionId(it) => it.into(),
            ValueTyDefId::StructId(it) => it.into(),
            ValueTyDefId::UnionId(it) => it.into(),
            ValueTyDefId::ConstId(it) => it.into(),
            ValueTyDefId::StaticId(_) => {
                return crate::next_solver::GenericArgs::new_from_iter(interner, []);
            }
            ValueTyDefId::EnumVariantId(var) => {
                // the generic args for an enum variant may be either specified
                // on the segment referring to the enum, or on the segment
                // referring to the variant. So `Option::<T>::None` and
                // `Option::None::<T>` are both allowed (though the former is
                // FIXME: This isn't strictly correct, enum variants may be used not through the enum
                // (via `use Enum::Variant`). The resolver returns whether they were, but we don't have its result
                // available here. The worst that can happen is that we will show some confusing diagnostics to the user,
                // if generics exist on the module and they don't match with the variant.
                // preferred). See also `def_ids_for_path_segments` in rustc.
                //
                // `wrapping_sub(1)` will return a number which `get` will return None for if current_segment_idx<2.
                // This simplifies the code a bit.
                let penultimate_idx = self.current_segment_idx.wrapping_sub(1);
                let penultimate = self.segments.get(penultimate_idx);
                if let Some(penultimate) = penultimate
                    && self.current_or_prev_segment.args_and_bindings.is_none()
                    && penultimate.args_and_bindings.is_some()
                {
                    self.current_segment_idx = penultimate_idx;
                    self.current_or_prev_segment = penultimate;
                }
                var.lookup(self.ctx.db).parent.into()
            }
        };
        let result = self.substs_from_path_segment(
            generic_def,
            infer_args,
            None,
            lowering_assoc_type_generics,
        );
        self.current_segment_idx = prev_current_segment_idx;
        self.current_or_prev_segment = prev_current_segment;
        result
    }

    pub(crate) fn substs_from_path_segment(
        &mut self,
        def: GenericDefId,
        infer_args: bool,
        explicit_self_ty: Option<Ty<'db>>,
        lowering_assoc_type_generics: bool,
    ) -> crate::next_solver::GenericArgs<'db> {
        let mut lifetime_elision = self.ctx.lifetime_elision.clone();

        if let Some(args) = self.current_or_prev_segment.args_and_bindings
            && args.parenthesized != GenericArgsParentheses::No
        {
            let prohibit_parens = match def {
                GenericDefId::TraitId(trait_) => {
                    // RTN is prohibited anyways if we got here.
                    let is_rtn = args.parenthesized == GenericArgsParentheses::ReturnTypeNotation;
                    let is_fn_trait = self
                        .ctx
                        .db
                        .trait_signature(trait_)
                        .flags
                        .contains(TraitFlags::RUSTC_PAREN_SUGAR);
                    is_rtn || !is_fn_trait
                }
                _ => true,
            };

            if prohibit_parens {
                let segment = self.current_segment_u32();
                self.on_diagnostic(
                    PathLoweringDiagnostic::ParenthesizedGenericArgsWithoutFnTrait { segment },
                );

                return unknown_subst(self.ctx.interner, def);
            }

            // `Fn()`-style generics are treated like functions for the purpose of lifetime elision.
            lifetime_elision =
                LifetimeElisionKind::AnonymousCreateParameter { report_in_path: false };
        }

        self.substs_from_args_and_bindings(
            self.current_or_prev_segment.args_and_bindings,
            def,
            infer_args,
            explicit_self_ty,
            PathGenericsSource::Segment(self.current_segment_u32()),
            lowering_assoc_type_generics,
            lifetime_elision,
        )
    }

    pub(super) fn substs_from_args_and_bindings(
        &mut self,
        args_and_bindings: Option<&GenericArgs>,
        def: GenericDefId,
        infer_args: bool,
        explicit_self_ty: Option<Ty<'db>>,
        generics_source: PathGenericsSource,
        lowering_assoc_type_generics: bool,
        lifetime_elision: LifetimeElisionKind<'db>,
    ) -> crate::next_solver::GenericArgs<'db> {
        struct LowererCtx<'a, 'b, 'c, 'db> {
            ctx: &'a mut PathLoweringContext<'b, 'c, 'db>,
            generics_source: PathGenericsSource,
        }

        impl<'db> GenericArgsLowerer<'db> for LowererCtx<'_, '_, '_, 'db> {
            fn report_len_mismatch(
                &mut self,
                def: GenericDefId,
                provided_count: u32,
                expected_count: u32,
                kind: IncorrectGenericsLenKind,
            ) {
                self.ctx.on_diagnostic(PathLoweringDiagnostic::IncorrectGenericsLen {
                    generics_source: self.generics_source,
                    provided_count,
                    expected_count,
                    kind,
                    def,
                });
            }

            fn report_arg_mismatch(
                &mut self,
                param_id: GenericParamId,
                arg_idx: u32,
                has_self_arg: bool,
            ) {
                self.ctx.on_diagnostic(PathLoweringDiagnostic::IncorrectGenericsOrder {
                    generics_source: self.generics_source,
                    param_id,
                    arg_idx,
                    has_self_arg,
                });
            }

            fn provided_kind(
                &mut self,
                param_id: GenericParamId,
                param: GenericParamDataRef<'_>,
                arg: &GenericArg,
            ) -> crate::next_solver::GenericArg<'db> {
                match (param, arg) {
                    (GenericParamDataRef::LifetimeParamData(_), GenericArg::Lifetime(lifetime)) => {
                        self.ctx.ctx.lower_lifetime(*lifetime).into()
                    }
                    (GenericParamDataRef::TypeParamData(_), GenericArg::Type(type_ref)) => {
                        self.ctx.ctx.lower_ty(*type_ref).into()
                    }
                    (GenericParamDataRef::ConstParamData(_), GenericArg::Const(konst)) => {
                        let GenericParamId::ConstParamId(const_id) = param_id else {
                            unreachable!("non-const param ID for const param");
                        };
                        self.ctx
                            .ctx
                            .lower_const(konst, const_param_ty_query(self.ctx.ctx.db, const_id))
                            .into()
                    }
                    _ => unreachable!("unmatching param kinds were passed to `provided_kind()`"),
                }
            }

            fn provided_type_like_const(
                &mut self,
                const_ty: Ty<'db>,
                arg: TypeLikeConst<'_>,
            ) -> crate::next_solver::Const<'db> {
                match arg {
                    TypeLikeConst::Path(path) => self.ctx.ctx.lower_path_as_const(path, const_ty),
                    TypeLikeConst::Infer => unknown_const(const_ty),
                }
            }

            fn inferred_kind(
                &mut self,
                def: GenericDefId,
                param_id: GenericParamId,
                param: GenericParamDataRef<'_>,
                infer_args: bool,
                preceding_args: &[crate::next_solver::GenericArg<'db>],
            ) -> crate::next_solver::GenericArg<'db> {
                let default = || {
                    self.ctx.ctx.db.generic_defaults(def).get(preceding_args.len()).map(|default| {
                        convert_binder_to_early_binder(
                            self.ctx.ctx.interner,
                            def,
                            default.to_nextsolver(self.ctx.ctx.interner),
                        )
                        .instantiate(self.ctx.ctx.interner, preceding_args)
                    })
                };
                match param {
                    GenericParamDataRef::LifetimeParamData(_) => {
                        Region::new(self.ctx.ctx.interner, rustc_type_ir::ReError(ErrorGuaranteed))
                            .into()
                    }
                    GenericParamDataRef::TypeParamData(param) => {
                        if !infer_args
                            && param.default.is_some()
                            && let Some(default) = default()
                        {
                            return default;
                        }
                        Ty::new_error(self.ctx.ctx.interner, ErrorGuaranteed).into()
                    }
                    GenericParamDataRef::ConstParamData(param) => {
                        if !infer_args
                            && param.default.is_some()
                            && let Some(default) = default()
                        {
                            return default;
                        }
                        let GenericParamId::ConstParamId(const_id) = param_id else {
                            unreachable!("non-const param ID for const param");
                        };
                        unknown_const_as_generic(const_param_ty_query(self.ctx.ctx.db, const_id))
                    }
                }
            }

            fn parent_arg(
                &mut self,
                param_id: GenericParamId,
            ) -> crate::next_solver::GenericArg<'db> {
                match param_id {
                    GenericParamId::TypeParamId(_) => {
                        Ty::new_error(self.ctx.ctx.interner, ErrorGuaranteed).into()
                    }
                    GenericParamId::ConstParamId(const_id) => {
                        unknown_const_as_generic(const_param_ty_query(self.ctx.ctx.db, const_id))
                    }
                    GenericParamId::LifetimeParamId(_) => {
                        Region::new(self.ctx.ctx.interner, rustc_type_ir::ReError(ErrorGuaranteed))
                            .into()
                    }
                }
            }

            fn report_elided_lifetimes_in_path(
                &mut self,
                def: GenericDefId,
                expected_count: u32,
                hard_error: bool,
            ) {
                self.ctx.on_diagnostic(PathLoweringDiagnostic::ElidedLifetimesInPath {
                    generics_source: self.generics_source,
                    def,
                    expected_count,
                    hard_error,
                });
            }

            fn report_elision_failure(&mut self, def: GenericDefId, expected_count: u32) {
                self.ctx.on_diagnostic(PathLoweringDiagnostic::ElisionFailure {
                    generics_source: self.generics_source,
                    def,
                    expected_count,
                });
            }

            fn report_missing_lifetime(&mut self, def: GenericDefId, expected_count: u32) {
                self.ctx.on_diagnostic(PathLoweringDiagnostic::MissingLifetime {
                    generics_source: self.generics_source,
                    def,
                    expected_count,
                });
            }
        }

        substs_from_args_and_bindings(
            self.ctx.db,
            self.ctx.store,
            args_and_bindings,
            def,
            infer_args,
            lifetime_elision,
            lowering_assoc_type_generics,
            explicit_self_ty,
            &mut LowererCtx { ctx: self, generics_source },
        )
    }

    pub(crate) fn lower_trait_ref_from_resolved_path(
        &mut self,
        resolved: TraitId,
        explicit_self_ty: Ty<'db>,
    ) -> TraitRef<'db> {
        let args = self.trait_ref_substs_from_path(resolved, explicit_self_ty);
        TraitRef::new_from_args(self.ctx.interner, resolved.into(), args)
    }

    fn trait_ref_substs_from_path(
        &mut self,
        resolved: TraitId,
        explicit_self_ty: Ty<'db>,
    ) -> crate::next_solver::GenericArgs<'db> {
        self.substs_from_path_segment(resolved.into(), false, Some(explicit_self_ty), false)
    }

    pub(super) fn assoc_type_bindings_from_type_bound<'c>(
        mut self,
        trait_ref: TraitRef<'db>,
    ) -> Option<impl Iterator<Item = Clause<'db>> + use<'a, 'b, 'c, 'db>> {
        let interner = self.ctx.interner;
        self.current_or_prev_segment.args_and_bindings.map(|args_and_bindings| {
            args_and_bindings.bindings.iter().enumerate().flat_map(move |(binding_idx, binding)| {
                let found = associated_type_by_name_including_super_traits(
                    self.ctx.db,
                    trait_ref,
                    &binding.name,
                );
                let (super_trait_ref, associated_ty) = match found {
                    None => return SmallVec::new(),
                    Some(t) => t,
                };
                let args =
                    self.with_lifetime_elision(LifetimeElisionKind::AnonymousReportError, |this| {
                        // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                        // generic params. It's inefficient to splice the `Substitution`s, so we may want
                        // that method to optionally take parent `Substitution` as we already know them at
                        // this point (`super_trait_ref.substitution`).
                        this.substs_from_args_and_bindings(
                            binding.args.as_ref(),
                            associated_ty.into(),
                            false, // this is not relevant
                            Some(super_trait_ref.self_ty()),
                            PathGenericsSource::AssocType {
                                segment: this.current_segment_u32(),
                                assoc_type: binding_idx as u32,
                            },
                            false,
                            this.ctx.lifetime_elision.clone(),
                        )
                    });
                let args = crate::next_solver::GenericArgs::new_from_iter(
                    interner,
                    super_trait_ref.args.iter().chain(args.iter().skip(super_trait_ref.args.len())),
                );
                let projection_term =
                    AliasTerm::new_from_args(interner, associated_ty.into(), args);
                let mut predicates: SmallVec<[_; 1]> = SmallVec::with_capacity(
                    binding.type_ref.as_ref().map_or(0, |_| 1) + binding.bounds.len(),
                );
                if let Some(type_ref) = binding.type_ref {
                    match (&self.ctx.store[type_ref], self.ctx.impl_trait_mode.mode) {
                        (TypeRef::ImplTrait(_), ImplTraitLoweringMode::Disallowed) => (),
                        (_, ImplTraitLoweringMode::Disallowed | ImplTraitLoweringMode::Opaque) => {
                            let ty = self.ctx.lower_ty(type_ref);
                            let pred = Clause(Predicate::new(
                                interner,
                                Binder::dummy(rustc_type_ir::PredicateKind::Clause(
                                    rustc_type_ir::ClauseKind::Projection(ProjectionPredicate {
                                        projection_term,
                                        term: ty.into(),
                                    }),
                                )),
                            ));
                            predicates.push(pred);
                        }
                    }
                }
                for bound in binding.bounds.iter() {
                    predicates.extend(self.ctx.lower_type_bound(
                        bound,
                        Ty::new_alias(
                            self.ctx.interner,
                            AliasTyKind::Projection,
                            AliasTy::new_from_args(self.ctx.interner, associated_ty.into(), args),
                        ),
                        false,
                    ));
                }
                predicates
            })
        })
    }
}

/// A const that were parsed like a type.
pub(crate) enum TypeLikeConst<'a> {
    Infer,
    Path(&'a Path),
}

pub(crate) trait GenericArgsLowerer<'db> {
    fn report_elided_lifetimes_in_path(
        &mut self,
        def: GenericDefId,
        expected_count: u32,
        hard_error: bool,
    );

    fn report_elision_failure(&mut self, def: GenericDefId, expected_count: u32);

    fn report_missing_lifetime(&mut self, def: GenericDefId, expected_count: u32);

    fn report_len_mismatch(
        &mut self,
        def: GenericDefId,
        provided_count: u32,
        expected_count: u32,
        kind: IncorrectGenericsLenKind,
    );

    fn report_arg_mismatch(&mut self, param_id: GenericParamId, arg_idx: u32, has_self_arg: bool);

    fn provided_kind(
        &mut self,
        param_id: GenericParamId,
        param: GenericParamDataRef<'_>,
        arg: &GenericArg,
    ) -> crate::next_solver::GenericArg<'db>;

    fn provided_type_like_const(&mut self, const_ty: Ty<'db>, arg: TypeLikeConst<'_>)
    -> Const<'db>;

    fn inferred_kind(
        &mut self,
        def: GenericDefId,
        param_id: GenericParamId,
        param: GenericParamDataRef<'_>,
        infer_args: bool,
        preceding_args: &[crate::next_solver::GenericArg<'db>],
    ) -> crate::next_solver::GenericArg<'db>;

    fn parent_arg(&mut self, param_id: GenericParamId) -> crate::next_solver::GenericArg<'db>;
}

/// Returns true if there was an error.
fn check_generic_args_len<'db>(
    args_and_bindings: Option<&GenericArgs>,
    def: GenericDefId,
    def_generics: &Generics,
    infer_args: bool,
    lifetime_elision: &LifetimeElisionKind<'db>,
    lowering_assoc_type_generics: bool,
    ctx: &mut impl GenericArgsLowerer<'db>,
) -> bool {
    let mut had_error = false;

    let (mut provided_lifetimes_count, mut provided_types_and_consts_count) = (0usize, 0usize);
    if let Some(args_and_bindings) = args_and_bindings {
        let args_no_self = &args_and_bindings.args[usize::from(args_and_bindings.has_self_type)..];
        for arg in args_no_self {
            match arg {
                GenericArg::Lifetime(_) => provided_lifetimes_count += 1,
                GenericArg::Type(_) | GenericArg::Const(_) => provided_types_and_consts_count += 1,
            }
        }
    }

    let lifetime_args_len = def_generics.len_lifetimes_self();
    if provided_lifetimes_count == 0 && lifetime_args_len > 0 && !lowering_assoc_type_generics {
        // In generic associated types, we never allow inferring the lifetimes.
        match lifetime_elision {
            &LifetimeElisionKind::AnonymousCreateParameter { report_in_path } => {
                ctx.report_elided_lifetimes_in_path(def, lifetime_args_len as u32, report_in_path);
                had_error |= report_in_path;
            }
            LifetimeElisionKind::AnonymousReportError => {
                ctx.report_missing_lifetime(def, lifetime_args_len as u32);
                had_error = true
            }
            LifetimeElisionKind::ElisionFailure => {
                ctx.report_elision_failure(def, lifetime_args_len as u32);
                had_error = true;
            }
            LifetimeElisionKind::StaticIfNoLifetimeInScope { only_lint: _ } => {
                // FIXME: Check there are other lifetimes in scope, and error/lint.
            }
            LifetimeElisionKind::Elided(_) => {
                ctx.report_elided_lifetimes_in_path(def, lifetime_args_len as u32, false);
            }
            LifetimeElisionKind::Infer => {
                // Allow eliding lifetimes.
            }
        }
    } else if lifetime_args_len != provided_lifetimes_count {
        ctx.report_len_mismatch(
            def,
            provided_lifetimes_count as u32,
            lifetime_args_len as u32,
            IncorrectGenericsLenKind::Lifetimes,
        );
        had_error = true;
    }

    let defaults_count =
        def_generics.iter_self_type_or_consts().filter(|(_, param)| param.has_default()).count();
    let named_type_and_const_params_count = def_generics
        .iter_self_type_or_consts()
        .filter(|(_, param)| match param {
            TypeOrConstParamData::TypeParamData(param) => {
                param.provenance == TypeParamProvenance::TypeParamList
            }
            TypeOrConstParamData::ConstParamData(_) => true,
        })
        .count();
    let expected_max = named_type_and_const_params_count;
    let expected_min =
        if infer_args { 0 } else { named_type_and_const_params_count - defaults_count };
    if provided_types_and_consts_count < expected_min
        || expected_max < provided_types_and_consts_count
    {
        ctx.report_len_mismatch(
            def,
            provided_types_and_consts_count as u32,
            named_type_and_const_params_count as u32,
            IncorrectGenericsLenKind::TypesAndConsts,
        );
        had_error = true;
    }

    had_error
}

pub(crate) fn substs_from_args_and_bindings<'db>(
    db: &'db dyn HirDatabase,
    store: &ExpressionStore,
    args_and_bindings: Option<&GenericArgs>,
    def: GenericDefId,
    mut infer_args: bool,
    lifetime_elision: LifetimeElisionKind<'db>,
    lowering_assoc_type_generics: bool,
    explicit_self_ty: Option<Ty<'db>>,
    ctx: &mut impl GenericArgsLowerer<'db>,
) -> crate::next_solver::GenericArgs<'db> {
    let interner = DbInterner::new_with(db, None, None);

    tracing::debug!(?args_and_bindings);

    // Order is
    // - Parent parameters
    // - Optional Self parameter
    // - Lifetime parameters
    // - Type or Const parameters
    let def_generics = generics(db, def);
    let args_slice = args_and_bindings.map(|it| &*it.args).unwrap_or_default();

    // We do not allow inference if there are specified args, i.e. we do not allow partial inference.
    let has_non_lifetime_args =
        args_slice.iter().any(|arg| !matches!(arg, GenericArg::Lifetime(_)));
    infer_args &= !has_non_lifetime_args;

    let had_count_error = check_generic_args_len(
        args_and_bindings,
        def,
        &def_generics,
        infer_args,
        &lifetime_elision,
        lowering_assoc_type_generics,
        ctx,
    );

    let mut substs = Vec::with_capacity(def_generics.len());

    substs.extend(def_generics.iter_parent_id().map(|id| ctx.parent_arg(id)));

    let mut args = args_slice.iter().enumerate().peekable();
    let mut params = def_generics.iter_self().peekable();

    // If we encounter a type or const when we expect a lifetime, we infer the lifetimes.
    // If we later encounter a lifetime, we know that the arguments were provided in the
    // wrong order. `force_infer_lt` records the type or const that forced lifetimes to be
    // inferred, so we can use it for diagnostics later.
    let mut force_infer_lt = None;

    let has_self_arg = args_and_bindings.is_some_and(|it| it.has_self_type);
    // First, handle `Self` parameter. Consume it from the args if provided, otherwise from `explicit_self_ty`,
    // and lastly infer it.
    if let Some(&(
        self_param_id,
        self_param @ GenericParamDataRef::TypeParamData(TypeParamData {
            provenance: TypeParamProvenance::TraitSelf,
            ..
        }),
    )) = params.peek()
    {
        let self_ty = if has_self_arg {
            let (_, self_ty) = args.next().expect("has_self_type=true, should have Self type");
            ctx.provided_kind(self_param_id, self_param, self_ty)
        } else {
            explicit_self_ty.map(|it| it.into()).unwrap_or_else(|| {
                ctx.inferred_kind(def, self_param_id, self_param, infer_args, &substs)
            })
        };
        params.next();
        substs.push(self_ty);
    }

    loop {
        // We're going to iterate through the generic arguments that the user
        // provided, matching them with the generic parameters we expect.
        // Mismatches can occur as a result of elided lifetimes, or for malformed
        // input. We try to handle both sensibly.
        match (args.peek(), params.peek()) {
            (Some(&(arg_idx, arg)), Some(&(param_id, param))) => match (arg, param) {
                (GenericArg::Type(_), GenericParamDataRef::TypeParamData(type_param))
                    if type_param.provenance == TypeParamProvenance::ArgumentImplTrait =>
                {
                    // Do not allow specifying `impl Trait` explicitly. We already err at that, but if we won't handle it here
                    // we will handle it as if it was specified, instead of inferring it.
                    substs.push(ctx.inferred_kind(def, param_id, param, infer_args, &substs));
                    params.next();
                }
                (GenericArg::Lifetime(_), GenericParamDataRef::LifetimeParamData(_))
                | (GenericArg::Type(_), GenericParamDataRef::TypeParamData(_))
                | (GenericArg::Const(_), GenericParamDataRef::ConstParamData(_)) => {
                    substs.push(ctx.provided_kind(param_id, param, arg));
                    args.next();
                    params.next();
                }
                (
                    GenericArg::Type(_) | GenericArg::Const(_),
                    GenericParamDataRef::LifetimeParamData(_),
                ) => {
                    // We expected a lifetime argument, but got a type or const
                    // argument. That means we're inferring the lifetime.
                    substs.push(ctx.inferred_kind(def, param_id, param, infer_args, &substs));
                    params.next();
                    force_infer_lt = Some((arg_idx as u32, param_id));
                }
                (GenericArg::Type(type_ref), GenericParamDataRef::ConstParamData(_)) => {
                    if let Some(konst) = type_looks_like_const(store, *type_ref) {
                        let GenericParamId::ConstParamId(param_id) = param_id else {
                            panic!("unmatching param kinds");
                        };
                        let const_ty = const_param_ty_query(db, param_id);
                        substs.push(ctx.provided_type_like_const(const_ty, konst).into());
                        args.next();
                        params.next();
                    } else {
                        // See the `_ => { ... }` branch.
                        if !had_count_error {
                            ctx.report_arg_mismatch(param_id, arg_idx as u32, has_self_arg);
                        }
                        while args.next().is_some() {}
                    }
                }
                _ => {
                    // We expected one kind of parameter, but the user provided
                    // another. This is an error. However, if we already know that
                    // the arguments don't match up with the parameters, we won't issue
                    // an additional error, as the user already knows what's wrong.
                    if !had_count_error {
                        ctx.report_arg_mismatch(param_id, arg_idx as u32, has_self_arg);
                    }

                    // We've reported the error, but we want to make sure that this
                    // problem doesn't bubble down and create additional, irrelevant
                    // errors. In this case, we're simply going to ignore the argument
                    // and any following arguments. The rest of the parameters will be
                    // inferred.
                    while args.next().is_some() {}
                }
            },

            (Some(&(_, arg)), None) => {
                // We should never be able to reach this point with well-formed input.
                // There are two situations in which we can encounter this issue.
                //
                //  1. The number of arguments is incorrect. In this case, an error
                //     will already have been emitted, and we can ignore it.
                //  2. We've inferred some lifetimes, which have been provided later (i.e.
                //     after a type or const). We want to throw an error in this case.
                if !had_count_error {
                    assert!(
                        matches!(arg, GenericArg::Lifetime(_)),
                        "the only possible situation here is incorrect lifetime order"
                    );
                    let (provided_arg_idx, param_id) =
                        force_infer_lt.expect("lifetimes ought to have been inferred");
                    ctx.report_arg_mismatch(param_id, provided_arg_idx, has_self_arg);
                }

                break;
            }

            (None, Some(&(param_id, param))) => {
                // If there are fewer arguments than parameters, it means we're inferring the remaining arguments.
                let param = if let GenericParamId::LifetimeParamId(_) = param_id {
                    match &lifetime_elision {
                        LifetimeElisionKind::ElisionFailure
                        | LifetimeElisionKind::AnonymousCreateParameter { report_in_path: true }
                        | LifetimeElisionKind::AnonymousReportError => {
                            assert!(had_count_error);
                            ctx.inferred_kind(def, param_id, param, infer_args, &substs)
                        }
                        LifetimeElisionKind::StaticIfNoLifetimeInScope { only_lint: _ } => {
                            Region::new_static(interner).into()
                        }
                        LifetimeElisionKind::Elided(lifetime) => (*lifetime).into(),
                        LifetimeElisionKind::AnonymousCreateParameter { report_in_path: false }
                        | LifetimeElisionKind::Infer => {
                            // FIXME: With `AnonymousCreateParameter`, we need to create a new lifetime parameter here
                            // (but this will probably be done in hir-def lowering instead).
                            ctx.inferred_kind(def, param_id, param, infer_args, &substs)
                        }
                    }
                } else {
                    ctx.inferred_kind(def, param_id, param, infer_args, &substs)
                };
                substs.push(param);
                params.next();
            }

            (None, None) => break,
        }
    }

    crate::next_solver::GenericArgs::new_from_iter(interner, substs)
}

fn type_looks_like_const(
    store: &ExpressionStore,
    type_ref: TypeRefId,
) -> Option<TypeLikeConst<'_>> {
    // A path/`_` const will be parsed as a type, instead of a const, because when parsing/lowering
    // in hir-def we don't yet know the expected argument kind. rustc does this a bit differently,
    // when lowering to HIR it resolves the path, and if it doesn't resolve to the type namespace
    // it is lowered as a const. Our behavior could deviate from rustc when the value is resolvable
    // in both the type and value namespaces, but I believe we only allow more code.
    let type_ref = &store[type_ref];
    match type_ref {
        TypeRef::Path(path) => Some(TypeLikeConst::Path(path)),
        TypeRef::Placeholder => Some(TypeLikeConst::Infer),
        _ => None,
    }
}

fn unknown_subst<'db>(
    interner: DbInterner<'db>,
    def: impl Into<GenericDefId>,
) -> crate::next_solver::GenericArgs<'db> {
    let params = generics(interner.db(), def.into());
    crate::next_solver::GenericArgs::new_from_iter(
        interner,
        params.iter_id().map(|id| match id {
            GenericParamId::TypeParamId(_) => Ty::new_error(interner, ErrorGuaranteed).into(),
            GenericParamId::ConstParamId(id) => {
                unknown_const_as_generic(const_param_ty_query(interner.db(), id))
            }
            GenericParamId::LifetimeParamId(_) => {
                crate::next_solver::Region::error(interner).into()
            }
        }),
    )
}

pub(crate) fn builtin<'db>(interner: DbInterner<'db>, builtin: BuiltinType) -> Ty<'db> {
    match builtin {
        BuiltinType::Char => Ty::new(interner, rustc_type_ir::TyKind::Char),
        BuiltinType::Bool => Ty::new_bool(interner),
        BuiltinType::Str => Ty::new(interner, rustc_type_ir::TyKind::Str),
        BuiltinType::Int(t) => {
            let int_ty = match primitive::int_ty_from_builtin(t) {
                chalk_ir::IntTy::Isize => rustc_type_ir::IntTy::Isize,
                chalk_ir::IntTy::I8 => rustc_type_ir::IntTy::I8,
                chalk_ir::IntTy::I16 => rustc_type_ir::IntTy::I16,
                chalk_ir::IntTy::I32 => rustc_type_ir::IntTy::I32,
                chalk_ir::IntTy::I64 => rustc_type_ir::IntTy::I64,
                chalk_ir::IntTy::I128 => rustc_type_ir::IntTy::I128,
            };
            Ty::new_int(interner, int_ty)
        }
        BuiltinType::Uint(t) => {
            let uint_ty = match primitive::uint_ty_from_builtin(t) {
                chalk_ir::UintTy::Usize => rustc_type_ir::UintTy::Usize,
                chalk_ir::UintTy::U8 => rustc_type_ir::UintTy::U8,
                chalk_ir::UintTy::U16 => rustc_type_ir::UintTy::U16,
                chalk_ir::UintTy::U32 => rustc_type_ir::UintTy::U32,
                chalk_ir::UintTy::U64 => rustc_type_ir::UintTy::U64,
                chalk_ir::UintTy::U128 => rustc_type_ir::UintTy::U128,
            };
            Ty::new_uint(interner, uint_ty)
        }
        BuiltinType::Float(t) => {
            let float_ty = match primitive::float_ty_from_builtin(t) {
                chalk_ir::FloatTy::F16 => rustc_type_ir::FloatTy::F16,
                chalk_ir::FloatTy::F32 => rustc_type_ir::FloatTy::F32,
                chalk_ir::FloatTy::F64 => rustc_type_ir::FloatTy::F64,
                chalk_ir::FloatTy::F128 => rustc_type_ir::FloatTy::F128,
            };
            Ty::new_float(interner, float_ty)
        }
    }
}
