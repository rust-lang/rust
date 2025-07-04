//! A wrapper around [`TyLoweringContext`] specifically for lowering paths.

use chalk_ir::{BoundVar, cast::Cast, fold::Shift};
use either::Either;
use hir_def::{
    GenericDefId, GenericParamId, Lookup, TraitId,
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
use smallvec::SmallVec;
use stdx::never;

use crate::{
    AliasEq, AliasTy, GenericArgsProhibitedReason, ImplTraitLoweringMode, IncorrectGenericsLenKind,
    Interner, ParamLoweringMode, PathGenericsSource, PathLoweringDiagnostic, ProjectionTy,
    QuantifiedWhereClause, Substitution, TraitRef, Ty, TyBuilder, TyDefId, TyKind,
    TyLoweringContext, ValueTyDefId, WhereClause,
    consteval::{unknown_const, unknown_const_as_generic},
    db::HirDatabase,
    error_lifetime,
    generics::{Generics, generics},
    lower::{LifetimeElisionKind, named_associated_type_shorthand_candidates},
    static_lifetime, to_assoc_type_id, to_chalk_trait_id, to_placeholder_idx,
    utils::associated_type_by_name_including_super_traits,
};

type CallbackData<'a> = Either<
    super::PathDiagnosticCallbackData,
    crate::infer::diagnostics::PathDiagnosticCallbackData<'a>,
>;

// We cannot use `&mut dyn FnMut()` because of lifetime issues, and we don't want to use `Box<dyn FnMut()>`
// because of the allocation, so we create a lifetime-less callback, tailored for our needs.
pub(crate) struct PathDiagnosticCallback<'a> {
    pub(crate) data: CallbackData<'a>,
    pub(crate) callback: fn(&CallbackData<'_>, &mut TyLoweringContext<'_>, PathLoweringDiagnostic),
}

pub(crate) struct PathLoweringContext<'a, 'b> {
    ctx: &'a mut TyLoweringContext<'b>,
    on_diagnostic: PathDiagnosticCallback<'a>,
    path: &'a Path,
    segments: PathSegments<'a>,
    current_segment_idx: usize,
    /// Contains the previous segment if `current_segment_idx == segments.len()`
    current_or_prev_segment: PathSegment<'a>,
}

impl<'a, 'b> PathLoweringContext<'a, 'b> {
    #[inline]
    pub(crate) fn new(
        ctx: &'a mut TyLoweringContext<'b>,
        on_diagnostic: PathDiagnosticCallback<'a>,
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
    pub(crate) fn ty_ctx(&mut self) -> &mut TyLoweringContext<'b> {
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
        lifetime_elision: LifetimeElisionKind,
        f: impl FnOnce(&mut PathLoweringContext<'_, '_>) -> T,
    ) -> T {
        let old_lifetime_elision =
            std::mem::replace(&mut self.ctx.lifetime_elision, lifetime_elision);
        let result = f(self);
        self.ctx.lifetime_elision = old_lifetime_elision;
        result
    }

    pub(crate) fn lower_ty_relative_path(
        &mut self,
        ty: Ty,
        // We need the original resolution to lower `Self::AssocTy` correctly
        res: Option<TypeNs>,
        infer_args: bool,
    ) -> (Ty, Option<TypeNs>) {
        match self.segments.len() - self.current_segment_idx {
            0 => (ty, res),
            1 => {
                // resolve unselected assoc types
                (self.select_associated_type(res, infer_args), None)
            }
            _ => {
                // FIXME report error (ambiguous associated type)
                (TyKind::Error.intern(Interner), None)
            }
        }
    }

    // When calling this, the current segment is the resolved segment (we don't advance it yet).
    pub(crate) fn lower_partly_resolved_path(
        &mut self,
        resolution: TypeNs,
        infer_args: bool,
    ) -> (Ty, Option<TypeNs>) {
        let remaining_segments = self.segments.skip(self.current_segment_idx + 1);

        let ty = match resolution {
            TypeNs::TraitId(trait_) => {
                let ty = match remaining_segments.len() {
                    1 => {
                        let trait_ref = self.lower_trait_ref_from_resolved_path(
                            trait_,
                            TyKind::Error.intern(Interner),
                            infer_args,
                        );

                        self.skip_resolved_segment();
                        let segment = self.current_or_prev_segment;
                        let found =
                            trait_.trait_items(self.ctx.db).associated_type_by_name(segment.name);

                        match found {
                            Some(associated_ty) => {
                                // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                                // generic params. It's inefficient to splice the `Substitution`s, so we may want
                                // that method to optionally take parent `Substitution` as we already know them at
                                // this point (`trait_ref.substitution`).
                                let substitution = self.substs_from_path_segment(
                                    associated_ty.into(),
                                    infer_args,
                                    None,
                                    true,
                                );
                                let substitution = Substitution::from_iter(
                                    Interner,
                                    trait_ref.substitution.iter(Interner).chain(
                                        substitution
                                            .iter(Interner)
                                            .skip(trait_ref.substitution.len(Interner)),
                                    ),
                                );
                                TyKind::Alias(AliasTy::Projection(ProjectionTy {
                                    associated_ty_id: to_assoc_type_id(associated_ty),
                                    substitution,
                                }))
                                .intern(Interner)
                            }
                            None => {
                                // FIXME: report error (associated type not found)
                                TyKind::Error.intern(Interner)
                            }
                        }
                    }
                    0 => {
                        // Trait object type without dyn; this should be handled in upstream. See
                        // `lower_path()`.
                        stdx::never!("unexpected fully resolved trait path");
                        TyKind::Error.intern(Interner)
                    }
                    _ => {
                        // FIXME report error (ambiguous associated type)
                        TyKind::Error.intern(Interner)
                    }
                };
                return (ty, None);
            }
            TypeNs::TraitAliasId(_) => {
                // FIXME(trait_alias): Implement trait alias.
                return (TyKind::Error.intern(Interner), None);
            }
            TypeNs::GenericParam(param_id) => match self.ctx.type_param_mode {
                ParamLoweringMode::Placeholder => {
                    TyKind::Placeholder(to_placeholder_idx(self.ctx.db, param_id.into()))
                }
                ParamLoweringMode::Variable => {
                    let idx = match self.ctx.generics().type_or_const_param_idx(param_id.into()) {
                        None => {
                            never!("no matching generics");
                            return (TyKind::Error.intern(Interner), None);
                        }
                        Some(idx) => idx,
                    };

                    TyKind::BoundVar(BoundVar::new(self.ctx.in_binders, idx))
                }
            }
            .intern(Interner),
            TypeNs::SelfType(impl_id) => {
                let generics = self.ctx.generics();

                match self.ctx.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        // `def` can be either impl itself or item within, and we need impl itself
                        // now.
                        let generics = generics.parent_or_self();
                        let subst = generics.placeholder_subst(self.ctx.db);
                        self.ctx.db.impl_self_ty(impl_id).substitute(Interner, &subst)
                    }
                    ParamLoweringMode::Variable => TyBuilder::impl_self_ty(self.ctx.db, impl_id)
                        .fill_with_bound_vars(self.ctx.in_binders, 0)
                        .build(),
                }
            }
            TypeNs::AdtSelfType(adt) => {
                let generics = generics(self.ctx.db, adt.into());
                let substs = match self.ctx.type_param_mode {
                    ParamLoweringMode::Placeholder => generics.placeholder_subst(self.ctx.db),
                    ParamLoweringMode::Variable => {
                        generics.bound_vars_subst(self.ctx.db, self.ctx.in_binders)
                    }
                };
                self.ctx.db.ty(adt.into()).substitute(Interner, &substs)
            }

            TypeNs::AdtId(it) => self.lower_path_inner(it.into(), infer_args),
            TypeNs::BuiltinType(it) => self.lower_path_inner(it.into(), infer_args),
            TypeNs::TypeAliasId(it) => self.lower_path_inner(it.into(), infer_args),
            // FIXME: report error
            TypeNs::EnumVariantId(_) | TypeNs::ModuleId(_) => {
                return (TyKind::Error.intern(Interner), None);
            }
        };

        self.skip_resolved_segment();
        self.lower_ty_relative_path(ty, Some(resolution), infer_args)
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
            | TypeNs::TraitId(_)
            | TypeNs::TraitAliasId(_) => {}
        }
    }

    pub(crate) fn resolve_path_in_type_ns_fully(&mut self) -> Option<TypeNs> {
        let (res, unresolved) = self.resolve_path_in_type_ns()?;
        if unresolved.is_some() {
            return None;
        }
        Some(res)
    }

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

        if let Some(enum_segment) = enum_segment {
            if segments.get(enum_segment).is_some_and(|it| it.args_and_bindings.is_some())
                && segments.get(enum_segment + 1).is_some_and(|it| it.args_and_bindings.is_some())
            {
                self.on_diagnostic(PathLoweringDiagnostic::GenericArgsProhibited {
                    segment: (enum_segment + 1) as u32,
                    reason: GenericArgsProhibitedReason::EnumVariant,
                });
            }
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

        if let Some(enum_segment) = enum_segment {
            if segments.get(enum_segment).is_some_and(|it| it.args_and_bindings.is_some())
                && segments.get(enum_segment + 1).is_some_and(|it| it.args_and_bindings.is_some())
            {
                self.on_diagnostic(PathLoweringDiagnostic::GenericArgsProhibited {
                    segment: (enum_segment + 1) as u32,
                    reason: GenericArgsProhibitedReason::EnumVariant,
                });
            }
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
                    ValueNs::GenericParam(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::Const)
                    }
                    ValueNs::StaticId(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::Static)
                    }
                    ValueNs::LocalBinding(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::LocalVariable)
                    }
                    ValueNs::FunctionId(_)
                    | ValueNs::StructId(_)
                    | ValueNs::EnumVariantId(_)
                    | ValueNs::ConstId(_) => {}
                }
            }
            ResolveValueResult::Partial(resolution, _, _) => {
                self.handle_type_ns_resolution(resolution);
            }
        };
        Some(res)
    }

    fn select_associated_type(&mut self, res: Option<TypeNs>, infer_args: bool) -> Ty {
        let Some(res) = res else {
            return TyKind::Error.intern(Interner);
        };
        let segment = self.current_or_prev_segment;
        let ty = named_associated_type_shorthand_candidates(
            self.ctx.db,
            self.ctx.def,
            res,
            Some(segment.name.clone()),
            move |name, t, associated_ty| {
                if name != segment.name {
                    return None;
                }
                let generics = self.ctx.generics();

                let parent_subst = t.substitution.clone();
                let parent_subst = match self.ctx.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        // if we're lowering to placeholders, we have to put them in now.
                        let s = generics.placeholder_subst(self.ctx.db);
                        s.apply(parent_subst, Interner)
                    }
                    ParamLoweringMode::Variable => {
                        // We need to shift in the bound vars, since
                        // `named_associated_type_shorthand_candidates` does not do that.
                        parent_subst.shifted_in_from(Interner, self.ctx.in_binders)
                    }
                };

                // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                // generic params. It's inefficient to splice the `Substitution`s, so we may want
                // that method to optionally take parent `Substitution` as we already know them at
                // this point (`t.substitution`).
                let substs =
                    self.substs_from_path_segment(associated_ty.into(), infer_args, None, true);

                let substs = Substitution::from_iter(
                    Interner,
                    parent_subst
                        .iter(Interner)
                        .chain(substs.iter(Interner).skip(parent_subst.len(Interner))),
                );

                Some(
                    TyKind::Alias(AliasTy::Projection(ProjectionTy {
                        associated_ty_id: to_assoc_type_id(associated_ty),
                        substitution: substs,
                    }))
                    .intern(Interner),
                )
            },
        );

        ty.unwrap_or_else(|| TyKind::Error.intern(Interner))
    }

    fn lower_path_inner(&mut self, typeable: TyDefId, infer_args: bool) -> Ty {
        let generic_def = match typeable {
            TyDefId::BuiltinType(builtin) => return TyBuilder::builtin(builtin),
            TyDefId::AdtId(it) => it.into(),
            TyDefId::TypeAliasId(it) => it.into(),
        };
        let substs = self.substs_from_path_segment(generic_def, infer_args, None, false);
        self.ctx.db.ty(typeable).substitute(Interner, &substs)
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
    ) -> Substitution {
        let prev_current_segment_idx = self.current_segment_idx;
        let prev_current_segment = self.current_or_prev_segment;

        let generic_def = match resolved {
            ValueTyDefId::FunctionId(it) => it.into(),
            ValueTyDefId::StructId(it) => it.into(),
            ValueTyDefId::UnionId(it) => it.into(),
            ValueTyDefId::ConstId(it) => it.into(),
            ValueTyDefId::StaticId(_) => return Substitution::empty(Interner),
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
                if let Some(penultimate) = penultimate {
                    if self.current_or_prev_segment.args_and_bindings.is_none()
                        && penultimate.args_and_bindings.is_some()
                    {
                        self.current_segment_idx = penultimate_idx;
                        self.current_or_prev_segment = penultimate;
                    }
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
        explicit_self_ty: Option<Ty>,
        lowering_assoc_type_generics: bool,
    ) -> Substitution {
        let mut lifetime_elision = self.ctx.lifetime_elision.clone();

        if let Some(args) = self.current_or_prev_segment.args_and_bindings {
            if args.parenthesized != GenericArgsParentheses::No {
                let prohibit_parens = match def {
                    GenericDefId::TraitId(trait_) => {
                        // RTN is prohibited anyways if we got here.
                        let is_rtn =
                            args.parenthesized == GenericArgsParentheses::ReturnTypeNotation;
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

                    return TyBuilder::unknown_subst(self.ctx.db, def);
                }

                // `Fn()`-style generics are treated like functions for the purpose of lifetime elision.
                lifetime_elision =
                    LifetimeElisionKind::AnonymousCreateParameter { report_in_path: false };
            }
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
        explicit_self_ty: Option<Ty>,
        generics_source: PathGenericsSource,
        lowering_assoc_type_generics: bool,
        lifetime_elision: LifetimeElisionKind,
    ) -> Substitution {
        struct LowererCtx<'a, 'b, 'c> {
            ctx: &'a mut PathLoweringContext<'b, 'c>,
            generics_source: PathGenericsSource,
        }

        impl GenericArgsLowerer for LowererCtx<'_, '_, '_> {
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
            ) -> crate::GenericArg {
                match (param, arg) {
                    (GenericParamDataRef::LifetimeParamData(_), GenericArg::Lifetime(lifetime)) => {
                        self.ctx.ctx.lower_lifetime(*lifetime).cast(Interner)
                    }
                    (GenericParamDataRef::TypeParamData(_), GenericArg::Type(type_ref)) => {
                        self.ctx.ctx.lower_ty(*type_ref).cast(Interner)
                    }
                    (GenericParamDataRef::ConstParamData(_), GenericArg::Const(konst)) => {
                        let GenericParamId::ConstParamId(const_id) = param_id else {
                            unreachable!("non-const param ID for const param");
                        };
                        self.ctx
                            .ctx
                            .lower_const(konst, self.ctx.ctx.db.const_param_ty(const_id))
                            .cast(Interner)
                    }
                    _ => unreachable!("unmatching param kinds were passed to `provided_kind()`"),
                }
            }

            fn provided_type_like_const(
                &mut self,
                const_ty: Ty,
                arg: TypeLikeConst<'_>,
            ) -> crate::Const {
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
                preceding_args: &[crate::GenericArg],
            ) -> crate::GenericArg {
                let default = || {
                    self.ctx
                        .ctx
                        .db
                        .generic_defaults(def)
                        .get(preceding_args.len())
                        .map(|default| default.clone().substitute(Interner, preceding_args))
                };
                match param {
                    GenericParamDataRef::LifetimeParamData(_) => error_lifetime().cast(Interner),
                    GenericParamDataRef::TypeParamData(param) => {
                        if !infer_args && param.default.is_some() {
                            if let Some(default) = default() {
                                return default;
                            }
                        }
                        TyKind::Error.intern(Interner).cast(Interner)
                    }
                    GenericParamDataRef::ConstParamData(param) => {
                        if !infer_args && param.default.is_some() {
                            if let Some(default) = default() {
                                return default;
                            }
                        }
                        let GenericParamId::ConstParamId(const_id) = param_id else {
                            unreachable!("non-const param ID for const param");
                        };
                        unknown_const_as_generic(self.ctx.ctx.db.const_param_ty(const_id))
                            .cast(Interner)
                    }
                }
            }

            fn parent_arg(&mut self, param_id: GenericParamId) -> crate::GenericArg {
                match param_id {
                    GenericParamId::TypeParamId(_) => TyKind::Error.intern(Interner).cast(Interner),
                    GenericParamId::ConstParamId(const_id) => {
                        unknown_const_as_generic(self.ctx.ctx.db.const_param_ty(const_id))
                    }
                    GenericParamId::LifetimeParamId(_) => error_lifetime().cast(Interner),
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
        explicit_self_ty: Ty,
        infer_args: bool,
    ) -> TraitRef {
        let substs = self.trait_ref_substs_from_path(resolved, explicit_self_ty, infer_args);
        TraitRef { trait_id: to_chalk_trait_id(resolved), substitution: substs }
    }

    fn trait_ref_substs_from_path(
        &mut self,
        resolved: TraitId,
        explicit_self_ty: Ty,
        infer_args: bool,
    ) -> Substitution {
        self.substs_from_path_segment(resolved.into(), infer_args, Some(explicit_self_ty), false)
    }

    pub(super) fn assoc_type_bindings_from_type_bound<'c>(
        mut self,
        trait_ref: TraitRef,
    ) -> Option<impl Iterator<Item = QuantifiedWhereClause> + use<'a, 'b, 'c>> {
        self.current_or_prev_segment.args_and_bindings.map(|args_and_bindings| {
            args_and_bindings.bindings.iter().enumerate().flat_map(move |(binding_idx, binding)| {
                let found = associated_type_by_name_including_super_traits(
                    self.ctx.db,
                    trait_ref.clone(),
                    &binding.name,
                );
                let (super_trait_ref, associated_ty) = match found {
                    None => return SmallVec::new(),
                    Some(t) => t,
                };
                let substitution =
                    self.with_lifetime_elision(LifetimeElisionKind::AnonymousReportError, |this| {
                        // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                        // generic params. It's inefficient to splice the `Substitution`s, so we may want
                        // that method to optionally take parent `Substitution` as we already know them at
                        // this point (`super_trait_ref.substitution`).
                        this.substs_from_args_and_bindings(
                            binding.args.as_ref(),
                            associated_ty.into(),
                            false, // this is not relevant
                            Some(super_trait_ref.self_type_parameter(Interner)),
                            PathGenericsSource::AssocType {
                                segment: this.current_segment_u32(),
                                assoc_type: binding_idx as u32,
                            },
                            false,
                            this.ctx.lifetime_elision.clone(),
                        )
                    });
                let substitution = Substitution::from_iter(
                    Interner,
                    super_trait_ref.substitution.iter(Interner).chain(
                        substitution
                            .iter(Interner)
                            .skip(super_trait_ref.substitution.len(Interner)),
                    ),
                );
                let projection_ty = ProjectionTy {
                    associated_ty_id: to_assoc_type_id(associated_ty),
                    substitution,
                };
                let mut predicates: SmallVec<[_; 1]> = SmallVec::with_capacity(
                    binding.type_ref.as_ref().map_or(0, |_| 1) + binding.bounds.len(),
                );

                if let Some(type_ref) = binding.type_ref {
                    let lifetime_elision =
                        if args_and_bindings.parenthesized == GenericArgsParentheses::ParenSugar {
                            // `Fn()`-style generics are elided like functions. This is `Output` (we lower to it in hir-def).
                            LifetimeElisionKind::for_fn_ret()
                        } else {
                            self.ctx.lifetime_elision.clone()
                        };
                    self.with_lifetime_elision(lifetime_elision, |this| {
                        match (&this.ctx.store[type_ref], this.ctx.impl_trait_mode.mode) {
                            (TypeRef::ImplTrait(_), ImplTraitLoweringMode::Disallowed) => (),
                            (
                                _,
                                ImplTraitLoweringMode::Disallowed | ImplTraitLoweringMode::Opaque,
                            ) => {
                                let ty = this.ctx.lower_ty(type_ref);
                                let alias_eq = AliasEq {
                                    alias: AliasTy::Projection(projection_ty.clone()),
                                    ty,
                                };
                                predicates.push(crate::wrap_empty_binders(WhereClause::AliasEq(
                                    alias_eq,
                                )));
                            }
                        }
                    });
                }

                self.with_lifetime_elision(LifetimeElisionKind::AnonymousReportError, |this| {
                    for bound in binding.bounds.iter() {
                        predicates.extend(
                            this.ctx.lower_type_bound(
                                bound,
                                TyKind::Alias(AliasTy::Projection(projection_ty.clone()))
                                    .intern(Interner),
                                false,
                            ),
                        );
                    }
                });

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

pub(crate) trait GenericArgsLowerer {
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
    ) -> crate::GenericArg;

    fn provided_type_like_const(&mut self, const_ty: Ty, arg: TypeLikeConst<'_>) -> crate::Const;

    fn inferred_kind(
        &mut self,
        def: GenericDefId,
        param_id: GenericParamId,
        param: GenericParamDataRef<'_>,
        infer_args: bool,
        preceding_args: &[crate::GenericArg],
    ) -> crate::GenericArg;

    fn parent_arg(&mut self, param_id: GenericParamId) -> crate::GenericArg;
}

/// Returns true if there was an error.
fn check_generic_args_len(
    args_and_bindings: Option<&GenericArgs>,
    def: GenericDefId,
    def_generics: &Generics,
    infer_args: bool,
    lifetime_elision: &LifetimeElisionKind,
    lowering_assoc_type_generics: bool,
    ctx: &mut impl GenericArgsLowerer,
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

pub(crate) fn substs_from_args_and_bindings(
    db: &dyn HirDatabase,
    store: &ExpressionStore,
    args_and_bindings: Option<&GenericArgs>,
    def: GenericDefId,
    mut infer_args: bool,
    lifetime_elision: LifetimeElisionKind,
    lowering_assoc_type_generics: bool,
    explicit_self_ty: Option<Ty>,
    ctx: &mut impl GenericArgsLowerer,
) -> Substitution {
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
            explicit_self_ty.map(|it| it.cast(Interner)).unwrap_or_else(|| {
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
                        let const_ty = db.const_param_ty(param_id);
                        substs.push(ctx.provided_type_like_const(const_ty, konst).cast(Interner));
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
                            static_lifetime().cast(Interner)
                        }
                        LifetimeElisionKind::Elided(lifetime) => lifetime.clone().cast(Interner),
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

    Substitution::from_iter(Interner, substs)
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
