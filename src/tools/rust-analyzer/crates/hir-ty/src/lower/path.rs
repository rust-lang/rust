//! A wrapper around [`TyLoweringContext`] specifically for lowering paths.

use std::iter;

use chalk_ir::{cast::Cast, fold::Shift, BoundVar};
use either::Either;
use hir_def::{
    data::TraitFlags,
    expr_store::HygieneId,
    generics::{TypeParamProvenance, WherePredicate, WherePredicateTypeTarget},
    path::{GenericArg, GenericArgs, Path, PathSegment, PathSegments},
    resolver::{ResolveValueResult, TypeNs, ValueNs},
    type_ref::{TypeBound, TypeRef},
    GenericDefId, GenericParamId, ItemContainerId, Lookup, TraitId,
};
use smallvec::SmallVec;
use stdx::never;

use crate::{
    consteval::unknown_const_as_generic,
    error_lifetime,
    generics::generics,
    lower::{
        generic_arg_to_chalk, named_associated_type_shorthand_candidates, ImplTraitLoweringState,
    },
    to_assoc_type_id, to_chalk_trait_id, to_placeholder_idx,
    utils::associated_type_by_name_including_super_traits,
    AliasEq, AliasTy, GenericArgsProhibitedReason, ImplTraitLoweringMode, Interner,
    ParamLoweringMode, PathLoweringDiagnostic, ProjectionTy, QuantifiedWhereClause, Substitution,
    TraitRef, Ty, TyBuilder, TyDefId, TyKind, TyLoweringContext, ValueTyDefId, WhereClause,
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

    pub(crate) fn lower_ty_relative_path(
        &mut self,
        ty: Ty,
        // We need the original resolution to lower `Self::AssocTy` correctly
        res: Option<TypeNs>,
    ) -> (Ty, Option<TypeNs>) {
        match self.segments.len() - self.current_segment_idx {
            0 => (ty, res),
            1 => {
                // resolve unselected assoc types
                (self.select_associated_type(res), None)
            }
            _ => {
                // FIXME report error (ambiguous associated type)
                (TyKind::Error.intern(Interner), None)
            }
        }
    }

    fn prohibit_parenthesized_generic_args(&mut self) -> bool {
        if let Some(generic_args) = self.current_or_prev_segment.args_and_bindings {
            if generic_args.desugared_from_fn {
                let segment = self.current_segment_u32();
                self.on_diagnostic(
                    PathLoweringDiagnostic::ParenthesizedGenericArgsWithoutFnTrait { segment },
                );
                return true;
            }
        }
        false
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
                        );

                        self.skip_resolved_segment();
                        let segment = self.current_or_prev_segment;
                        let found =
                            self.ctx.db.trait_data(trait_).associated_type_by_name(segment.name);

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
                                );
                                let len_self =
                                    generics(self.ctx.db.upcast(), associated_ty.into()).len_self();
                                let substitution = Substitution::from_iter(
                                    Interner,
                                    substitution
                                        .iter(Interner)
                                        .take(len_self)
                                        .chain(trait_ref.substitution.iter(Interner)),
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
                    let idx = match self
                        .ctx
                        .generics()
                        .expect("generics in scope")
                        .type_or_const_param_idx(param_id.into())
                    {
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
                let generics = self.ctx.generics().expect("impl should have generic param scope");

                match self.ctx.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        // `def` can be either impl itself or item within, and we need impl itself
                        // now.
                        let generics = generics.parent_or_self();
                        let subst = generics.placeholder_subst(self.ctx.db);
                        self.ctx.db.impl_self_ty(impl_id).substitute(Interner, &subst)
                    }
                    ParamLoweringMode::Variable => {
                        let starting_from = match generics.def() {
                            GenericDefId::ImplId(_) => 0,
                            // `def` is an item within impl. We need to substitute `BoundVar`s but
                            // remember that they are for parent (i.e. impl) generic params so they
                            // come after our own params.
                            _ => generics.len_self(),
                        };
                        TyBuilder::impl_self_ty(self.ctx.db, impl_id)
                            .fill_with_bound_vars(self.ctx.in_binders, starting_from)
                            .build()
                    }
                }
            }
            TypeNs::AdtSelfType(adt) => {
                let generics = generics(self.ctx.db.upcast(), adt.into());
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
            TypeNs::EnumVariantId(_) => return (TyKind::Error.intern(Interner), None),
        };

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
        let (resolution, remaining_index, _, prefix_info) = self
            .ctx
            .resolver
            .resolve_path_in_type_ns_with_prefix_info(self.ctx.db.upcast(), self.path)?;

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
            self.ctx.db.upcast(),
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

    fn select_associated_type(&mut self, res: Option<TypeNs>) -> Ty {
        let Some((generics, res)) = self.ctx.generics().zip(res) else {
            return TyKind::Error.intern(Interner);
        };
        let segment = self.current_or_prev_segment;
        let ty = named_associated_type_shorthand_candidates(
            self.ctx.db,
            generics.def(),
            res,
            Some(segment.name.clone()),
            move |name, t, associated_ty| {
                let generics = self.ctx.generics().unwrap();

                if name != segment.name {
                    return None;
                }

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
                let substs = self.substs_from_path_segment(associated_ty.into(), false, None);

                let len_self =
                    crate::generics::generics(self.ctx.db.upcast(), associated_ty.into())
                        .len_self();

                let substs = Substitution::from_iter(
                    Interner,
                    substs.iter(Interner).take(len_self).chain(parent_subst.iter(Interner)),
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
        let substs = self.substs_from_path_segment(generic_def, infer_args, None);
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
                var.lookup(self.ctx.db.upcast()).parent.into()
            }
        };
        let result = self.substs_from_path_segment(generic_def, infer_args, None);
        self.current_segment_idx = prev_current_segment_idx;
        self.current_or_prev_segment = prev_current_segment;
        result
    }

    pub(crate) fn substs_from_path_segment(
        &mut self,
        def: GenericDefId,
        infer_args: bool,
        explicit_self_ty: Option<Ty>,
    ) -> Substitution {
        let prohibit_parens = match def {
            GenericDefId::TraitId(trait_) => {
                let trait_data = self.ctx.db.trait_data(trait_);
                !trait_data.flags.contains(TraitFlags::RUSTC_PAREN_SUGAR)
            }
            _ => true,
        };
        if prohibit_parens && self.prohibit_parenthesized_generic_args() {
            return TyBuilder::unknown_subst(self.ctx.db, def);
        }

        self.substs_from_args_and_bindings(
            self.current_or_prev_segment.args_and_bindings,
            def,
            infer_args,
            explicit_self_ty,
        )
    }

    pub(super) fn substs_from_args_and_bindings(
        &mut self,
        args_and_bindings: Option<&GenericArgs>,
        def: GenericDefId,
        infer_args: bool,
        explicit_self_ty: Option<Ty>,
    ) -> Substitution {
        // Order is
        // - Optional Self parameter
        // - Lifetime parameters
        // - Type or Const parameters
        // - Parent parameters
        let def_generics = generics(self.ctx.db.upcast(), def);
        let (
            parent_params,
            self_param,
            type_params,
            const_params,
            impl_trait_params,
            lifetime_params,
        ) = def_generics.provenance_split();
        let item_len =
            self_param as usize + type_params + const_params + impl_trait_params + lifetime_params;
        let total_len = parent_params + item_len;

        let mut substs = Vec::new();

        // we need to iterate the lifetime and type/const params separately as our order of them
        // differs from the supplied syntax

        let ty_error = || TyKind::Error.intern(Interner).cast(Interner);
        let mut def_toc_iter = def_generics.iter_self_type_or_consts_id();
        let fill_self_param = || {
            if self_param {
                let self_ty = explicit_self_ty.map(|x| x.cast(Interner)).unwrap_or_else(ty_error);

                if let Some(id) = def_toc_iter.next() {
                    assert!(matches!(id, GenericParamId::TypeParamId(_)));
                    substs.push(self_ty);
                }
            }
        };
        let mut had_explicit_args = false;

        if let Some(&GenericArgs { ref args, has_self_type, .. }) = args_and_bindings {
            // Fill in the self param first
            if has_self_type && self_param {
                had_explicit_args = true;
                if let Some(id) = def_toc_iter.next() {
                    assert!(matches!(id, GenericParamId::TypeParamId(_)));
                    had_explicit_args = true;
                    if let GenericArg::Type(ty) = &args[0] {
                        substs.push(self.ctx.lower_ty(*ty).cast(Interner));
                    }
                }
            } else {
                fill_self_param()
            };

            // Then fill in the supplied lifetime args, or error lifetimes if there are too few
            // (default lifetimes aren't a thing)
            for arg in args
                .iter()
                .filter_map(|arg| match arg {
                    GenericArg::Lifetime(arg) => Some(self.ctx.lower_lifetime(arg)),
                    _ => None,
                })
                .chain(iter::repeat(error_lifetime()))
                .take(lifetime_params)
            {
                substs.push(arg.cast(Interner));
            }

            let skip = if has_self_type { 1 } else { 0 };
            // Fill in supplied type and const args
            // Note if non-lifetime args are provided, it should be all of them, but we can't rely on that
            for (arg, id) in args
                .iter()
                .filter(|arg| !matches!(arg, GenericArg::Lifetime(_)))
                .skip(skip)
                .take(type_params + const_params)
                .zip(def_toc_iter)
            {
                had_explicit_args = true;
                let arg = generic_arg_to_chalk(
                    self.ctx.db,
                    id,
                    arg,
                    self.ctx,
                    self.ctx.types_map,
                    |ctx, type_ref| ctx.lower_ty(type_ref),
                    |ctx, const_ref, ty| ctx.lower_const(const_ref, ty),
                    |ctx, lifetime_ref| ctx.lower_lifetime(lifetime_ref),
                );
                substs.push(arg);
            }
        } else {
            fill_self_param();
        }

        let param_to_err = |id| match id {
            GenericParamId::ConstParamId(x) => {
                unknown_const_as_generic(self.ctx.db.const_param_ty(x))
            }
            GenericParamId::TypeParamId(_) => ty_error(),
            GenericParamId::LifetimeParamId(_) => error_lifetime().cast(Interner),
        };
        // handle defaults. In expression or pattern path segments without
        // explicitly specified type arguments, missing type arguments are inferred
        // (i.e. defaults aren't used).
        // Generic parameters for associated types are not supposed to have defaults, so we just
        // ignore them.
        let is_assoc_ty = || match def {
            GenericDefId::TypeAliasId(id) => {
                matches!(id.lookup(self.ctx.db.upcast()).container, ItemContainerId::TraitId(_))
            }
            _ => false,
        };
        let fill_defaults = (!infer_args || had_explicit_args) && !is_assoc_ty();
        if fill_defaults {
            let defaults = &*self.ctx.db.generic_defaults(def);
            let (item, _parent) = defaults.split_at(item_len);
            let parent_from = item_len - substs.len();

            let mut rem =
                def_generics.iter_id().skip(substs.len()).map(param_to_err).collect::<Vec<_>>();
            // Fill in defaults for type/const params
            for (idx, default_ty) in item[substs.len()..].iter().enumerate() {
                // each default can depend on the previous parameters
                let substs_so_far = Substitution::from_iter(
                    Interner,
                    substs.iter().cloned().chain(rem[idx..].iter().cloned()),
                );
                substs.push(default_ty.clone().substitute(Interner, &substs_so_far));
            }
            // Fill in remaining parent params
            substs.extend(rem.drain(parent_from..));
        } else {
            // Fill in remaining def params and parent params
            substs.extend(def_generics.iter_id().skip(substs.len()).map(param_to_err));
        }

        assert_eq!(substs.len(), total_len, "expected {} substs, got {}", total_len, substs.len());
        Substitution::from_iter(Interner, substs)
    }

    pub(crate) fn lower_trait_ref_from_resolved_path(
        &mut self,
        resolved: TraitId,
        explicit_self_ty: Ty,
    ) -> TraitRef {
        let substs = self.trait_ref_substs_from_path(resolved, explicit_self_ty);
        TraitRef { trait_id: to_chalk_trait_id(resolved), substitution: substs }
    }

    fn trait_ref_substs_from_path(
        &mut self,
        resolved: TraitId,
        explicit_self_ty: Ty,
    ) -> Substitution {
        self.substs_from_path_segment(resolved.into(), false, Some(explicit_self_ty))
    }

    pub(super) fn assoc_type_bindings_from_type_bound<'c>(
        mut self,
        bound: &'c TypeBound,
        trait_ref: TraitRef,
    ) -> Option<impl Iterator<Item = QuantifiedWhereClause> + use<'a, 'b, 'c>> {
        self.current_or_prev_segment.args_and_bindings.map(|args_and_bindings| {
            args_and_bindings.bindings.iter().flat_map(move |binding| {
                let found = associated_type_by_name_including_super_traits(
                    self.ctx.db,
                    trait_ref.clone(),
                    &binding.name,
                );
                let (super_trait_ref, associated_ty) = match found {
                    None => return SmallVec::new(),
                    Some(t) => t,
                };
                // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                // generic params. It's inefficient to splice the `Substitution`s, so we may want
                // that method to optionally take parent `Substitution` as we already know them at
                // this point (`super_trait_ref.substitution`).
                let substitution = self.substs_from_args_and_bindings(
                    binding.args.as_ref(),
                    associated_ty.into(),
                    false, // this is not relevant
                    Some(super_trait_ref.self_type_parameter(Interner)),
                );
                let self_params = generics(self.ctx.db.upcast(), associated_ty.into()).len_self();
                let substitution = Substitution::from_iter(
                    Interner,
                    substitution
                        .iter(Interner)
                        .take(self_params)
                        .chain(super_trait_ref.substitution.iter(Interner)),
                );
                let projection_ty = ProjectionTy {
                    associated_ty_id: to_assoc_type_id(associated_ty),
                    substitution,
                };
                let mut predicates: SmallVec<[_; 1]> = SmallVec::with_capacity(
                    binding.type_ref.as_ref().map_or(0, |_| 1) + binding.bounds.len(),
                );
                if let Some(type_ref) = binding.type_ref {
                    match (&self.ctx.types_map[type_ref], self.ctx.impl_trait_mode.mode) {
                        (TypeRef::ImplTrait(_), ImplTraitLoweringMode::Disallowed) => (),
                        (_, ImplTraitLoweringMode::Disallowed | ImplTraitLoweringMode::Opaque) => {
                            let ty = self.ctx.lower_ty(type_ref);
                            let alias_eq =
                                AliasEq { alias: AliasTy::Projection(projection_ty.clone()), ty };
                            predicates
                                .push(crate::wrap_empty_binders(WhereClause::AliasEq(alias_eq)));
                        }
                        (_, ImplTraitLoweringMode::Param | ImplTraitLoweringMode::Variable) => {
                            // Find the generic index for the target of our `bound`
                            let target_param_idx =
                                self.ctx.resolver.where_predicates_in_scope().find_map(|(p, _)| {
                                    match p {
                                        WherePredicate::TypeBound {
                                            target: WherePredicateTypeTarget::TypeOrConstParam(idx),
                                            bound: b,
                                        } if b == bound => Some(idx),
                                        _ => None,
                                    }
                                });
                            let ty = if let Some(target_param_idx) = target_param_idx {
                                let mut counter = 0;
                                let generics = self.ctx.generics().expect("generics in scope");
                                for (idx, data) in generics.iter_self_type_or_consts() {
                                    // Count the number of `impl Trait` things that appear before
                                    // the target of our `bound`.
                                    // Our counter within `impl_trait_mode` should be that number
                                    // to properly lower each types within `type_ref`
                                    if data.type_param().is_some_and(|p| {
                                        p.provenance == TypeParamProvenance::ArgumentImplTrait
                                    }) {
                                        counter += 1;
                                    }
                                    if idx == *target_param_idx {
                                        break;
                                    }
                                }
                                let mut ext = TyLoweringContext::new_maybe_unowned(
                                    self.ctx.db,
                                    self.ctx.resolver,
                                    self.ctx.types_map,
                                    self.ctx.types_source_map,
                                    self.ctx.owner,
                                )
                                .with_type_param_mode(self.ctx.type_param_mode);
                                match self.ctx.impl_trait_mode.mode {
                                    ImplTraitLoweringMode::Param => {
                                        ext.impl_trait_mode =
                                            ImplTraitLoweringState::param(counter);
                                    }
                                    ImplTraitLoweringMode::Variable => {
                                        ext.impl_trait_mode =
                                            ImplTraitLoweringState::variable(counter);
                                    }
                                    _ => unreachable!(),
                                }
                                let ty = ext.lower_ty(type_ref);
                                self.ctx.diagnostics.extend(ext.diagnostics);
                                ty
                            } else {
                                self.ctx.lower_ty(type_ref)
                            };

                            let alias_eq =
                                AliasEq { alias: AliasTy::Projection(projection_ty.clone()), ty };
                            predicates
                                .push(crate::wrap_empty_binders(WhereClause::AliasEq(alias_eq)));
                        }
                    }
                }
                for bound in binding.bounds.iter() {
                    predicates.extend(self.ctx.lower_type_bound(
                        bound,
                        TyKind::Alias(AliasTy::Projection(projection_ty.clone())).intern(Interner),
                        false,
                    ));
                }
                predicates
            })
        })
    }
}
