use std::sync::Arc;

use rustc_ast::{self as ast, *};
use rustc_hir::def::{DefKind, PartialRes, PerNS, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, GenericArg};
use rustc_middle::{span_bug, ty};
use rustc_session::parse::add_feature_diagnostics;
use rustc_span::{BytePos, DUMMY_SP, DesugaringKind, Ident, Span, Symbol, sym};
use smallvec::smallvec;
use tracing::{debug, instrument};

use super::errors::{
    AsyncBoundNotOnTrait, AsyncBoundOnlyForFnTraits, BadReturnTypeNotation,
    GenericTypeWithParentheses, RTNSuggestion, UseAngleBrackets,
};
use super::{
    AllowReturnTypeNotation, GenericArgsCtor, GenericArgsMode, ImplTraitContext, ImplTraitPosition,
    LifetimeRes, LoweringContext, ParamMode, ResolverAstLoweringExt,
};

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    #[instrument(level = "trace", skip(self))]
    pub(crate) fn lower_qpath(
        &mut self,
        id: NodeId,
        qself: &Option<Box<QSelf>>,
        p: &Path,
        param_mode: ParamMode,
        allow_return_type_notation: AllowReturnTypeNotation,
        itctx: ImplTraitContext,
        // modifiers of the impl/bound if this is a trait path
        modifiers: Option<ast::TraitBoundModifiers>,
    ) -> hir::QPath<'hir> {
        let qself_position = qself.as_ref().map(|q| q.position);
        let qself = qself
            .as_ref()
            // Reject cases like `<impl Trait>::Assoc` and `<impl Trait as Trait>::Assoc`.
            .map(|q| self.lower_ty(&q.ty, ImplTraitContext::Disallowed(ImplTraitPosition::Path)));

        let partial_res =
            self.resolver.get_partial_res(id).unwrap_or_else(|| PartialRes::new(Res::Err));
        let base_res = partial_res.base_res();
        let unresolved_segments = partial_res.unresolved_segments();

        let mut res = self.lower_res(base_res);

        // When we have an `async` kw on a bound, map the trait it resolves to.
        if let Some(TraitBoundModifiers { asyncness: BoundAsyncness::Async(_), .. }) = modifiers {
            match res {
                Res::Def(DefKind::Trait, def_id) => {
                    if let Some(async_def_id) = self.map_trait_to_async_trait(def_id) {
                        res = Res::Def(DefKind::Trait, async_def_id);
                    } else {
                        self.dcx().emit_err(AsyncBoundOnlyForFnTraits { span: p.span });
                    }
                }
                Res::Err => {
                    // No additional error.
                }
                _ => {
                    // This error isn't actually emitted AFAICT, but it's best to keep
                    // it around in case the resolver doesn't always check the defkind
                    // of an item or something.
                    self.dcx().emit_err(AsyncBoundNotOnTrait { span: p.span, descr: res.descr() });
                }
            }
        }

        // Ungate the `async_fn_traits` feature in the path if the trait is
        // named via either `async Fn*()` or `AsyncFn*()`.
        let bound_modifier_allowed_features = if let Res::Def(DefKind::Trait, async_def_id) = res
            && self.tcx.async_fn_trait_kind_from_def_id(async_def_id).is_some()
        {
            Some(Arc::clone(&self.allow_async_fn_traits))
        } else {
            None
        };

        // Only permit `impl Trait` in the final segment. E.g., we permit `Option<impl Trait>`,
        // `option::Option<T>::Xyz<impl Trait>` and reject `option::Option<impl Trait>::Xyz`.
        let itctx = |i| {
            if i + 1 == p.segments.len() {
                itctx
            } else {
                ImplTraitContext::Disallowed(ImplTraitPosition::Path)
            }
        };

        let path_span_lo = p.span.shrink_to_lo();
        let proj_start = p.segments.len() - unresolved_segments;
        let path = self.arena.alloc(hir::Path {
            res,
            segments: self.arena.alloc_from_iter(p.segments[..proj_start].iter().enumerate().map(
                |(i, segment)| {
                    let param_mode = match (qself_position, param_mode) {
                        (Some(j), ParamMode::Optional) if i < j => {
                            // This segment is part of the trait path in a
                            // qualified path - one of `a`, `b` or `Trait`
                            // in `<X as a::b::Trait>::T::U::method`.
                            ParamMode::Explicit
                        }
                        _ => param_mode,
                    };

                    let generic_args_mode = match base_res {
                        // `a::b::Trait(Args)`
                        Res::Def(DefKind::Trait, _) if i + 1 == proj_start => {
                            GenericArgsMode::ParenSugar
                        }
                        // `a::b::Trait(Args)::TraitItem`
                        Res::Def(DefKind::AssocFn, _)
                        | Res::Def(DefKind::AssocConst, _)
                        | Res::Def(DefKind::AssocTy, _)
                            if i + 2 == proj_start =>
                        {
                            GenericArgsMode::ParenSugar
                        }
                        Res::Def(DefKind::AssocFn, _) if i + 1 == proj_start => {
                            match allow_return_type_notation {
                                AllowReturnTypeNotation::Yes => GenericArgsMode::ReturnTypeNotation,
                                AllowReturnTypeNotation::No => GenericArgsMode::Err,
                            }
                        }
                        // Avoid duplicated errors.
                        Res::Err => GenericArgsMode::Silence,
                        // An error
                        _ => GenericArgsMode::Err,
                    };

                    self.lower_path_segment(
                        p.span,
                        segment,
                        param_mode,
                        generic_args_mode,
                        itctx(i),
                        bound_modifier_allowed_features.clone(),
                    )
                },
            )),
            span: self.lower_span(
                p.segments[..proj_start]
                    .last()
                    .map_or(path_span_lo, |segment| path_span_lo.to(segment.span())),
            ),
        });

        if let Some(bound_modifier_allowed_features) = bound_modifier_allowed_features {
            path.span = self.mark_span_with_reason(
                DesugaringKind::BoundModifier,
                path.span,
                Some(bound_modifier_allowed_features),
            );
        }

        // Simple case, either no projections, or only fully-qualified.
        // E.g., `std::mem::size_of` or `<I as Iterator>::Item`.
        if unresolved_segments == 0 {
            return hir::QPath::Resolved(qself, path);
        }

        // Create the innermost type that we're projecting from.
        let mut ty = if path.segments.is_empty() {
            // If the base path is empty that means there exists a
            // syntactical `Self`, e.g., `&i32` in `<&i32>::clone`.
            qself.expect("missing QSelf for <T>::...")
        } else {
            // Otherwise, the base path is an implicit `Self` type path,
            // e.g., `Vec` in `Vec::new` or `<I as Iterator>::Item` in
            // `<I as Iterator>::Item::default`.
            let new_id = self.next_id();
            self.arena.alloc(self.ty_path(new_id, path.span, hir::QPath::Resolved(qself, path)))
        };

        // Anything after the base path are associated "extensions",
        // out of which all but the last one are associated types,
        // e.g., for `std::vec::Vec::<T>::IntoIter::Item::clone`:
        // * base path is `std::vec::Vec<T>`
        // * "extensions" are `IntoIter`, `Item` and `clone`
        // * type nodes are:
        //   1. `std::vec::Vec<T>` (created above)
        //   2. `<std::vec::Vec<T>>::IntoIter`
        //   3. `<<std::vec::Vec<T>>::IntoIter>::Item`
        // * final path is `<<<std::vec::Vec<T>>::IntoIter>::Item>::clone`
        for (i, segment) in p.segments.iter().enumerate().skip(proj_start) {
            // If this is a type-dependent `T::method(..)`.
            let generic_args_mode = if i + 1 == p.segments.len()
                && matches!(allow_return_type_notation, AllowReturnTypeNotation::Yes)
            {
                GenericArgsMode::ReturnTypeNotation
            } else {
                GenericArgsMode::Err
            };

            let hir_segment = self.arena.alloc(self.lower_path_segment(
                p.span,
                segment,
                param_mode,
                generic_args_mode,
                itctx(i),
                None,
            ));
            let qpath = hir::QPath::TypeRelative(ty, hir_segment);

            // It's finished, return the extension of the right node type.
            if i == p.segments.len() - 1 {
                return qpath;
            }

            // Wrap the associated extension in another type node.
            let new_id = self.next_id();
            ty = self.arena.alloc(self.ty_path(new_id, path_span_lo.to(segment.span()), qpath));
        }

        // We should've returned in the for loop above.

        self.dcx().span_bug(
            p.span,
            format!(
                "lower_qpath: no final extension segment in {}..{}",
                proj_start,
                p.segments.len()
            ),
        );
    }

    pub(crate) fn lower_use_path(
        &mut self,
        res: PerNS<Option<Res>>,
        p: &Path,
        param_mode: ParamMode,
    ) -> &'hir hir::UsePath<'hir> {
        assert!(!res.is_empty());
        self.arena.alloc(hir::UsePath {
            res,
            segments: self.arena.alloc_from_iter(p.segments.iter().map(|segment| {
                self.lower_path_segment(
                    p.span,
                    segment,
                    param_mode,
                    GenericArgsMode::Err,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                    None,
                )
            })),
            span: self.lower_span(p.span),
        })
    }

    pub(crate) fn lower_path_segment(
        &mut self,
        path_span: Span,
        segment: &PathSegment,
        param_mode: ParamMode,
        generic_args_mode: GenericArgsMode,
        itctx: ImplTraitContext,
        // Additional features ungated with a bound modifier like `async`.
        // This is passed down to the implicit associated type binding in
        // parenthesized bounds.
        bound_modifier_allowed_features: Option<Arc<[Symbol]>>,
    ) -> hir::PathSegment<'hir> {
        debug!("path_span: {:?}, lower_path_segment(segment: {:?})", path_span, segment);
        let (mut generic_args, infer_args) = if let Some(generic_args) = segment.args.as_deref() {
            match generic_args {
                GenericArgs::AngleBracketed(data) => {
                    self.lower_angle_bracketed_parameter_data(data, param_mode, itctx)
                }
                GenericArgs::Parenthesized(data) => match generic_args_mode {
                    GenericArgsMode::ReturnTypeNotation => {
                        let err = match (&data.inputs[..], &data.output) {
                            ([_, ..], FnRetTy::Default(_)) => {
                                BadReturnTypeNotation::Inputs { span: data.inputs_span }
                            }
                            ([], FnRetTy::Default(_)) => {
                                BadReturnTypeNotation::NeedsDots { span: data.inputs_span }
                            }
                            // The case `T: Trait<method(..) -> Ret>` is handled in the parser.
                            (_, FnRetTy::Ty(ty)) => {
                                let span = data.inputs_span.shrink_to_hi().to(ty.span);
                                BadReturnTypeNotation::Output {
                                    span,
                                    suggestion: RTNSuggestion {
                                        output: span,
                                        input: data.inputs_span,
                                    },
                                }
                            }
                        };
                        let mut err = self.dcx().create_err(err);
                        if !self.tcx.features().return_type_notation()
                            && self.tcx.sess.is_nightly_build()
                        {
                            add_feature_diagnostics(
                                &mut err,
                                &self.tcx.sess,
                                sym::return_type_notation,
                            );
                        }
                        err.emit();
                        (
                            GenericArgsCtor {
                                args: Default::default(),
                                constraints: &[],
                                parenthesized: hir::GenericArgsParentheses::ReturnTypeNotation,
                                span: path_span,
                            },
                            false,
                        )
                    }
                    GenericArgsMode::ParenSugar | GenericArgsMode::Silence => self
                        .lower_parenthesized_parameter_data(
                            data,
                            itctx,
                            bound_modifier_allowed_features,
                        ),
                    GenericArgsMode::Err => {
                        // Suggest replacing parentheses with angle brackets `Trait(params...)` to `Trait<params...>`
                        let sub = if !data.inputs.is_empty() {
                            // Start of the span to the 1st character of 1st argument
                            let open_param = data.inputs_span.shrink_to_lo().to(data
                                .inputs
                                .first()
                                .unwrap()
                                .span
                                .shrink_to_lo());
                            // Last character position of last argument to the end of the span
                            let close_param = data
                                .inputs
                                .last()
                                .unwrap()
                                .span
                                .shrink_to_hi()
                                .to(data.inputs_span.shrink_to_hi());

                            Some(UseAngleBrackets { open_param, close_param })
                        } else {
                            None
                        };
                        self.dcx().emit_err(GenericTypeWithParentheses { span: data.span, sub });
                        (
                            self.lower_angle_bracketed_parameter_data(
                                &data.as_angle_bracketed_args(),
                                param_mode,
                                itctx,
                            )
                            .0,
                            false,
                        )
                    }
                },
                GenericArgs::ParenthesizedElided(span) => {
                    match generic_args_mode {
                        GenericArgsMode::ReturnTypeNotation | GenericArgsMode::Silence => {
                            // Ok
                        }
                        GenericArgsMode::ParenSugar | GenericArgsMode::Err => {
                            self.dcx().emit_err(BadReturnTypeNotation::Position { span: *span });
                        }
                    }
                    (
                        GenericArgsCtor {
                            args: Default::default(),
                            constraints: &[],
                            parenthesized: hir::GenericArgsParentheses::ReturnTypeNotation,
                            span: *span,
                        },
                        false,
                    )
                }
            }
        } else {
            (
                GenericArgsCtor {
                    args: Default::default(),
                    constraints: &[],
                    parenthesized: hir::GenericArgsParentheses::No,
                    span: path_span.shrink_to_hi(),
                },
                param_mode == ParamMode::Optional,
            )
        };

        let has_lifetimes =
            generic_args.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)));

        // FIXME(return_type_notation): Is this correct? I think so.
        if generic_args.parenthesized != hir::GenericArgsParentheses::ParenSugar && !has_lifetimes {
            self.maybe_insert_elided_lifetimes_in_path(
                path_span,
                segment.id,
                segment.ident.span,
                &mut generic_args,
            );
        }

        let res = self.expect_full_res(segment.id);
        let hir_id = self.lower_node_id(segment.id);
        debug!(
            "lower_path_segment: ident={:?} original-id={:?} new-id={:?}",
            segment.ident, segment.id, hir_id,
        );

        hir::PathSegment {
            ident: self.lower_ident(segment.ident),
            hir_id,
            res: self.lower_res(res),
            infer_args,
            args: if generic_args.is_empty() && generic_args.span.is_empty() {
                None
            } else {
                Some(generic_args.into_generic_args(self))
            },
        }
    }

    fn maybe_insert_elided_lifetimes_in_path(
        &mut self,
        path_span: Span,
        segment_id: NodeId,
        segment_ident_span: Span,
        generic_args: &mut GenericArgsCtor<'hir>,
    ) {
        let (start, end) = match self.resolver.get_lifetime_res(segment_id) {
            Some(LifetimeRes::ElidedAnchor { start, end }) => (start, end),
            None => return,
            Some(res) => {
                span_bug!(path_span, "expected an elided lifetime to insert. found {res:?}")
            }
        };
        let expected_lifetimes = end.as_usize() - start.as_usize();
        debug!(expected_lifetimes);

        // Note: these spans are used for diagnostics when they can't be inferred.
        // See rustc_resolve::late::lifetimes::LifetimeContext::add_missing_lifetime_specifiers_label
        let (elided_lifetime_span, angle_brackets) = if generic_args.span.is_empty() {
            // No brackets, e.g. `Path`: use an empty span just past the end of the identifier.
            // HACK: we use find_ancestor_inside to properly suggest elided spans in paths
            // originating from macros, since the segment's span might be from a macro arg.
            (
                segment_ident_span.find_ancestor_inside(path_span).unwrap_or(path_span),
                hir::AngleBrackets::Missing,
            )
        } else {
            // Brackets, e.g. `Path<>` or `Path<T>`: use an empty span just after the `<`.
            (
                generic_args.span.with_lo(generic_args.span.lo() + BytePos(1)).shrink_to_lo(),
                if generic_args.is_empty() {
                    hir::AngleBrackets::Empty
                } else {
                    hir::AngleBrackets::Full
                },
            )
        };

        generic_args.args.insert_many(
            0,
            (start..end).map(|id| {
                let l =
                    self.lower_lifetime_hidden_in_path(id, elided_lifetime_span, angle_brackets);
                GenericArg::Lifetime(l)
            }),
        );
    }

    pub(crate) fn lower_angle_bracketed_parameter_data(
        &mut self,
        data: &AngleBracketedArgs,
        param_mode: ParamMode,
        itctx: ImplTraitContext,
    ) -> (GenericArgsCtor<'hir>, bool) {
        let has_non_lt_args = data.args.iter().any(|arg| match arg {
            AngleBracketedArg::Arg(ast::GenericArg::Lifetime(_))
            | AngleBracketedArg::Constraint(_) => false,
            AngleBracketedArg::Arg(ast::GenericArg::Type(_) | ast::GenericArg::Const(_)) => true,
        });
        let args = data
            .args
            .iter()
            .filter_map(|arg| match arg {
                AngleBracketedArg::Arg(arg) => Some(self.lower_generic_arg(arg, itctx)),
                AngleBracketedArg::Constraint(_) => None,
            })
            .collect();
        let constraints =
            self.arena.alloc_from_iter(data.args.iter().filter_map(|arg| match arg {
                AngleBracketedArg::Constraint(c) => {
                    Some(self.lower_assoc_item_constraint(c, itctx))
                }
                AngleBracketedArg::Arg(_) => None,
            }));
        let ctor = GenericArgsCtor {
            args,
            constraints,
            parenthesized: hir::GenericArgsParentheses::No,
            span: data.span,
        };
        (ctor, !has_non_lt_args && param_mode == ParamMode::Optional)
    }

    fn lower_parenthesized_parameter_data(
        &mut self,
        data: &ParenthesizedArgs,
        itctx: ImplTraitContext,
        bound_modifier_allowed_features: Option<Arc<[Symbol]>>,
    ) -> (GenericArgsCtor<'hir>, bool) {
        // Switch to `PassThrough` mode for anonymous lifetimes; this
        // means that we permit things like `&Ref<T>`, where `Ref` has
        // a hidden lifetime parameter. This is needed for backwards
        // compatibility, even in contexts like an impl header where
        // we generally don't permit such things (see #51008).
        let ParenthesizedArgs { span, inputs, inputs_span, output } = data;
        let inputs = self.arena.alloc_from_iter(inputs.iter().map(|ty| {
            self.lower_ty_direct(ty, ImplTraitContext::Disallowed(ImplTraitPosition::FnTraitParam))
        }));
        let output_ty = match output {
            // Only allow `impl Trait` in return position. i.e.:
            // ```rust
            // fn f(_: impl Fn() -> impl Debug) -> impl Fn() -> impl Debug
            // //      disallowed --^^^^^^^^^^        allowed --^^^^^^^^^^
            // ```
            FnRetTy::Ty(ty) if matches!(itctx, ImplTraitContext::OpaqueTy { .. }) => {
                if self.tcx.features().impl_trait_in_fn_trait_return() {
                    self.lower_ty(ty, itctx)
                } else {
                    self.lower_ty(
                        ty,
                        ImplTraitContext::FeatureGated(
                            ImplTraitPosition::FnTraitReturn,
                            sym::impl_trait_in_fn_trait_return,
                        ),
                    )
                }
            }
            FnRetTy::Ty(ty) => {
                self.lower_ty(ty, ImplTraitContext::Disallowed(ImplTraitPosition::FnTraitReturn))
            }
            FnRetTy::Default(_) => self.arena.alloc(self.ty_tup(*span, &[])),
        };
        let args = smallvec![GenericArg::Type(
            self.arena.alloc(self.ty_tup(*inputs_span, inputs)).try_as_ambig_ty().unwrap()
        )];

        // If we have a bound like `async Fn() -> T`, make sure that we mark the
        // `Output = T` associated type bound with the right feature gates.
        let mut output_span = output_ty.span;
        if let Some(bound_modifier_allowed_features) = bound_modifier_allowed_features {
            output_span = self.mark_span_with_reason(
                DesugaringKind::BoundModifier,
                output_span,
                Some(bound_modifier_allowed_features),
            );
        }
        let constraint = self.assoc_ty_binding(sym::Output, output_span, output_ty);

        (
            GenericArgsCtor {
                args,
                constraints: arena_vec![self; constraint],
                parenthesized: hir::GenericArgsParentheses::ParenSugar,
                span: data.inputs_span,
            },
            false,
        )
    }

    /// An associated type binding (i.e., associated type equality constraint).
    pub(crate) fn assoc_ty_binding(
        &mut self,
        assoc_ty_name: rustc_span::Symbol,
        span: Span,
        ty: &'hir hir::Ty<'hir>,
    ) -> hir::AssocItemConstraint<'hir> {
        let ident = Ident::with_dummy_span(assoc_ty_name);
        let kind = hir::AssocItemConstraintKind::Equality { term: ty.into() };
        let args = arena_vec![self;];
        let constraints = arena_vec![self;];
        let gen_args = self.arena.alloc(hir::GenericArgs {
            args,
            constraints,
            parenthesized: hir::GenericArgsParentheses::No,
            span_ext: DUMMY_SP,
        });
        hir::AssocItemConstraint {
            hir_id: self.next_id(),
            gen_args,
            span: self.lower_span(span),
            ident,
            kind,
        }
    }

    /// When a bound is annotated with `async`, it signals to lowering that the trait
    /// that the bound refers to should be mapped to the "async" flavor of the trait.
    ///
    /// This only needs to be done until we unify `AsyncFn` and `Fn` traits into one
    /// that is generic over `async`ness, if that's ever possible, or modify the
    /// lowering of `async Fn()` bounds to desugar to another trait like `LendingFn`.
    fn map_trait_to_async_trait(&self, def_id: DefId) -> Option<DefId> {
        let lang_items = self.tcx.lang_items();
        match self.tcx.fn_trait_kind_from_def_id(def_id)? {
            ty::ClosureKind::Fn => lang_items.async_fn_trait(),
            ty::ClosureKind::FnMut => lang_items.async_fn_mut_trait(),
            ty::ClosureKind::FnOnce => lang_items.async_fn_once_trait(),
        }
    }
}
