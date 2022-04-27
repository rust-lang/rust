use crate::ImplTraitPosition;

use super::{GenericArgsCtor, LifetimeRes, ParenthesizedGenericArgs};
use super::{ImplTraitContext, LoweringContext, ParamMode};

use rustc_ast::{self as ast, *};
use rustc_errors::{struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, PartialRes, Res};
use rustc_hir::GenericArg;
use rustc_span::symbol::{kw, Ident};
use rustc_span::{BytePos, Span, DUMMY_SP};

use smallvec::smallvec;
use tracing::debug;

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    crate fn lower_qpath(
        &mut self,
        id: NodeId,
        qself: &Option<QSelf>,
        p: &Path,
        param_mode: ParamMode,
        mut itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::QPath<'hir> {
        debug!("lower_qpath(id: {:?}, qself: {:?}, p: {:?})", id, qself, p);
        let qself_position = qself.as_ref().map(|q| q.position);
        let qself = qself.as_ref().map(|q| self.lower_ty(&q.ty, itctx.reborrow()));

        let partial_res =
            self.resolver.get_partial_res(id).unwrap_or_else(|| PartialRes::new(Res::Err));

        let path_span_lo = p.span.shrink_to_lo();
        let proj_start = p.segments.len() - partial_res.unresolved_segments();
        let path = self.arena.alloc(hir::Path {
            res: self.lower_res(partial_res.base_res()),
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

                    let parenthesized_generic_args = match partial_res.base_res() {
                        // `a::b::Trait(Args)`
                        Res::Def(DefKind::Trait, _) if i + 1 == proj_start => {
                            ParenthesizedGenericArgs::Ok
                        }
                        // `a::b::Trait(Args)::TraitItem`
                        Res::Def(DefKind::AssocFn, _)
                        | Res::Def(DefKind::AssocConst, _)
                        | Res::Def(DefKind::AssocTy, _)
                            if i + 2 == proj_start =>
                        {
                            ParenthesizedGenericArgs::Ok
                        }
                        // Avoid duplicated errors.
                        Res::Err => ParenthesizedGenericArgs::Ok,
                        // An error
                        _ => ParenthesizedGenericArgs::Err,
                    };

                    self.lower_path_segment(
                        p.span,
                        segment,
                        param_mode,
                        parenthesized_generic_args,
                        itctx.reborrow(),
                    )
                },
            )),
            span: self.lower_span(
                p.segments[..proj_start]
                    .last()
                    .map_or(path_span_lo, |segment| path_span_lo.to(segment.span())),
            ),
        });

        // Simple case, either no projections, or only fully-qualified.
        // E.g., `std::mem::size_of` or `<I as Iterator>::Item`.
        if partial_res.unresolved_segments() == 0 {
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
            let hir_segment = self.arena.alloc(self.lower_path_segment(
                p.span,
                segment,
                param_mode,
                ParenthesizedGenericArgs::Err,
                itctx.reborrow(),
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

        self.sess.diagnostic().span_bug(
            p.span,
            &format!(
                "lower_qpath: no final extension segment in {}..{}",
                proj_start,
                p.segments.len()
            ),
        );
    }

    crate fn lower_path_extra(
        &mut self,
        res: Res,
        p: &Path,
        param_mode: ParamMode,
    ) -> &'hir hir::Path<'hir> {
        self.arena.alloc(hir::Path {
            res,
            segments: self.arena.alloc_from_iter(p.segments.iter().map(|segment| {
                self.lower_path_segment(
                    p.span,
                    segment,
                    param_mode,
                    ParenthesizedGenericArgs::Err,
                    ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                )
            })),
            span: self.lower_span(p.span),
        })
    }

    crate fn lower_path(
        &mut self,
        id: NodeId,
        p: &Path,
        param_mode: ParamMode,
    ) -> &'hir hir::Path<'hir> {
        let res = self.expect_full_res(id);
        let res = self.lower_res(res);
        self.lower_path_extra(res, p, param_mode)
    }

    crate fn lower_path_segment(
        &mut self,
        path_span: Span,
        segment: &PathSegment,
        param_mode: ParamMode,
        parenthesized_generic_args: ParenthesizedGenericArgs,
        itctx: ImplTraitContext<'_, 'hir>,
    ) -> hir::PathSegment<'hir> {
        debug!("path_span: {:?}, lower_path_segment(segment: {:?})", path_span, segment,);
        let (mut generic_args, infer_args) = if let Some(ref generic_args) = segment.args {
            let msg = "parenthesized type parameters may only be used with a `Fn` trait";
            match **generic_args {
                GenericArgs::AngleBracketed(ref data) => {
                    self.lower_angle_bracketed_parameter_data(data, param_mode, itctx)
                }
                GenericArgs::Parenthesized(ref data) => match parenthesized_generic_args {
                    ParenthesizedGenericArgs::Ok => {
                        self.lower_parenthesized_parameter_data(segment.id, data)
                    }
                    ParenthesizedGenericArgs::Err => {
                        let mut err = struct_span_err!(self.sess, data.span, E0214, "{}", msg);
                        err.span_label(data.span, "only `Fn` traits may use parentheses");
                        if let Ok(snippet) = self.sess.source_map().span_to_snippet(data.span) {
                            // Do not suggest going from `Trait()` to `Trait<>`
                            if !data.inputs.is_empty() {
                                // Suggest replacing `(` and `)` with `<` and `>`
                                // The snippet may be missing the closing `)`, skip that case
                                if snippet.ends_with(')') {
                                    if let Some(split) = snippet.find('(') {
                                        let trait_name = &snippet[0..split];
                                        let args = &snippet[split + 1..snippet.len() - 1];
                                        err.span_suggestion(
                                            data.span,
                                            "use angle brackets instead",
                                            format!("{}<{}>", trait_name, args),
                                            Applicability::MaybeIncorrect,
                                        );
                                    }
                                }
                            }
                        };
                        err.emit();
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
            }
        } else {
            (
                GenericArgsCtor {
                    args: Default::default(),
                    bindings: &[],
                    parenthesized: false,
                    span: path_span.shrink_to_hi(),
                },
                param_mode == ParamMode::Optional,
            )
        };

        let has_lifetimes =
            generic_args.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)));
        if !generic_args.parenthesized && !has_lifetimes {
            self.maybe_insert_elided_lifetimes_in_path(
                path_span,
                segment.id,
                segment.ident.span,
                &mut generic_args,
            );
        }

        let res = self.expect_full_res(segment.id);
        let id = self.lower_node_id(segment.id);
        debug!(
            "lower_path_segment: ident={:?} original-id={:?} new-id={:?}",
            segment.ident, segment.id, id,
        );

        hir::PathSegment {
            ident: self.lower_ident(segment.ident),
            hir_id: Some(id),
            res: Some(self.lower_res(res)),
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
            Some(_) => panic!(),
        };
        let expected_lifetimes = end.as_usize() - start.as_usize();
        debug!(expected_lifetimes);

        // Note: these spans are used for diagnostics when they can't be inferred.
        // See rustc_resolve::late::lifetimes::LifetimeContext::add_missing_lifetime_specifiers_label
        let elided_lifetime_span = if generic_args.span.is_empty() {
            // If there are no brackets, use the identifier span.
            // HACK: we use find_ancestor_inside to properly suggest elided spans in paths
            // originating from macros, since the segment's span might be from a macro arg.
            segment_ident_span.find_ancestor_inside(path_span).unwrap_or(path_span)
        } else if generic_args.is_empty() {
            // If there are brackets, but not generic arguments, then use the opening bracket
            generic_args.span.with_hi(generic_args.span.lo() + BytePos(1))
        } else {
            // Else use an empty span right after the opening bracket.
            generic_args.span.with_lo(generic_args.span.lo() + BytePos(1)).shrink_to_lo()
        };

        generic_args.args.insert_many(
            0,
            (start.as_u32()..end.as_u32()).map(|i| {
                let id = NodeId::from_u32(i);
                let l = self.lower_lifetime(&Lifetime {
                    id,
                    ident: Ident::new(kw::UnderscoreLifetime, elided_lifetime_span),
                });
                GenericArg::Lifetime(l)
            }),
        );
    }

    pub(crate) fn lower_angle_bracketed_parameter_data(
        &mut self,
        data: &AngleBracketedArgs,
        param_mode: ParamMode,
        mut itctx: ImplTraitContext<'_, 'hir>,
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
                AngleBracketedArg::Arg(arg) => Some(self.lower_generic_arg(arg, itctx.reborrow())),
                AngleBracketedArg::Constraint(_) => None,
            })
            .collect();
        let bindings = self.arena.alloc_from_iter(data.args.iter().filter_map(|arg| match arg {
            AngleBracketedArg::Constraint(c) => {
                Some(self.lower_assoc_ty_constraint(c, itctx.reborrow()))
            }
            AngleBracketedArg::Arg(_) => None,
        }));
        let ctor = GenericArgsCtor { args, bindings, parenthesized: false, span: data.span };
        (ctor, !has_non_lt_args && param_mode == ParamMode::Optional)
    }

    fn lower_parenthesized_parameter_data(
        &mut self,
        id: NodeId,
        data: &ParenthesizedArgs,
    ) -> (GenericArgsCtor<'hir>, bool) {
        // Switch to `PassThrough` mode for anonymous lifetimes; this
        // means that we permit things like `&Ref<T>`, where `Ref` has
        // a hidden lifetime parameter. This is needed for backwards
        // compatibility, even in contexts like an impl header where
        // we generally don't permit such things (see #51008).
        self.with_lifetime_binder(id, |this| {
            let ParenthesizedArgs { span, inputs, inputs_span, output } = data;
            let inputs = this.arena.alloc_from_iter(inputs.iter().map(|ty| {
                this.lower_ty_direct(
                    ty,
                    ImplTraitContext::Disallowed(ImplTraitPosition::FnTraitParam),
                )
            }));
            let output_ty = match output {
                FnRetTy::Ty(ty) => this
                    .lower_ty(&ty, ImplTraitContext::Disallowed(ImplTraitPosition::FnTraitReturn)),
                FnRetTy::Default(_) => this.arena.alloc(this.ty_tup(*span, &[])),
            };
            let args = smallvec![GenericArg::Type(this.ty_tup(*inputs_span, inputs))];
            let binding = this.output_ty_binding(output_ty.span, output_ty);
            (
                GenericArgsCtor {
                    args,
                    bindings: arena_vec![this; binding],
                    parenthesized: true,
                    span: data.inputs_span,
                },
                false,
            )
        })
    }

    /// An associated type binding `Output = $ty`.
    crate fn output_ty_binding(
        &mut self,
        span: Span,
        ty: &'hir hir::Ty<'hir>,
    ) -> hir::TypeBinding<'hir> {
        let ident = Ident::with_dummy_span(hir::FN_OUTPUT_NAME);
        let kind = hir::TypeBindingKind::Equality { term: ty.into() };
        let args = arena_vec![self;];
        let bindings = arena_vec![self;];
        let gen_args = self.arena.alloc(hir::GenericArgs {
            args,
            bindings,
            parenthesized: false,
            span_ext: DUMMY_SP,
        });
        hir::TypeBinding {
            hir_id: self.next_id(),
            gen_args,
            span: self.lower_span(span),
            ident,
            kind,
        }
    }
}
