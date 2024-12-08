use rustc_ast::ptr::P;
use rustc_ast::*;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_span::Span;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Ident;

use super::errors::{
    ArbitraryExpressionInPattern, ExtraDoubleDot, MisplacedDoubleDot, SubTupleBinding,
};
use super::{ImplTraitContext, LoweringContext, ParamMode, ResolverAstLoweringExt};
use crate::{AllowReturnTypeNotation, ImplTraitPosition};

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    pub(crate) fn lower_pat(&mut self, pattern: &Pat) -> &'hir hir::Pat<'hir> {
        self.arena.alloc(self.lower_pat_mut(pattern))
    }

    fn lower_pat_mut(&mut self, mut pattern: &Pat) -> hir::Pat<'hir> {
        ensure_sufficient_stack(|| {
            // loop here to avoid recursion
            let pat_hir_id = self.lower_node_id(pattern.id);
            let node = loop {
                match &pattern.kind {
                    PatKind::Wild => break hir::PatKind::Wild,
                    PatKind::Never => break hir::PatKind::Never,
                    PatKind::Ident(binding_mode, ident, sub) => {
                        let lower_sub = |this: &mut Self| sub.as_ref().map(|s| this.lower_pat(s));
                        break self.lower_pat_ident(
                            pattern,
                            *binding_mode,
                            *ident,
                            pat_hir_id,
                            lower_sub,
                        );
                    }
                    PatKind::Lit(e) => {
                        break hir::PatKind::Lit(self.lower_expr_within_pat(e, false));
                    }
                    PatKind::TupleStruct(qself, path, pats) => {
                        let qpath = self.lower_qpath(
                            pattern.id,
                            qself,
                            path,
                            ParamMode::Optional,
                            AllowReturnTypeNotation::No,
                            ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                            None,
                        );
                        let (pats, ddpos) = self.lower_pat_tuple(pats, "tuple struct");
                        break hir::PatKind::TupleStruct(qpath, pats, ddpos);
                    }
                    PatKind::Or(pats) => {
                        break hir::PatKind::Or(
                            self.arena.alloc_from_iter(pats.iter().map(|x| self.lower_pat_mut(x))),
                        );
                    }
                    PatKind::Path(qself, path) => {
                        let qpath = self.lower_qpath(
                            pattern.id,
                            qself,
                            path,
                            ParamMode::Optional,
                            AllowReturnTypeNotation::No,
                            ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                            None,
                        );
                        break hir::PatKind::Path(qpath);
                    }
                    PatKind::Struct(qself, path, fields, etc) => {
                        let qpath = self.lower_qpath(
                            pattern.id,
                            qself,
                            path,
                            ParamMode::Optional,
                            AllowReturnTypeNotation::No,
                            ImplTraitContext::Disallowed(ImplTraitPosition::Path),
                            None,
                        );

                        let fs = self.arena.alloc_from_iter(fields.iter().map(|f| {
                            let hir_id = self.lower_node_id(f.id);
                            self.lower_attrs(hir_id, &f.attrs);

                            hir::PatField {
                                hir_id,
                                ident: self.lower_ident(f.ident),
                                pat: self.lower_pat(&f.pat),
                                is_shorthand: f.is_shorthand,
                                span: self.lower_span(f.span),
                            }
                        }));
                        break hir::PatKind::Struct(qpath, fs, *etc == ast::PatFieldsRest::Rest);
                    }
                    PatKind::Tuple(pats) => {
                        let (pats, ddpos) = self.lower_pat_tuple(pats, "tuple");
                        break hir::PatKind::Tuple(pats, ddpos);
                    }
                    PatKind::Box(inner) => {
                        break hir::PatKind::Box(self.lower_pat(inner));
                    }
                    PatKind::Deref(inner) => {
                        break hir::PatKind::Deref(self.lower_pat(inner));
                    }
                    PatKind::Ref(inner, mutbl) => {
                        break hir::PatKind::Ref(self.lower_pat(inner), *mutbl);
                    }
                    PatKind::Range(e1, e2, Spanned { node: end, .. }) => {
                        break hir::PatKind::Range(
                            e1.as_deref().map(|e| self.lower_expr_within_pat(e, true)),
                            e2.as_deref().map(|e| self.lower_expr_within_pat(e, true)),
                            self.lower_range_end(end, e2.is_some()),
                        );
                    }
                    PatKind::Slice(pats) => break self.lower_pat_slice(pats),
                    PatKind::Rest => {
                        // If we reach here the `..` pattern is not semantically allowed.
                        break self.ban_illegal_rest_pat(pattern.span);
                    }
                    // return inner to be processed in next loop
                    PatKind::Paren(inner) => pattern = inner,
                    PatKind::MacCall(_) => panic!("{:?} shouldn't exist here", pattern.span),
                    PatKind::Err(guar) => break hir::PatKind::Err(*guar),
                }
            };

            self.pat_with_node_id_of(pattern, node, pat_hir_id)
        })
    }

    fn lower_pat_tuple(
        &mut self,
        pats: &[P<Pat>],
        ctx: &str,
    ) -> (&'hir [hir::Pat<'hir>], hir::DotDotPos) {
        let mut elems = Vec::with_capacity(pats.len());
        let mut rest = None;

        let mut iter = pats.iter().enumerate();
        for (idx, pat) in iter.by_ref() {
            // Interpret the first `..` pattern as a sub-tuple pattern.
            // Note that unlike for slice patterns,
            // where `xs @ ..` is a legal sub-slice pattern,
            // it is not a legal sub-tuple pattern.
            match &pat.kind {
                // Found a sub-tuple rest pattern
                PatKind::Rest => {
                    rest = Some((idx, pat.span));
                    break;
                }
                // Found a sub-tuple pattern `$binding_mode $ident @ ..`.
                // This is not allowed as a sub-tuple pattern
                PatKind::Ident(_, ident, Some(sub)) if sub.is_rest() => {
                    let sp = pat.span;
                    self.dcx().emit_err(SubTupleBinding {
                        span: sp,
                        ident_name: ident.name,
                        ident: *ident,
                        ctx,
                    });
                }
                _ => {}
            }

            // It was not a sub-tuple pattern so lower it normally.
            elems.push(self.lower_pat_mut(pat));
        }

        for (_, pat) in iter {
            // There was a previous sub-tuple pattern; make sure we don't allow more...
            if pat.is_rest() {
                // ...but there was one again, so error.
                self.ban_extra_rest_pat(pat.span, rest.unwrap().1, ctx);
            } else {
                elems.push(self.lower_pat_mut(pat));
            }
        }

        (self.arena.alloc_from_iter(elems), hir::DotDotPos::new(rest.map(|(ddpos, _)| ddpos)))
    }

    /// Lower a slice pattern of form `[pat_0, ..., pat_n]` into
    /// `hir::PatKind::Slice(before, slice, after)`.
    ///
    /// When encountering `($binding_mode $ident @)? ..` (`slice`),
    /// this is interpreted as a sub-slice pattern semantically.
    /// Patterns that follow, which are not like `slice` -- or an error occurs, are in `after`.
    fn lower_pat_slice(&mut self, pats: &[P<Pat>]) -> hir::PatKind<'hir> {
        let mut before = Vec::new();
        let mut after = Vec::new();
        let mut slice = None;
        let mut prev_rest_span = None;

        // Lowers `$bm $ident @ ..` to `$bm $ident @ _`.
        let lower_rest_sub = |this: &mut Self, pat: &Pat, &ann, &ident, sub: &Pat| {
            let sub_hir_id = this.lower_node_id(sub.id);
            let lower_sub = |this: &mut Self| Some(this.pat_wild_with_node_id_of(sub, sub_hir_id));
            let pat_hir_id = this.lower_node_id(pat.id);
            let node = this.lower_pat_ident(pat, ann, ident, pat_hir_id, lower_sub);
            this.pat_with_node_id_of(pat, node, pat_hir_id)
        };

        let mut iter = pats.iter();
        // Lower all the patterns until the first occurrence of a sub-slice pattern.
        for pat in iter.by_ref() {
            match &pat.kind {
                // Found a sub-slice pattern `..`. Record, lower it to `_`, and stop here.
                PatKind::Rest => {
                    prev_rest_span = Some(pat.span);
                    let hir_id = self.lower_node_id(pat.id);
                    slice = Some(self.pat_wild_with_node_id_of(pat, hir_id));
                    break;
                }
                // Found a sub-slice pattern `$binding_mode $ident @ ..`.
                // Record, lower it to `$binding_mode $ident @ _`, and stop here.
                PatKind::Ident(ann, ident, Some(sub)) if sub.is_rest() => {
                    prev_rest_span = Some(sub.span);
                    slice = Some(self.arena.alloc(lower_rest_sub(self, pat, ann, ident, sub)));
                    break;
                }
                // It was not a subslice pattern so lower it normally.
                _ => before.push(self.lower_pat_mut(pat)),
            }
        }

        // Lower all the patterns after the first sub-slice pattern.
        for pat in iter {
            // There was a previous subslice pattern; make sure we don't allow more.
            let rest_span = match &pat.kind {
                PatKind::Rest => Some(pat.span),
                PatKind::Ident(ann, ident, Some(sub)) if sub.is_rest() => {
                    // #69103: Lower into `binding @ _` as above to avoid ICEs.
                    after.push(lower_rest_sub(self, pat, ann, ident, sub));
                    Some(sub.span)
                }
                _ => None,
            };
            if let Some(rest_span) = rest_span {
                // We have e.g., `[a, .., b, ..]`. That's no good, error!
                self.ban_extra_rest_pat(rest_span, prev_rest_span.unwrap(), "slice");
            } else {
                // Lower the pattern normally.
                after.push(self.lower_pat_mut(pat));
            }
        }

        hir::PatKind::Slice(
            self.arena.alloc_from_iter(before),
            slice,
            self.arena.alloc_from_iter(after),
        )
    }

    fn lower_pat_ident(
        &mut self,
        p: &Pat,
        annotation: BindingMode,
        ident: Ident,
        hir_id: hir::HirId,
        lower_sub: impl FnOnce(&mut Self) -> Option<&'hir hir::Pat<'hir>>,
    ) -> hir::PatKind<'hir> {
        match self.resolver.get_partial_res(p.id).map(|d| d.expect_full_res()) {
            // `None` can occur in body-less function signatures
            res @ (None | Some(Res::Local(_))) => {
                let binding_id = match res {
                    Some(Res::Local(id)) => {
                        // In `Or` patterns like `VariantA(s) | VariantB(s, _)`, multiple identifier patterns
                        // will be resolved to the same `Res::Local`. Thus they just share a single
                        // `HirId`.
                        if id == p.id {
                            self.ident_and_label_to_local_id.insert(id, hir_id.local_id);
                            hir_id
                        } else {
                            hir::HirId {
                                owner: self.current_hir_id_owner,
                                local_id: self.ident_and_label_to_local_id[&id],
                            }
                        }
                    }
                    _ => {
                        self.ident_and_label_to_local_id.insert(p.id, hir_id.local_id);
                        hir_id
                    }
                };
                hir::PatKind::Binding(
                    annotation,
                    binding_id,
                    self.lower_ident(ident),
                    lower_sub(self),
                )
            }
            Some(res) => {
                let hir_id = self.next_id();
                let res = self.lower_res(res);
                hir::PatKind::Path(hir::QPath::Resolved(
                    None,
                    self.arena.alloc(hir::Path {
                        span: self.lower_span(ident.span),
                        res,
                        segments: arena_vec![self; hir::PathSegment::new(self.lower_ident(ident), hir_id, res)],
                    }),
            ))
            }
        }
    }

    fn pat_wild_with_node_id_of(&mut self, p: &Pat, hir_id: hir::HirId) -> &'hir hir::Pat<'hir> {
        self.arena.alloc(self.pat_with_node_id_of(p, hir::PatKind::Wild, hir_id))
    }

    /// Construct a `Pat` with the `HirId` of `p.id` already lowered.
    fn pat_with_node_id_of(
        &mut self,
        p: &Pat,
        kind: hir::PatKind<'hir>,
        hir_id: hir::HirId,
    ) -> hir::Pat<'hir> {
        hir::Pat { hir_id, kind, span: self.lower_span(p.span), default_binding_modes: true }
    }

    /// Emit a friendly error for extra `..` patterns in a tuple/tuple struct/slice pattern.
    pub(crate) fn ban_extra_rest_pat(&self, sp: Span, prev_sp: Span, ctx: &str) {
        self.dcx().emit_err(ExtraDoubleDot { span: sp, prev_span: prev_sp, ctx });
    }

    /// Used to ban the `..` pattern in places it shouldn't be semantically.
    fn ban_illegal_rest_pat(&self, sp: Span) -> hir::PatKind<'hir> {
        self.dcx().emit_err(MisplacedDoubleDot { span: sp });

        // We're not in a list context so `..` can be reasonably treated
        // as `_` because it should always be valid and roughly matches the
        // intent of `..` (notice that the rest of a single slot is that slot).
        hir::PatKind::Wild
    }

    fn lower_range_end(&mut self, e: &RangeEnd, has_end: bool) -> hir::RangeEnd {
        match *e {
            RangeEnd::Excluded if has_end => hir::RangeEnd::Excluded,
            // No end; so `X..` behaves like `RangeFrom`.
            RangeEnd::Excluded | RangeEnd::Included(_) => hir::RangeEnd::Included,
        }
    }

    /// Matches `'-' lit | lit (cf. parser::Parser::parse_literal_maybe_minus)`,
    /// or paths for ranges.
    //
    // FIXME: do we want to allow `expr -> pattern` conversion to create path expressions?
    // That means making this work:
    //
    // ```rust,ignore (FIXME)
    // struct S;
    // macro_rules! m {
    //     ($a:expr) => {
    //         let $a = S;
    //     }
    // }
    // m!(S);
    // ```
    fn lower_expr_within_pat(&mut self, expr: &Expr, allow_paths: bool) -> &'hir hir::Expr<'hir> {
        match &expr.kind {
            ExprKind::Lit(..)
            | ExprKind::ConstBlock(..)
            | ExprKind::IncludedBytes(..)
            | ExprKind::Err(_)
            | ExprKind::Dummy => {}
            ExprKind::Path(..) if allow_paths => {}
            ExprKind::Unary(UnOp::Neg, inner) if matches!(inner.kind, ExprKind::Lit(_)) => {}
            _ => {
                let pattern_from_macro = expr.is_approximately_pattern();
                let guar = self.dcx().emit_err(ArbitraryExpressionInPattern {
                    span: expr.span,
                    pattern_from_macro_note: pattern_from_macro,
                });
                return self.arena.alloc(self.expr_err(expr.span, guar));
            }
        }
        self.lower_expr(expr)
    }
}
