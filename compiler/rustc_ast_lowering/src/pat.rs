use super::{ImplTraitContext, LoweringContext, ParamMode};

use rustc_ast::ptr::P;
use rustc_ast::*;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_span::symbol::Ident;
use rustc_span::{source_map::Spanned, Span};

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    crate fn lower_pat(&mut self, mut pattern: &Pat) -> &'hir hir::Pat<'hir> {
        ensure_sufficient_stack(|| {
            // loop here to avoid recursion
            let node = loop {
                match pattern.kind {
                    PatKind::Wild => break hir::PatKind::Wild,
                    PatKind::Ident(ref binding_mode, ident, ref sub) => {
                        let lower_sub = |this: &mut Self| sub.as_ref().map(|s| this.lower_pat(&*s));
                        break self.lower_pat_ident(pattern, binding_mode, ident, lower_sub);
                    }
                    PatKind::Lit(ref e) => break hir::PatKind::Lit(self.lower_expr(e)),
                    PatKind::TupleStruct(ref path, ref pats) => {
                        let qpath = self.lower_qpath(
                            pattern.id,
                            &None,
                            path,
                            ParamMode::Optional,
                            ImplTraitContext::disallowed(),
                        );
                        let (pats, ddpos) = self.lower_pat_tuple(pats, "tuple struct");
                        break hir::PatKind::TupleStruct(qpath, pats, ddpos);
                    }
                    PatKind::Or(ref pats) => {
                        break hir::PatKind::Or(
                            self.arena.alloc_from_iter(pats.iter().map(|x| self.lower_pat(x))),
                        );
                    }
                    PatKind::Path(ref qself, ref path) => {
                        let qpath = self.lower_qpath(
                            pattern.id,
                            qself,
                            path,
                            ParamMode::Optional,
                            ImplTraitContext::disallowed(),
                        );
                        break hir::PatKind::Path(qpath);
                    }
                    PatKind::Struct(ref path, ref fields, etc) => {
                        let qpath = self.lower_qpath(
                            pattern.id,
                            &None,
                            path,
                            ParamMode::Optional,
                            ImplTraitContext::disallowed(),
                        );

                        let fs = self.arena.alloc_from_iter(fields.iter().map(|f| hir::FieldPat {
                            hir_id: self.next_id(),
                            ident: f.ident,
                            pat: self.lower_pat(&f.pat),
                            is_shorthand: f.is_shorthand,
                            span: f.span,
                        }));
                        break hir::PatKind::Struct(qpath, fs, etc);
                    }
                    PatKind::Tuple(ref pats) => {
                        let (pats, ddpos) = self.lower_pat_tuple(pats, "tuple");
                        break hir::PatKind::Tuple(pats, ddpos);
                    }
                    PatKind::Box(ref inner) => {
                        break hir::PatKind::Box(self.lower_pat(inner));
                    }
                    PatKind::Ref(ref inner, mutbl) => {
                        break hir::PatKind::Ref(self.lower_pat(inner), mutbl);
                    }
                    PatKind::Range(ref e1, ref e2, Spanned { node: ref end, .. }) => {
                        break hir::PatKind::Range(
                            e1.as_deref().map(|e| self.lower_expr(e)),
                            e2.as_deref().map(|e| self.lower_expr(e)),
                            self.lower_range_end(end, e2.is_some()),
                        );
                    }
                    PatKind::Slice(ref pats) => break self.lower_pat_slice(pats),
                    PatKind::Rest => {
                        // If we reach here the `..` pattern is not semantically allowed.
                        break self.ban_illegal_rest_pat(pattern.span);
                    }
                    // return inner to be processed in next loop
                    PatKind::Paren(ref inner) => pattern = inner,
                    PatKind::MacCall(_) => panic!("{:?} shouldn't exist here", pattern.span),
                }
            };

            self.pat_with_node_id_of(pattern, node)
        })
    }

    fn lower_pat_tuple(
        &mut self,
        pats: &[P<Pat>],
        ctx: &str,
    ) -> (&'hir [&'hir hir::Pat<'hir>], Option<usize>) {
        let mut elems = Vec::with_capacity(pats.len());
        let mut rest = None;

        let mut iter = pats.iter().enumerate();
        for (idx, pat) in iter.by_ref() {
            // Interpret the first `..` pattern as a sub-tuple pattern.
            // Note that unlike for slice patterns,
            // where `xs @ ..` is a legal sub-slice pattern,
            // it is not a legal sub-tuple pattern.
            match pat.kind {
                // Found a sub-tuple rest pattern
                PatKind::Rest => {
                    rest = Some((idx, pat.span));
                    break;
                }
                // Found a sub-tuple pattern `$binding_mode $ident @ ..`.
                // This is not allowed as a sub-tuple pattern
                PatKind::Ident(ref _bm, ident, Some(ref sub)) if sub.is_rest() => {
                    let sp = pat.span;
                    self.diagnostic()
                        .struct_span_err(
                            sp,
                            &format!("`{} @` is not allowed in a {}", ident.name, ctx),
                        )
                        .span_label(sp, "this is only allowed in slice patterns")
                        .help("remove this and bind each tuple field independently")
                        .span_suggestion_verbose(
                            sp,
                            &format!("if you don't need to use the contents of {}, discard the tuple's remaining fields", ident),
                            "..".to_string(),
                            Applicability::MaybeIncorrect,
                        )
                        .emit();
                }
                _ => {}
            }

            // It was not a sub-tuple pattern so lower it normally.
            elems.push(self.lower_pat(pat));
        }

        for (_, pat) in iter {
            // There was a previous sub-tuple pattern; make sure we don't allow more...
            if pat.is_rest() {
                // ...but there was one again, so error.
                self.ban_extra_rest_pat(pat.span, rest.unwrap().1, ctx);
            } else {
                elems.push(self.lower_pat(pat));
            }
        }

        (self.arena.alloc_from_iter(elems), rest.map(|(ddpos, _)| ddpos))
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
        let lower_rest_sub = |this: &mut Self, pat, bm, ident, sub| {
            let lower_sub = |this: &mut Self| Some(this.pat_wild_with_node_id_of(sub));
            let node = this.lower_pat_ident(pat, bm, ident, lower_sub);
            this.pat_with_node_id_of(pat, node)
        };

        let mut iter = pats.iter();
        // Lower all the patterns until the first occurrence of a sub-slice pattern.
        for pat in iter.by_ref() {
            match pat.kind {
                // Found a sub-slice pattern `..`. Record, lower it to `_`, and stop here.
                PatKind::Rest => {
                    prev_rest_span = Some(pat.span);
                    slice = Some(self.pat_wild_with_node_id_of(pat));
                    break;
                }
                // Found a sub-slice pattern `$binding_mode $ident @ ..`.
                // Record, lower it to `$binding_mode $ident @ _`, and stop here.
                PatKind::Ident(ref bm, ident, Some(ref sub)) if sub.is_rest() => {
                    prev_rest_span = Some(sub.span);
                    slice = Some(lower_rest_sub(self, pat, bm, ident, sub));
                    break;
                }
                // It was not a subslice pattern so lower it normally.
                _ => before.push(self.lower_pat(pat)),
            }
        }

        // Lower all the patterns after the first sub-slice pattern.
        for pat in iter {
            // There was a previous subslice pattern; make sure we don't allow more.
            let rest_span = match pat.kind {
                PatKind::Rest => Some(pat.span),
                PatKind::Ident(ref bm, ident, Some(ref sub)) if sub.is_rest() => {
                    // #69103: Lower into `binding @ _` as above to avoid ICEs.
                    after.push(lower_rest_sub(self, pat, bm, ident, sub));
                    Some(sub.span)
                }
                _ => None,
            };
            if let Some(rest_span) = rest_span {
                // We have e.g., `[a, .., b, ..]`. That's no good, error!
                self.ban_extra_rest_pat(rest_span, prev_rest_span.unwrap(), "slice");
            } else {
                // Lower the pattern normally.
                after.push(self.lower_pat(pat));
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
        binding_mode: &BindingMode,
        ident: Ident,
        lower_sub: impl FnOnce(&mut Self) -> Option<&'hir hir::Pat<'hir>>,
    ) -> hir::PatKind<'hir> {
        match self.resolver.get_partial_res(p.id).map(|d| d.base_res()) {
            // `None` can occur in body-less function signatures
            res @ (None | Some(Res::Local(_))) => {
                let canonical_id = match res {
                    Some(Res::Local(id)) => id,
                    _ => p.id,
                };

                hir::PatKind::Binding(
                    self.lower_binding_mode(binding_mode),
                    self.lower_node_id(canonical_id),
                    ident,
                    lower_sub(self),
                )
            }
            Some(res) => hir::PatKind::Path(hir::QPath::Resolved(
                None,
                self.arena.alloc(hir::Path {
                    span: ident.span,
                    res: self.lower_res(res),
                    segments: arena_vec![self; hir::PathSegment::from_ident(ident)],
                }),
            )),
        }
    }

    fn lower_binding_mode(&mut self, b: &BindingMode) -> hir::BindingAnnotation {
        match *b {
            BindingMode::ByValue(Mutability::Not) => hir::BindingAnnotation::Unannotated,
            BindingMode::ByRef(Mutability::Not) => hir::BindingAnnotation::Ref,
            BindingMode::ByValue(Mutability::Mut) => hir::BindingAnnotation::Mutable,
            BindingMode::ByRef(Mutability::Mut) => hir::BindingAnnotation::RefMut,
        }
    }

    fn pat_wild_with_node_id_of(&mut self, p: &Pat) -> &'hir hir::Pat<'hir> {
        self.pat_with_node_id_of(p, hir::PatKind::Wild)
    }

    /// Construct a `Pat` with the `HirId` of `p.id` lowered.
    fn pat_with_node_id_of(&mut self, p: &Pat, kind: hir::PatKind<'hir>) -> &'hir hir::Pat<'hir> {
        self.arena.alloc(hir::Pat { hir_id: self.lower_node_id(p.id), kind, span: p.span })
    }

    /// Emit a friendly error for extra `..` patterns in a tuple/tuple struct/slice pattern.
    fn ban_extra_rest_pat(&self, sp: Span, prev_sp: Span, ctx: &str) {
        self.diagnostic()
            .struct_span_err(sp, &format!("`..` can only be used once per {} pattern", ctx))
            .span_label(sp, &format!("can only be used once per {} pattern", ctx))
            .span_label(prev_sp, "previously used here")
            .emit();
    }

    /// Used to ban the `..` pattern in places it shouldn't be semantically.
    fn ban_illegal_rest_pat(&self, sp: Span) -> hir::PatKind<'hir> {
        self.diagnostic()
            .struct_span_err(sp, "`..` patterns are not allowed here")
            .note("only allowed in tuple, tuple struct, and slice patterns")
            .emit();

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
}
