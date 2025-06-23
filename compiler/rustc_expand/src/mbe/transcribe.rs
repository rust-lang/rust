use std::mem;

use rustc_ast::token::{
    self, Delimiter, IdentIsRaw, InvisibleOrigin, Lit, LitKind, MetaVarKind, Token, TokenKind,
};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast::{ExprKind, StmtKind, TyKind, UnOp};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Diag, DiagCtxtHandle, PResult, pluralize};
use rustc_parse::lexer::nfc_normalize;
use rustc_parse::parser::ParseNtResult;
use rustc_session::parse::ParseSess;
use rustc_span::hygiene::{LocalExpnId, Transparency};
use rustc_span::{
    Ident, MacroRulesNormalizedIdent, Span, Symbol, SyntaxContext, sym, with_metavar_spans,
};
use smallvec::{SmallVec, smallvec};

use crate::errors::{
    CountRepetitionMisplaced, MetaVarExprUnrecognizedVar, MetaVarsDifSeqMatchers, MustRepeatOnce,
    NoSyntaxVarsExprRepeat, VarStillRepeating,
};
use crate::mbe::macro_parser::NamedMatch;
use crate::mbe::macro_parser::NamedMatch::*;
use crate::mbe::metavar_expr::{MetaVarExprConcatElem, RAW_IDENT_ERR};
use crate::mbe::{self, KleeneOp, MetaVarExpr};

/// Context needed to perform transcription of metavariable expressions.
struct TranscrCtx<'psess, 'itp> {
    psess: &'psess ParseSess,

    /// Map from metavars to matched tokens
    interp: &'itp FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,

    /// Allow marking spans.
    marker: Marker,

    /// The stack of things yet to be completely expanded.
    ///
    /// We descend into the RHS (`src`), expanding things as we go. This stack contains the things
    /// we have yet to expand/are still expanding. We start the stack off with the whole RHS. The
    /// choice of spacing values doesn't matter.
    stack: SmallVec<[Frame<'itp>; 1]>,

    /// A stack of where we are in the repeat expansion.
    ///
    /// As we descend in the RHS, we will need to be able to match nested sequences of matchers.
    /// `repeats` keeps track of where we are in matching at each level, with the last element
    /// being the most deeply nested sequence. This is used as a stack.
    repeats: Vec<(usize, usize)>,

    /// The resulting token stream from the `TokenTree` we just finished processing.
    ///
    /// At the end, this will contain the full result of transcription, but at arbitrary points
    /// during `transcribe`, `result` will contain subsets of the final result.
    ///
    /// Specifically, as we descend into each TokenTree, we will push the existing results onto the
    /// `result_stack` and clear `results`. We will then produce the results of transcribing the
    /// TokenTree into `results`. Then, as we unwind back out of the `TokenTree`, we will pop the
    /// `result_stack` and append `results` too it to produce the new `results` up to that point.
    ///
    /// Thus, if we try to pop the `result_stack` and it is empty, we have reached the top-level
    /// again, and we are done transcribing.
    result: Vec<TokenTree>,

    /// The in-progress `result` lives at the top of this stack. Each entered `TokenTree` adds a
    /// new entry.
    result_stack: Vec<Vec<TokenTree>>,
}

impl<'psess> TranscrCtx<'psess, '_> {
    /// Span marked with the correct expansion and transparency.
    fn visited_dspan(&mut self, dspan: DelimSpan) -> Span {
        let mut span = dspan.entire();
        self.marker.mark_span(&mut span);
        span
    }
}

/// A Marker adds the given mark to the syntax context.
struct Marker {
    expand_id: LocalExpnId,
    transparency: Transparency,
    cache: FxHashMap<SyntaxContext, SyntaxContext>,
}

impl Marker {
    /// Mark a span with the stored expansion ID and transparency.
    fn mark_span(&mut self, span: &mut Span) {
        // `apply_mark` is a relatively expensive operation, both due to taking hygiene lock, and
        // by itself. All tokens in a macro body typically have the same syntactic context, unless
        // it's some advanced case with macro-generated macros. So if we cache the marked version
        // of that context once, we'll typically have a 100% cache hit rate after that.
        *span = span.map_ctxt(|ctxt| {
            *self
                .cache
                .entry(ctxt)
                .or_insert_with(|| ctxt.apply_mark(self.expand_id.to_expn_id(), self.transparency))
        });
    }
}

/// An iterator over the token trees in a delimited token tree (`{ ... }`) or a sequence (`$(...)`).
struct Frame<'a> {
    tts: &'a [mbe::TokenTree],
    idx: usize,
    kind: FrameKind,
}

enum FrameKind {
    Delimited { delim: Delimiter, span: DelimSpan, spacing: DelimSpacing },
    Sequence { sep: Option<Token>, kleene_op: KleeneOp },
}

impl<'a> Frame<'a> {
    fn new_delimited(src: &'a mbe::Delimited, span: DelimSpan, spacing: DelimSpacing) -> Frame<'a> {
        Frame {
            tts: &src.tts,
            idx: 0,
            kind: FrameKind::Delimited { delim: src.delim, span, spacing },
        }
    }

    fn new_sequence(
        src: &'a mbe::SequenceRepetition,
        sep: Option<Token>,
        kleene_op: KleeneOp,
    ) -> Frame<'a> {
        Frame { tts: &src.tts, idx: 0, kind: FrameKind::Sequence { sep, kleene_op } }
    }
}

impl<'a> Iterator for Frame<'a> {
    type Item = &'a mbe::TokenTree;

    fn next(&mut self) -> Option<&'a mbe::TokenTree> {
        let res = self.tts.get(self.idx);
        self.idx += 1;
        res
    }
}

/// This can do Macro-By-Example transcription.
/// - `interp` is a map of meta-variables to the tokens (non-terminals) they matched in the
///   invocation. We are assuming we already know there is a match.
/// - `src` is the RHS of the MBE, that is, the "example" we are filling in.
///
/// For example,
///
/// ```rust
/// macro_rules! foo {
///     ($id:ident) => { println!("{}", stringify!($id)); }
/// }
///
/// foo!(bar);
/// ```
///
/// `interp` would contain `$id => bar` and `src` would contain `println!("{}", stringify!($id));`.
///
/// `transcribe` would return a `TokenStream` containing `println!("{}", stringify!(bar));`.
///
/// Along the way, we do some additional error checking.
pub(super) fn transcribe<'a>(
    psess: &'a ParseSess,
    interp: &FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    src: &mbe::Delimited,
    src_span: DelimSpan,
    transparency: Transparency,
    expand_id: LocalExpnId,
) -> PResult<'a, TokenStream> {
    // Nothing for us to transcribe...
    if src.tts.is_empty() {
        return Ok(TokenStream::default());
    }

    let mut tscx = TranscrCtx {
        psess,
        interp,
        marker: Marker { expand_id, transparency, cache: Default::default() },
        repeats: Vec::new(),
        stack: smallvec![Frame::new_delimited(
            src,
            src_span,
            DelimSpacing::new(Spacing::Alone, Spacing::Alone)
        )],
        result: Vec::new(),
        result_stack: Vec::new(),
    };

    loop {
        // Look at the last frame on the stack.
        // If it still has a TokenTree we have not looked at yet, use that tree.
        let Some(tree) = tscx.stack.last_mut().unwrap().next() else {
            // This else-case never produces a value for `tree` (it `continue`s or `return`s).

            // Otherwise, if we have just reached the end of a sequence and we can keep repeating,
            // go back to the beginning of the sequence.
            let frame = tscx.stack.last_mut().unwrap();
            if let FrameKind::Sequence { sep, .. } = &frame.kind {
                let (repeat_idx, repeat_len) = tscx.repeats.last_mut().unwrap();
                *repeat_idx += 1;
                if repeat_idx < repeat_len {
                    frame.idx = 0;
                    if let Some(sep) = sep {
                        tscx.result.push(TokenTree::Token(*sep, Spacing::Alone));
                    }
                    continue;
                }
            }

            // We are done with the top of the stack. Pop it. Depending on what it was, we do
            // different things. Note that the outermost item must be the delimited, wrapped RHS
            // that was passed in originally to `transcribe`.
            match tscx.stack.pop().unwrap().kind {
                // Done with a sequence. Pop from repeats.
                FrameKind::Sequence { .. } => {
                    tscx.repeats.pop();
                }

                // We are done processing a Delimited. If this is the top-level delimited, we are
                // done. Otherwise, we unwind the result_stack to append what we have produced to
                // any previous results.
                FrameKind::Delimited { delim, span, mut spacing, .. } => {
                    // Hack to force-insert a space after `]` in certain case.
                    // See discussion of the `hex-literal` crate in #114571.
                    if delim == Delimiter::Bracket {
                        spacing.close = Spacing::Alone;
                    }
                    if tscx.result_stack.is_empty() {
                        // No results left to compute! We are back at the top-level.
                        return Ok(TokenStream::new(tscx.result));
                    }

                    // Step back into the parent Delimited.
                    let tree =
                        TokenTree::Delimited(span, spacing, delim, TokenStream::new(tscx.result));
                    tscx.result = tscx.result_stack.pop().unwrap();
                    tscx.result.push(tree);
                }
            }
            continue;
        };

        // At this point, we know we are in the middle of a TokenTree (the last one on `stack`).
        // `tree` contains the next `TokenTree` to be processed.
        match tree {
            // Replace the sequence with its expansion.
            seq @ mbe::TokenTree::Sequence(_, seq_rep) => {
                transcribe_sequence(&mut tscx, seq, seq_rep)?;
            }

            // Replace the meta-var with the matched token tree from the invocation.
            &mbe::TokenTree::MetaVar(sp, original_ident) => {
                transcribe_metavar(&mut tscx, sp, original_ident)?;
            }

            // Replace meta-variable expressions with the result of their expansion.
            mbe::TokenTree::MetaVarExpr(dspan, expr) => {
                transcribe_metavar_expr(&mut tscx, *dspan, expr)?;
            }

            // If we are entering a new delimiter, we push its contents to the `stack` to be
            // processed, and we push all of the currently produced results to the `result_stack`.
            // We will produce all of the results of the inside of the `Delimited` and then we will
            // jump back out of the Delimited, pop the result_stack and add the new results back to
            // the previous results (from outside the Delimited).
            &mbe::TokenTree::Delimited(mut span, ref spacing, ref delimited) => {
                tscx.marker.mark_span(&mut span.open);
                tscx.marker.mark_span(&mut span.close);
                tscx.stack.push(Frame::new_delimited(delimited, span, *spacing));
                tscx.result_stack.push(mem::take(&mut tscx.result));
            }

            // Nothing much to do here. Just push the token to the result, being careful to
            // preserve syntax context.
            &mbe::TokenTree::Token(mut token) => {
                tscx.marker.mark_span(&mut token.span);
                if let token::NtIdent(ident, _) | token::NtLifetime(ident, _) = &mut token.kind {
                    tscx.marker.mark_span(&mut ident.span);
                }
                let tt = TokenTree::Token(token, Spacing::Alone);
                tscx.result.push(tt);
            }

            // There should be no meta-var declarations in the invocation of a macro.
            mbe::TokenTree::MetaVarDecl(..) => panic!("unexpected `TokenTree::MetaVarDecl`"),
        }
    }
}

/// Turn `$(...)*` sequences into tokens.
fn transcribe_sequence<'tx, 'itp>(
    tscx: &mut TranscrCtx<'tx, 'itp>,
    seq: &mbe::TokenTree,
    seq_rep: &'itp mbe::SequenceRepetition,
) -> PResult<'tx, ()> {
    let dcx = tscx.psess.dcx();

    // We are descending into a sequence. We first make sure that the matchers in the RHS
    // and the matches in `interp` have the same shape. Otherwise, either the caller or the
    // macro writer has made a mistake.
    match lockstep_iter_size(seq, tscx.interp, &tscx.repeats) {
        LockstepIterSize::Unconstrained => {
            return Err(dcx.create_err(NoSyntaxVarsExprRepeat { span: seq.span() }));
        }

        LockstepIterSize::Contradiction(msg) => {
            // FIXME: this really ought to be caught at macro definition time... It
            // happens when two meta-variables are used in the same repetition in a
            // sequence, but they come from different sequence matchers and repeat
            // different amounts.
            return Err(dcx.create_err(MetaVarsDifSeqMatchers { span: seq.span(), msg }));
        }

        LockstepIterSize::Constraint(len, _) => {
            // We do this to avoid an extra clone above. We know that this is a
            // sequence already.
            let mbe::TokenTree::Sequence(sp, seq) = seq else { unreachable!() };

            // Is the repetition empty?
            if len == 0 {
                if seq.kleene.op == KleeneOp::OneOrMore {
                    // FIXME: this really ought to be caught at macro definition
                    // time... It happens when the Kleene operator in the matcher and
                    // the body for the same meta-variable do not match.
                    return Err(dcx.create_err(MustRepeatOnce { span: sp.entire() }));
                }
            } else {
                // 0 is the initial counter (we have done 0 repetitions so far). `len`
                // is the total number of repetitions we should generate.
                tscx.repeats.push((0, len));

                // The first time we encounter the sequence we push it to the stack. It
                // then gets reused (see the beginning of the loop) until we are done
                // repeating.
                tscx.stack.push(Frame::new_sequence(seq_rep, seq.separator.clone(), seq.kleene.op));
            }
        }
    }

    Ok(())
}

/// Find the matched nonterminal from the macro invocation, and use it to replace
/// the meta-var.
///
/// We use `Spacing::Alone` everywhere here, because that's the conservative choice
/// and spacing of declarative macros is tricky. E.g. in this macro:
/// ```
/// macro_rules! idents {
///     ($($a:ident,)*) => { stringify!($($a)*) }
/// }
/// ```
/// `$a` has no whitespace after it and will be marked `JointHidden`. If you then
/// call `idents!(x,y,z,)`, each of `x`, `y`, and `z` will be marked as `Joint`. So
/// if you choose to use `$x`'s spacing or the identifier's spacing, you'll end up
/// producing "xyz", which is bad because it effectively merges tokens.
/// `Spacing::Alone` is the safer option. Fortunately, `space_between` will avoid
/// some of the unnecessary whitespace.
fn transcribe_metavar<'tx>(
    tscx: &mut TranscrCtx<'tx, '_>,
    mut sp: Span,
    mut original_ident: Ident,
) -> PResult<'tx, ()> {
    let dcx = tscx.psess.dcx();

    let ident = MacroRulesNormalizedIdent::new(original_ident);
    let Some(cur_matched) = lookup_cur_matched(ident, tscx.interp, &tscx.repeats) else {
        // If we aren't able to match the meta-var, we push it back into the result but
        // with modified syntax context. (I believe this supports nested macros).
        tscx.marker.mark_span(&mut sp);
        tscx.marker.mark_span(&mut original_ident.span);
        tscx.result.push(TokenTree::token_joint_hidden(token::Dollar, sp));
        tscx.result.push(TokenTree::Token(Token::from_ast_ident(original_ident), Spacing::Alone));
        return Ok(());
    };

    // We wrap the tokens in invisible delimiters, unless they are already wrapped
    // in invisible delimiters with the same `MetaVarKind`. Because some proc
    // macros can't handle multiple layers of invisible delimiters of the same
    // `MetaVarKind`. This loses some span info, though it hopefully won't matter.
    let mut mk_delimited = |mk_span, mv_kind, mut stream: TokenStream| {
        if stream.len() == 1 {
            let tree = stream.iter().next().unwrap();
            if let TokenTree::Delimited(_, _, delim, inner) = tree
                && let Delimiter::Invisible(InvisibleOrigin::MetaVar(mvk)) = delim
                && mv_kind == *mvk
            {
                stream = inner.clone();
            }
        }

        // Emit as a token stream within `Delimiter::Invisible` to maintain
        // parsing priorities.
        tscx.marker.mark_span(&mut sp);
        with_metavar_spans(|mspans| mspans.insert(mk_span, sp));
        // Both the open delim and close delim get the same span, which covers the
        // `$foo` in the decl macro RHS.
        TokenTree::Delimited(
            DelimSpan::from_single(sp),
            DelimSpacing::new(Spacing::Alone, Spacing::Alone),
            Delimiter::Invisible(InvisibleOrigin::MetaVar(mv_kind)),
            stream,
        )
    };

    let tt = match cur_matched {
        MatchedSingle(ParseNtResult::Tt(tt)) => {
            // `tt`s are emitted into the output stream directly as "raw tokens",
            // without wrapping them into groups. Other variables are emitted into
            // the output stream as groups with `Delimiter::Invisible` to maintain
            // parsing priorities.
            maybe_use_metavar_location(tscx.psess, &tscx.stack, sp, tt, &mut tscx.marker)
        }
        MatchedSingle(ParseNtResult::Ident(ident, is_raw)) => {
            tscx.marker.mark_span(&mut sp);
            with_metavar_spans(|mspans| mspans.insert(ident.span, sp));
            let kind = token::NtIdent(*ident, *is_raw);
            TokenTree::token_alone(kind, sp)
        }
        MatchedSingle(ParseNtResult::Lifetime(ident, is_raw)) => {
            tscx.marker.mark_span(&mut sp);
            with_metavar_spans(|mspans| mspans.insert(ident.span, sp));
            let kind = token::NtLifetime(*ident, *is_raw);
            TokenTree::token_alone(kind, sp)
        }
        MatchedSingle(ParseNtResult::Item(item)) => {
            mk_delimited(item.span, MetaVarKind::Item, TokenStream::from_ast(item))
        }
        MatchedSingle(ParseNtResult::Block(block)) => {
            mk_delimited(block.span, MetaVarKind::Block, TokenStream::from_ast(block))
        }
        MatchedSingle(ParseNtResult::Stmt(stmt)) => {
            let stream = if let StmtKind::Empty = stmt.kind {
                // FIXME: Properly collect tokens for empty statements.
                TokenStream::token_alone(token::Semi, stmt.span)
            } else {
                TokenStream::from_ast(stmt)
            };
            mk_delimited(stmt.span, MetaVarKind::Stmt, stream)
        }
        MatchedSingle(ParseNtResult::Pat(pat, pat_kind)) => {
            mk_delimited(pat.span, MetaVarKind::Pat(*pat_kind), TokenStream::from_ast(pat))
        }
        MatchedSingle(ParseNtResult::Expr(expr, kind)) => {
            let (can_begin_literal_maybe_minus, can_begin_string_literal) = match &expr.kind {
                ExprKind::Lit(_) => (true, true),
                ExprKind::Unary(UnOp::Neg, e) if matches!(&e.kind, ExprKind::Lit(_)) => {
                    (true, false)
                }
                _ => (false, false),
            };
            mk_delimited(
                expr.span,
                MetaVarKind::Expr {
                    kind: *kind,
                    can_begin_literal_maybe_minus,
                    can_begin_string_literal,
                },
                TokenStream::from_ast(expr),
            )
        }
        MatchedSingle(ParseNtResult::Literal(lit)) => {
            mk_delimited(lit.span, MetaVarKind::Literal, TokenStream::from_ast(lit))
        }
        MatchedSingle(ParseNtResult::Ty(ty)) => {
            let is_path = matches!(&ty.kind, TyKind::Path(None, _path));
            mk_delimited(ty.span, MetaVarKind::Ty { is_path }, TokenStream::from_ast(ty))
        }
        MatchedSingle(ParseNtResult::Meta(attr_item)) => {
            let has_meta_form = attr_item.meta_kind().is_some();
            mk_delimited(
                attr_item.span(),
                MetaVarKind::Meta { has_meta_form },
                TokenStream::from_ast(attr_item),
            )
        }
        MatchedSingle(ParseNtResult::Path(path)) => {
            mk_delimited(path.span, MetaVarKind::Path, TokenStream::from_ast(path))
        }
        MatchedSingle(ParseNtResult::Vis(vis)) => {
            mk_delimited(vis.span, MetaVarKind::Vis, TokenStream::from_ast(vis))
        }
        MatchedSeq(..) => {
            // We were unable to descend far enough. This is an error.
            return Err(dcx.create_err(VarStillRepeating { span: sp, ident }));
        }
    };

    tscx.result.push(tt);
    Ok(())
}

/// Turn `${expr(...)}` metavariable expressionss into tokens.
fn transcribe_metavar_expr<'tx>(
    tscx: &mut TranscrCtx<'tx, '_>,
    dspan: DelimSpan,
    expr: &MetaVarExpr,
) -> PResult<'tx, ()> {
    let dcx = tscx.psess.dcx();
    let tt = match *expr {
        MetaVarExpr::Concat(ref elements) => metavar_expr_concat(tscx, dspan, elements)?,
        MetaVarExpr::Count(original_ident, depth) => {
            let matched = matched_from_ident(dcx, original_ident, tscx.interp)?;
            let count = count_repetitions(dcx, depth, matched, &tscx.repeats, &dspan)?;
            TokenTree::token_alone(
                TokenKind::lit(token::Integer, sym::integer(count), None),
                tscx.visited_dspan(dspan),
            )
        }
        MetaVarExpr::Ignore(original_ident) => {
            // Used to ensure that `original_ident` is present in the LHS
            let _ = matched_from_ident(dcx, original_ident, tscx.interp)?;
            return Ok(());
        }
        MetaVarExpr::Index(depth) => match tscx.repeats.iter().nth_back(depth) {
            Some((index, _)) => TokenTree::token_alone(
                TokenKind::lit(token::Integer, sym::integer(*index), None),
                tscx.visited_dspan(dspan),
            ),
            None => {
                return Err(out_of_bounds_err(dcx, tscx.repeats.len(), dspan.entire(), "index"));
            }
        },
        MetaVarExpr::Len(depth) => match tscx.repeats.iter().nth_back(depth) {
            Some((_, length)) => TokenTree::token_alone(
                TokenKind::lit(token::Integer, sym::integer(*length), None),
                tscx.visited_dspan(dspan),
            ),
            None => {
                return Err(out_of_bounds_err(dcx, tscx.repeats.len(), dspan.entire(), "len"));
            }
        },
    };
    tscx.result.push(tt);
    Ok(())
}

/// Handle the `${concat(...)}` metavariable expression.
fn metavar_expr_concat<'tx>(
    tscx: &mut TranscrCtx<'tx, '_>,
    dspan: DelimSpan,
    elements: &[MetaVarExprConcatElem],
) -> PResult<'tx, TokenTree> {
    let dcx = tscx.psess.dcx();
    let mut concatenated = String::new();
    for element in elements.into_iter() {
        let symbol = match element {
            MetaVarExprConcatElem::Ident(elem) => elem.name,
            MetaVarExprConcatElem::Literal(elem) => *elem,
            MetaVarExprConcatElem::Var(ident) => {
                match matched_from_ident(dcx, *ident, tscx.interp)? {
                    NamedMatch::MatchedSeq(named_matches) => {
                        let Some((curr_idx, _)) = tscx.repeats.last() else {
                            return Err(dcx.struct_span_err(dspan.entire(), "invalid syntax"));
                        };
                        match &named_matches[*curr_idx] {
                            // FIXME(c410-f3r) Nested repetitions are unimplemented
                            MatchedSeq(_) => unimplemented!(),
                            MatchedSingle(pnr) => extract_symbol_from_pnr(dcx, pnr, ident.span)?,
                        }
                    }
                    NamedMatch::MatchedSingle(pnr) => {
                        extract_symbol_from_pnr(dcx, pnr, ident.span)?
                    }
                }
            }
        };
        concatenated.push_str(symbol.as_str());
    }
    let symbol = nfc_normalize(&concatenated);
    let concatenated_span = tscx.visited_dspan(dspan);
    if !rustc_lexer::is_ident(symbol.as_str()) {
        return Err(dcx.struct_span_err(
            concatenated_span,
            "`${concat(..)}` is not generating a valid identifier",
        ));
    }
    tscx.psess.symbol_gallery.insert(symbol, concatenated_span);

    // The current implementation marks the span as coming from the macro regardless of
    // contexts of the concatenated identifiers but this behavior may change in the
    // future.
    Ok(TokenTree::Token(
        Token::from_ast_ident(Ident::new(symbol, concatenated_span)),
        Spacing::Alone,
    ))
}

/// Store the metavariable span for this original span into a side table.
/// FIXME: Try to put the metavariable span into `SpanData` instead of a side table (#118517).
/// An optimal encoding for inlined spans will need to be selected to minimize regressions.
/// The side table approach is relatively good, but not perfect due to collisions.
/// In particular, collisions happen when token is passed as an argument through several macro
/// calls, like in recursive macros.
/// The old heuristic below is used to improve spans in case of collisions, but diagnostics are
/// still degraded sometimes in those cases.
///
/// The old heuristic:
///
/// Usually metavariables `$var` produce interpolated tokens, which have an additional place for
/// keeping both the original span and the metavariable span. For `tt` metavariables that's not the
/// case however, and there's no place for keeping a second span. So we try to give the single
/// produced span a location that would be most useful in practice (the hygiene part of the span
/// must not be changed).
///
/// Different locations are useful for different purposes:
/// - The original location is useful when we need to report a diagnostic for the original token in
///   isolation, without combining it with any surrounding tokens. This case occurs, but it is not
///   very common in practice.
/// - The metavariable location is useful when we need to somehow combine the token span with spans
///   of its surrounding tokens. This is the most common way to use token spans.
///
/// So this function replaces the original location with the metavariable location in all cases
/// except these two:
/// - The metavariable is an element of undelimited sequence `$($tt)*`.
///   These are typically used for passing larger amounts of code, and tokens in that code usually
///   combine with each other and not with tokens outside of the sequence.
/// - The metavariable span comes from a different crate, then we prefer the more local span.
fn maybe_use_metavar_location(
    psess: &ParseSess,
    stack: &[Frame<'_>],
    mut metavar_span: Span,
    orig_tt: &TokenTree,
    marker: &mut Marker,
) -> TokenTree {
    let undelimited_seq = matches!(
        stack.last(),
        Some(Frame {
            tts: [_],
            kind: FrameKind::Sequence {
                sep: None,
                kleene_op: KleeneOp::ZeroOrMore | KleeneOp::OneOrMore,
                ..
            },
            ..
        })
    );
    if undelimited_seq {
        // Do not record metavar spans for tokens from undelimited sequences, for perf reasons.
        return orig_tt.clone();
    }

    marker.mark_span(&mut metavar_span);
    let no_collision = match orig_tt {
        TokenTree::Token(token, ..) => {
            with_metavar_spans(|mspans| mspans.insert(token.span, metavar_span))
        }
        TokenTree::Delimited(dspan, ..) => with_metavar_spans(|mspans| {
            mspans.insert(dspan.open, metavar_span)
                && mspans.insert(dspan.close, metavar_span)
                && mspans.insert(dspan.entire(), metavar_span)
        }),
    };
    if no_collision || psess.source_map().is_imported(metavar_span) {
        return orig_tt.clone();
    }

    // Setting metavar spans for the heuristic spans gives better opportunities for combining them
    // with neighboring spans even despite their different syntactic contexts.
    match orig_tt {
        TokenTree::Token(Token { kind, span }, spacing) => {
            let span = metavar_span.with_ctxt(span.ctxt());
            with_metavar_spans(|mspans| mspans.insert(span, metavar_span));
            TokenTree::Token(Token { kind: kind.clone(), span }, *spacing)
        }
        TokenTree::Delimited(dspan, dspacing, delimiter, tts) => {
            let open = metavar_span.with_ctxt(dspan.open.ctxt());
            let close = metavar_span.with_ctxt(dspan.close.ctxt());
            with_metavar_spans(|mspans| {
                mspans.insert(open, metavar_span) && mspans.insert(close, metavar_span)
            });
            let dspan = DelimSpan::from_pair(open, close);
            TokenTree::Delimited(dspan, *dspacing, *delimiter, tts.clone())
        }
    }
}

/// Lookup the meta-var named `ident` and return the matched token tree from the invocation using
/// the set of matches `interpolations`.
///
/// See the definition of `repeats` in the `transcribe` function. `repeats` is used to descend
/// into the right place in nested matchers. If we attempt to descend too far, the macro writer has
/// made a mistake, and we return `None`.
fn lookup_cur_matched<'a>(
    ident: MacroRulesNormalizedIdent,
    interpolations: &'a FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    repeats: &[(usize, usize)],
) -> Option<&'a NamedMatch> {
    interpolations.get(&ident).map(|mut matched| {
        for &(idx, _) in repeats {
            match matched {
                MatchedSingle(_) => break,
                MatchedSeq(ads) => matched = ads.get(idx).unwrap(),
            }
        }

        matched
    })
}

/// An accumulator over a TokenTree to be used with `fold`. During transcription, we need to make
/// sure that the size of each sequence and all of its nested sequences are the same as the sizes
/// of all the matched (nested) sequences in the macro invocation. If they don't match, somebody
/// has made a mistake (either the macro writer or caller).
#[derive(Clone)]
enum LockstepIterSize {
    /// No constraints on length of matcher. This is true for any TokenTree variants except a
    /// `MetaVar` with an actual `MatchedSeq` (as opposed to a `MatchedNonterminal`).
    Unconstrained,

    /// A `MetaVar` with an actual `MatchedSeq`. The length of the match and the name of the
    /// meta-var are returned.
    Constraint(usize, MacroRulesNormalizedIdent),

    /// Two `Constraint`s on the same sequence had different lengths. This is an error.
    Contradiction(String),
}

impl LockstepIterSize {
    /// Find incompatibilities in matcher/invocation sizes.
    /// - `Unconstrained` is compatible with everything.
    /// - `Contradiction` is incompatible with everything.
    /// - `Constraint(len)` is only compatible with other constraints of the same length.
    fn with(self, other: LockstepIterSize) -> LockstepIterSize {
        match self {
            LockstepIterSize::Unconstrained => other,
            LockstepIterSize::Contradiction(_) => self,
            LockstepIterSize::Constraint(l_len, l_id) => match other {
                LockstepIterSize::Unconstrained => self,
                LockstepIterSize::Contradiction(_) => other,
                LockstepIterSize::Constraint(r_len, _) if l_len == r_len => self,
                LockstepIterSize::Constraint(r_len, r_id) => {
                    let msg = format!(
                        "meta-variable `{}` repeats {} time{}, but `{}` repeats {} time{}",
                        l_id,
                        l_len,
                        pluralize!(l_len),
                        r_id,
                        r_len,
                        pluralize!(r_len),
                    );
                    LockstepIterSize::Contradiction(msg)
                }
            },
        }
    }
}

/// Given a `tree`, make sure that all sequences have the same length as the matches for the
/// appropriate meta-vars in `interpolations`.
///
/// Note that if `repeats` does not match the exact correct depth of a meta-var,
/// `lookup_cur_matched` will return `None`, which is why this still works even in the presence of
/// multiple nested matcher sequences.
///
/// Example: `$($($x $y)+*);+` -- we need to make sure that `x` and `y` repeat the same amount as
/// each other at the given depth when the macro was invoked. If they don't it might mean they were
/// declared at depths which weren't equal or there was a compiler bug. For example, if we have 3 repetitions of
/// the outer sequence and 4 repetitions of the inner sequence for `x`, we should have the same for
/// `y`; otherwise, we can't transcribe them both at the given depth.
fn lockstep_iter_size(
    tree: &mbe::TokenTree,
    interpolations: &FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    repeats: &[(usize, usize)],
) -> LockstepIterSize {
    use mbe::TokenTree;
    match tree {
        TokenTree::Delimited(.., delimited) => {
            delimited.tts.iter().fold(LockstepIterSize::Unconstrained, |size, tt| {
                size.with(lockstep_iter_size(tt, interpolations, repeats))
            })
        }
        TokenTree::Sequence(_, seq) => {
            seq.tts.iter().fold(LockstepIterSize::Unconstrained, |size, tt| {
                size.with(lockstep_iter_size(tt, interpolations, repeats))
            })
        }
        TokenTree::MetaVar(_, name) | TokenTree::MetaVarDecl(_, name, _) => {
            let name = MacroRulesNormalizedIdent::new(*name);
            match lookup_cur_matched(name, interpolations, repeats) {
                Some(matched) => match matched {
                    MatchedSingle(_) => LockstepIterSize::Unconstrained,
                    MatchedSeq(ads) => LockstepIterSize::Constraint(ads.len(), name),
                },
                _ => LockstepIterSize::Unconstrained,
            }
        }
        TokenTree::MetaVarExpr(_, expr) => {
            expr.for_each_metavar(LockstepIterSize::Unconstrained, |lis, ident| {
                lis.with(lockstep_iter_size(
                    &TokenTree::MetaVar(ident.span, *ident),
                    interpolations,
                    repeats,
                ))
            })
        }
        TokenTree::Token(..) => LockstepIterSize::Unconstrained,
    }
}

/// Used solely by the `count` meta-variable expression, counts the outermost repetitions at a
/// given optional nested depth.
///
/// For example, a macro parameter of `$( { $( $foo:ident ),* } )*` called with `{ a, b } { c }`:
///
/// * `[ $( ${count(foo)} ),* ]` will return [2, 1] with a, b = 2 and c = 1
/// * `[ $( ${count(foo, 0)} ),* ]` will be the same as `[ $( ${count(foo)} ),* ]`
/// * `[ $( ${count(foo, 1)} ),* ]` will return an error because `${count(foo, 1)}` is
///   declared inside a single repetition and the index `1` implies two nested repetitions.
fn count_repetitions<'dx>(
    dcx: DiagCtxtHandle<'dx>,
    depth_user: usize,
    mut matched: &NamedMatch,
    repeats: &[(usize, usize)],
    sp: &DelimSpan,
) -> PResult<'dx, usize> {
    // Recursively count the number of matches in `matched` at given depth
    // (or at the top-level of `matched` if no depth is given).
    fn count<'a>(depth_curr: usize, depth_max: usize, matched: &NamedMatch) -> PResult<'a, usize> {
        match matched {
            MatchedSingle(_) => Ok(1),
            MatchedSeq(named_matches) => {
                if depth_curr == depth_max {
                    Ok(named_matches.len())
                } else {
                    named_matches.iter().map(|elem| count(depth_curr + 1, depth_max, elem)).sum()
                }
            }
        }
    }

    /// Maximum depth
    fn depth(counter: usize, matched: &NamedMatch) -> usize {
        match matched {
            MatchedSingle(_) => counter,
            MatchedSeq(named_matches) => {
                let rslt = counter + 1;
                if let Some(elem) = named_matches.first() { depth(rslt, elem) } else { rslt }
            }
        }
    }

    let depth_max = depth(0, matched)
        .checked_sub(1)
        .and_then(|el| el.checked_sub(repeats.len()))
        .unwrap_or_default();
    if depth_user > depth_max {
        return Err(out_of_bounds_err(dcx, depth_max + 1, sp.entire(), "count"));
    }

    // `repeats` records all of the nested levels at which we are currently
    // matching meta-variables. The meta-var-expr `count($x)` only counts
    // matches that occur in this "subtree" of the `NamedMatch` where we
    // are currently transcribing, so we need to descend to that subtree
    // before we start counting. `matched` contains the various levels of the
    // tree as we descend, and its final value is the subtree we are currently at.
    for &(idx, _) in repeats {
        if let MatchedSeq(ads) = matched {
            matched = &ads[idx];
        }
    }

    if let MatchedSingle(_) = matched {
        return Err(dcx.create_err(CountRepetitionMisplaced { span: sp.entire() }));
    }

    count(depth_user, depth_max, matched)
}

/// Returns a `NamedMatch` item declared on the LHS given an arbitrary [Ident]
fn matched_from_ident<'ctx, 'interp, 'rslt>(
    dcx: DiagCtxtHandle<'ctx>,
    ident: Ident,
    interp: &'interp FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
) -> PResult<'ctx, &'rslt NamedMatch>
where
    'interp: 'rslt,
{
    let span = ident.span;
    let key = MacroRulesNormalizedIdent::new(ident);
    interp.get(&key).ok_or_else(|| dcx.create_err(MetaVarExprUnrecognizedVar { span, key }))
}

/// Used by meta-variable expressions when an user input is out of the actual declared bounds. For
/// example, index(999999) in an repetition of only three elements.
fn out_of_bounds_err<'a>(dcx: DiagCtxtHandle<'a>, max: usize, span: Span, ty: &str) -> Diag<'a> {
    let msg = if max == 0 {
        format!(
            "meta-variable expression `{ty}` with depth parameter \
             must be called inside of a macro repetition"
        )
    } else {
        format!(
            "depth parameter of meta-variable expression `{ty}` \
             must be less than {max}"
        )
    };
    dcx.struct_span_err(span, msg)
}

/// Extracts an metavariable symbol that can be an identifier, a token tree or a literal.
fn extract_symbol_from_pnr<'a>(
    dcx: DiagCtxtHandle<'a>,
    pnr: &ParseNtResult,
    span_err: Span,
) -> PResult<'a, Symbol> {
    match pnr {
        ParseNtResult::Ident(nt_ident, is_raw) => {
            if let IdentIsRaw::Yes = is_raw {
                Err(dcx.struct_span_err(span_err, RAW_IDENT_ERR))
            } else {
                Ok(nt_ident.name)
            }
        }
        ParseNtResult::Tt(TokenTree::Token(
            Token { kind: TokenKind::Ident(symbol, is_raw), .. },
            _,
        )) => {
            if let IdentIsRaw::Yes = is_raw {
                Err(dcx.struct_span_err(span_err, RAW_IDENT_ERR))
            } else {
                Ok(*symbol)
            }
        }
        ParseNtResult::Tt(TokenTree::Token(
            Token {
                kind: TokenKind::Literal(Lit { kind: LitKind::Str, symbol, suffix: None }),
                ..
            },
            _,
        )) => Ok(*symbol),
        ParseNtResult::Literal(expr)
            if let ExprKind::Lit(Lit { kind: LitKind::Str, symbol, suffix: None }) = &expr.kind =>
        {
            Ok(*symbol)
        }
        _ => Err(dcx
            .struct_err(
                "metavariables of `${concat(..)}` must be of type `ident`, `literal` or `tt`",
            )
            .with_note("currently only string literals are supported")
            .with_span(span_err)),
    }
}
