use crate::base::ExtCtxt;
use crate::mbe;
use crate::mbe::macro_parser::{MatchedNonterminal, MatchedSeq, NamedMatch};

use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::token::{self, NtTT, Token};
use rustc_ast::tokenstream::{DelimSpan, TokenStream, TokenTree, TreeAndSpacing};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{pluralize, PResult};
use rustc_span::hygiene::{LocalExpnId, Transparency};
use rustc_span::symbol::MacroRulesNormalizedIdent;
use rustc_span::Span;

use smallvec::{smallvec, SmallVec};
use std::mem;

// A Marker adds the given mark to the syntax context.
struct Marker(LocalExpnId, Transparency);

impl MutVisitor for Marker {
    const VISIT_TOKENS: bool = true;

    fn visit_span(&mut self, span: &mut Span) {
        *span = span.apply_mark(self.0.to_expn_id(), self.1)
    }
}

/// An iterator over the token trees in a delimited token tree (`{ ... }`) or a sequence (`$(...)`).
enum Frame {
    Delimited { forest: Lrc<mbe::Delimited>, idx: usize, span: DelimSpan },
    Sequence { forest: Lrc<mbe::SequenceRepetition>, idx: usize, sep: Option<Token> },
}

impl Frame {
    /// Construct a new frame around the delimited set of tokens.
    fn new(tts: Vec<mbe::TokenTree>) -> Frame {
        let forest = Lrc::new(mbe::Delimited { delim: token::NoDelim, tts });
        Frame::Delimited { forest, idx: 0, span: DelimSpan::dummy() }
    }
}

impl Iterator for Frame {
    type Item = mbe::TokenTree;

    fn next(&mut self) -> Option<mbe::TokenTree> {
        match *self {
            Frame::Delimited { ref forest, ref mut idx, .. } => {
                *idx += 1;
                forest.tts.get(*idx - 1).cloned()
            }
            Frame::Sequence { ref forest, ref mut idx, .. } => {
                *idx += 1;
                forest.tts.get(*idx - 1).cloned()
            }
        }
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
    cx: &ExtCtxt<'a>,
    interp: &FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    src: Vec<mbe::TokenTree>,
    transparency: Transparency,
) -> PResult<'a, TokenStream> {
    // Nothing for us to transcribe...
    if src.is_empty() {
        return Ok(TokenStream::default());
    }

    // We descend into the RHS (`src`), expanding things as we go. This stack contains the things
    // we have yet to expand/are still expanding. We start the stack off with the whole RHS.
    let mut stack: SmallVec<[Frame; 1]> = smallvec![Frame::new(src)];

    // As we descend in the RHS, we will need to be able to match nested sequences of matchers.
    // `repeats` keeps track of where we are in matching at each level, with the last element being
    // the most deeply nested sequence. This is used as a stack.
    let mut repeats = Vec::new();

    // `result` contains resulting token stream from the TokenTree we just finished processing. At
    // the end, this will contain the full result of transcription, but at arbitrary points during
    // `transcribe`, `result` will contain subsets of the final result.
    //
    // Specifically, as we descend into each TokenTree, we will push the existing results onto the
    // `result_stack` and clear `results`. We will then produce the results of transcribing the
    // TokenTree into `results`. Then, as we unwind back out of the `TokenTree`, we will pop the
    // `result_stack` and append `results` too it to produce the new `results` up to that point.
    //
    // Thus, if we try to pop the `result_stack` and it is empty, we have reached the top-level
    // again, and we are done transcribing.
    let mut result: Vec<TreeAndSpacing> = Vec::new();
    let mut result_stack = Vec::new();
    let mut marker = Marker(cx.current_expansion.id, transparency);

    loop {
        // Look at the last frame on the stack.
        // If it still has a TokenTree we have not looked at yet, use that tree.
        let Some(tree) = stack.last_mut().unwrap().next() else {
            // This else-case never produces a value for `tree` (it `continue`s or `return`s).

            // Otherwise, if we have just reached the end of a sequence and we can keep repeating,
            // go back to the beginning of the sequence.
            if let Frame::Sequence { idx, sep, .. } = stack.last_mut().unwrap() {
                let (repeat_idx, repeat_len) = repeats.last_mut().unwrap();
                *repeat_idx += 1;
                if repeat_idx < repeat_len {
                    *idx = 0;
                    if let Some(sep) = sep {
                        result.push(TokenTree::Token(sep.clone()).into());
                    }
                    continue;
                }
            }

            // We are done with the top of the stack. Pop it. Depending on what it was, we do
            // different things. Note that the outermost item must be the delimited, wrapped RHS
            // that was passed in originally to `transcribe`.
            match stack.pop().unwrap() {
                // Done with a sequence. Pop from repeats.
                Frame::Sequence { .. } => {
                    repeats.pop();
                }

                // We are done processing a Delimited. If this is the top-level delimited, we are
                // done. Otherwise, we unwind the result_stack to append what we have produced to
                // any previous results.
                Frame::Delimited { forest, span, .. } => {
                    if result_stack.is_empty() {
                        // No results left to compute! We are back at the top-level.
                        return Ok(TokenStream::new(result));
                    }

                    // Step back into the parent Delimited.
                    let tree = TokenTree::Delimited(span, forest.delim, TokenStream::new(result));
                    result = result_stack.pop().unwrap();
                    result.push(tree.into());
                }
            }
            continue;
        };

        // At this point, we know we are in the middle of a TokenTree (the last one on `stack`).
        // `tree` contains the next `TokenTree` to be processed.
        match tree {
            // We are descending into a sequence. We first make sure that the matchers in the RHS
            // and the matches in `interp` have the same shape. Otherwise, either the caller or the
            // macro writer has made a mistake.
            seq @ mbe::TokenTree::Sequence(..) => {
                match lockstep_iter_size(&seq, interp, &repeats) {
                    LockstepIterSize::Unconstrained => {
                        return Err(cx.struct_span_err(
                            seq.span(), /* blame macro writer */
                            "attempted to repeat an expression containing no syntax variables \
                             matched as repeating at this depth",
                        ));
                    }

                    LockstepIterSize::Contradiction(msg) => {
                        // FIXME: this really ought to be caught at macro definition time... It
                        // happens when two meta-variables are used in the same repetition in a
                        // sequence, but they come from different sequence matchers and repeat
                        // different amounts.
                        return Err(cx.struct_span_err(seq.span(), &msg));
                    }

                    LockstepIterSize::Constraint(len, _) => {
                        // We do this to avoid an extra clone above. We know that this is a
                        // sequence already.
                        let mbe::TokenTree::Sequence(sp, seq) = seq else {
                            unreachable!()
                        };

                        // Is the repetition empty?
                        if len == 0 {
                            if seq.kleene.op == mbe::KleeneOp::OneOrMore {
                                // FIXME: this really ought to be caught at macro definition
                                // time... It happens when the Kleene operator in the matcher and
                                // the body for the same meta-variable do not match.
                                return Err(cx.struct_span_err(
                                    sp.entire(),
                                    "this must repeat at least once",
                                ));
                            }
                        } else {
                            // 0 is the initial counter (we have done 0 repretitions so far). `len`
                            // is the total number of repetitions we should generate.
                            repeats.push((0, len));

                            // The first time we encounter the sequence we push it to the stack. It
                            // then gets reused (see the beginning of the loop) until we are done
                            // repeating.
                            stack.push(Frame::Sequence {
                                idx: 0,
                                sep: seq.separator.clone(),
                                forest: seq,
                            });
                        }
                    }
                }
            }

            // Replace the meta-var with the matched token tree from the invocation.
            mbe::TokenTree::MetaVar(mut sp, mut orignal_ident) => {
                // Find the matched nonterminal from the macro invocation, and use it to replace
                // the meta-var.
                let ident = MacroRulesNormalizedIdent::new(orignal_ident);
                if let Some(cur_matched) = lookup_cur_matched(ident, interp, &repeats) {
                    if let MatchedNonterminal(nt) = cur_matched {
                        let token = if let NtTT(tt) = &**nt {
                            // `tt`s are emitted into the output stream directly as "raw tokens",
                            // without wrapping them into groups.
                            tt.clone()
                        } else {
                            // Other variables are emitted into the output stream as groups with
                            // `Delimiter::None` to maintain parsing priorities.
                            // `Interpolated` is currenty used for such groups in rustc parser.
                            marker.visit_span(&mut sp);
                            TokenTree::token(token::Interpolated(nt.clone()), sp)
                        };
                        result.push(token.into());
                    } else {
                        // We were unable to descend far enough. This is an error.
                        return Err(cx.struct_span_err(
                            sp, /* blame the macro writer */
                            &format!("variable '{}' is still repeating at this depth", ident),
                        ));
                    }
                } else {
                    // If we aren't able to match the meta-var, we push it back into the result but
                    // with modified syntax context. (I believe this supports nested macros).
                    marker.visit_span(&mut sp);
                    marker.visit_ident(&mut orignal_ident);
                    result.push(TokenTree::token(token::Dollar, sp).into());
                    result.push(TokenTree::Token(Token::from_ast_ident(orignal_ident)).into());
                }
            }

            // If we are entering a new delimiter, we push its contents to the `stack` to be
            // processed, and we push all of the currently produced results to the `result_stack`.
            // We will produce all of the results of the inside of the `Delimited` and then we will
            // jump back out of the Delimited, pop the result_stack and add the new results back to
            // the previous results (from outside the Delimited).
            mbe::TokenTree::Delimited(mut span, delimited) => {
                mut_visit::visit_delim_span(&mut span, &mut marker);
                stack.push(Frame::Delimited { forest: delimited, idx: 0, span });
                result_stack.push(mem::take(&mut result));
            }

            // Nothing much to do here. Just push the token to the result, being careful to
            // preserve syntax context.
            mbe::TokenTree::Token(token) => {
                let mut tt = TokenTree::Token(token);
                mut_visit::visit_tt(&mut tt, &mut marker);
                result.push(tt.into());
            }

            // There should be no meta-var declarations in the invocation of a macro.
            mbe::TokenTree::MetaVarDecl(..) => panic!("unexpected `TokenTree::MetaVarDecl"),
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
    interpolations.get(&ident).map(|matched| {
        let mut matched = matched;
        for &(idx, _) in repeats {
            match matched {
                MatchedNonterminal(_) => break,
                MatchedSeq(ref ads) => matched = ads.get(idx).unwrap(),
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
            LockstepIterSize::Constraint(l_len, ref l_id) => match other {
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
fn lockstep_iter_size(
    tree: &mbe::TokenTree,
    interpolations: &FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    repeats: &[(usize, usize)],
) -> LockstepIterSize {
    use mbe::TokenTree;
    match *tree {
        TokenTree::Delimited(_, ref delimed) => {
            delimed.tts.iter().fold(LockstepIterSize::Unconstrained, |size, tt| {
                size.with(lockstep_iter_size(tt, interpolations, repeats))
            })
        }
        TokenTree::Sequence(_, ref seq) => {
            seq.tts.iter().fold(LockstepIterSize::Unconstrained, |size, tt| {
                size.with(lockstep_iter_size(tt, interpolations, repeats))
            })
        }
        TokenTree::MetaVar(_, name) | TokenTree::MetaVarDecl(_, name, _) => {
            let name = MacroRulesNormalizedIdent::new(name);
            match lookup_cur_matched(name, interpolations, repeats) {
                Some(matched) => match matched {
                    MatchedNonterminal(_) => LockstepIterSize::Unconstrained,
                    MatchedSeq(ref ads) => LockstepIterSize::Constraint(ads.len(), name),
                },
                _ => LockstepIterSize::Unconstrained,
            }
        }
        TokenTree::Token(..) => LockstepIterSize::Unconstrained,
    }
}
