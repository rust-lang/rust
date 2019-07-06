use crate::ast::Ident;
use crate::ext::base::ExtCtxt;
use crate::ext::expand::Marker;
use crate::ext::tt::macro_parser::{MatchedNonterminal, MatchedSeq, NamedMatch};
use crate::ext::tt::quoted;
use crate::mut_visit::noop_visit_tt;
use crate::parse::token::{self, NtTT, Token};
use crate::tokenstream::{DelimSpan, TokenStream, TokenTree, TreeAndJoint};

use smallvec::{smallvec, SmallVec};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use std::mem;
use std::rc::Rc;

/// An iterator over the token trees in a delimited token tree (`{ ... }`) or a sequence (`$(...)`).
enum Frame {
    Delimited { forest: Lrc<quoted::Delimited>, idx: usize, span: DelimSpan },
    Sequence { forest: Lrc<quoted::SequenceRepetition>, idx: usize, sep: Option<Token> },
}

impl Frame {
    /// Construct a new frame around the delimited set of tokens.
    fn new(tts: Vec<quoted::TokenTree>) -> Frame {
        let forest = Lrc::new(quoted::Delimited { delim: token::NoDelim, tts });
        Frame::Delimited { forest, idx: 0, span: DelimSpan::dummy() }
    }
}

impl Iterator for Frame {
    type Item = quoted::TokenTree;

    fn next(&mut self) -> Option<quoted::TokenTree> {
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
pub fn transcribe(
    cx: &ExtCtxt<'_>,
    interp: &FxHashMap<Ident, Rc<NamedMatch>>,
    src: Vec<quoted::TokenTree>,
) -> TokenStream {
    // Nothing for us to transcribe...
    if src.is_empty() {
        return TokenStream::empty();
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
    let mut result: Vec<TreeAndJoint> = Vec::new();
    let mut result_stack = Vec::new();

    loop {
        // Look at the last frame on the stack.
        let tree = if let Some(tree) = stack.last_mut().unwrap().next() {
            // If it still has a TokenTree we have not looked at yet, use that tree.
            tree
        }
        // The else-case never produces a value for `tree` (it `continue`s or `return`s).
        else {
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
                        return TokenStream::new(result);
                    }

                    // Step back into the parent Delimited.
                    let tree =
                        TokenTree::Delimited(span, forest.delim, TokenStream::new(result).into());
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
            seq @ quoted::TokenTree::Sequence(..) => {
                match lockstep_iter_size(&seq, interp, &repeats) {
                    LockstepIterSize::Unconstrained => {
                        cx.span_fatal(
                            seq.span(), /* blame macro writer */
                            "attempted to repeat an expression containing no syntax variables \
                             matched as repeating at this depth",
                        );
                    }

                    LockstepIterSize::Contradiction(ref msg) => {
                        // FIXME: this really ought to be caught at macro definition time... It
                        // happens when two meta-variables are used in the same repetition in a
                        // sequence, but they come from different sequence matchers and repeat
                        // different amounts.
                        cx.span_fatal(seq.span(), &msg[..]);
                    }

                    LockstepIterSize::Constraint(len, _) => {
                        // We do this to avoid an extra clone above. We know that this is a
                        // sequence already.
                        let (sp, seq) = if let quoted::TokenTree::Sequence(sp, seq) = seq {
                            (sp, seq)
                        } else {
                            unreachable!()
                        };

                        // Is the repetition empty?
                        if len == 0 {
                            if seq.op == quoted::KleeneOp::OneOrMore {
                                // FIXME: this really ought to be caught at macro definition
                                // time... It happens when the Kleene operator in the matcher and
                                // the body for the same meta-variable do not match.
                                cx.span_fatal(sp.entire(), "this must repeat at least once");
                            }
                        } else {
                            // 0 is the initial counter (we have done 0 repretitions so far). `len`
                            // is the total number of reptitions we should generate.
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
            quoted::TokenTree::MetaVar(mut sp, ident) => {
                // Find the matched nonterminal from the macro invocation, and use it to replace
                // the meta-var.
                if let Some(cur_matched) = lookup_cur_matched(ident, interp, &repeats) {
                    if let MatchedNonterminal(ref nt) = *cur_matched {
                        // FIXME #2887: why do we apply a mark when matching a token tree meta-var
                        // (e.g. `$x:tt`), but not when we are matching any other type of token
                        // tree?
                        if let NtTT(ref tt) = **nt {
                            result.push(tt.clone().into());
                        } else {
                            sp = sp.apply_mark(cx.current_expansion.mark);
                            let token = TokenTree::token(token::Interpolated(nt.clone()), sp);
                            result.push(token.into());
                        }
                    } else {
                        // We were unable to descend far enough. This is an error.
                        cx.span_fatal(
                            sp, /* blame the macro writer */
                            &format!("variable '{}' is still repeating at this depth", ident),
                        );
                    }
                } else {
                    // If we aren't able to match the meta-var, we push it back into the result but
                    // with modified syntax context. (I believe this supports nested macros).
                    let ident =
                        Ident::new(ident.name, ident.span.apply_mark(cx.current_expansion.mark));
                    sp = sp.apply_mark(cx.current_expansion.mark);
                    result.push(TokenTree::token(token::Dollar, sp).into());
                    result.push(TokenTree::Token(Token::from_ast_ident(ident)).into());
                }
            }

            // If we are entering a new delimiter, we push its contents to the `stack` to be
            // processed, and we push all of the currently produced results to the `result_stack`.
            // We will produce all of the results of the inside of the `Delimited` and then we will
            // jump back out of the Delimited, pop the result_stack and add the new results back to
            // the previous results (from outside the Delimited).
            quoted::TokenTree::Delimited(mut span, delimited) => {
                span = span.apply_mark(cx.current_expansion.mark);
                stack.push(Frame::Delimited { forest: delimited, idx: 0, span });
                result_stack.push(mem::take(&mut result));
            }

            // Nothing much to do here. Just push the token to the result, being careful to
            // preserve syntax context.
            quoted::TokenTree::Token(token) => {
                let mut marker = Marker(cx.current_expansion.mark);
                let mut tt = TokenTree::Token(token);
                noop_visit_tt(&mut tt, &mut marker);
                result.push(tt.into());
            }

            // There should be no meta-var declarations in the invocation of a macro.
            quoted::TokenTree::MetaVarDecl(..) => panic!("unexpected `TokenTree::MetaVarDecl"),
        }
    }
}

/// Lookup the meta-var named `ident` and return the matched token tree from the invocation using
/// the set of matches `interpolations`.
///
/// See the definition of `repeats` in the `transcribe` function. `repeats` is used to descend
/// into the right place in nested matchers. If we attempt to descend too far, the macro writer has
/// made a mistake, and we return `None`.
fn lookup_cur_matched(
    ident: Ident,
    interpolations: &FxHashMap<Ident, Rc<NamedMatch>>,
    repeats: &[(usize, usize)],
) -> Option<Rc<NamedMatch>> {
    interpolations.get(&ident).map(|matched| {
        let mut matched = matched.clone();
        for &(idx, _) in repeats {
            let m = matched.clone();
            match *m {
                MatchedNonterminal(_) => break,
                MatchedSeq(ref ads, _) => matched = Rc::new(ads[idx].clone()),
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
    Constraint(usize, Ident),

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
                        "meta-variable `{}` repeats {} times, but `{}` repeats {} times",
                        l_id, l_len, r_id, r_len
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
/// `lookup_cur_matched` will return `None`, which is why this still works even in the presnece of
/// multiple nested matcher sequences.
fn lockstep_iter_size(
    tree: &quoted::TokenTree,
    interpolations: &FxHashMap<Ident, Rc<NamedMatch>>,
    repeats: &[(usize, usize)],
) -> LockstepIterSize {
    use quoted::TokenTree;
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
            match lookup_cur_matched(name, interpolations, repeats) {
                Some(matched) => match *matched {
                    MatchedNonterminal(_) => LockstepIterSize::Unconstrained,
                    MatchedSeq(ref ads, _) => LockstepIterSize::Constraint(ads.len(), name),
                },
                _ => LockstepIterSize::Unconstrained,
            }
        }
        TokenTree::Token(..) => LockstepIterSize::Unconstrained,
    }
}
