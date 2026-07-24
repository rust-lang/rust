//! This is an NFA-based parser, which calls out to the main Rust parser for named non-terminals
//! (which it commits to fully when it hits one in a grammar). There's a set of current NFA threads
//! and a set of next ones. Instead of NTs, we have a special case for Kleene star. The big-O, in
//! pathological cases, is worse than traditional use of NFA or Earley parsing, but it's an easier
//! fit for Macro-by-Example-style rules.
//!
//! (In order to prevent the pathological case, we'd need to lazily construct the resulting
//! `NamedMatch`es at the very end. It'd be a pain, and require more memory to keep around old
//! matcher positions, but it would also save overhead)
//!
//! We don't say this parser uses the Earley algorithm, because it's unnecessarily inaccurate.
//! The macro parser restricts itself to the features of finite state automata. Earley parsers
//! can be described as an extension of NFAs with completion rules, prediction rules, and recursion.
//!
//! Quick intro to how the parser works:
//!
//! A "matcher position" (a.k.a. "position" or "mp") is a dot in the middle of a matcher, usually
//! written as a `·`. For example `· a $( a )* a b` is one, as is `a $( · a )* a b`.
//!
//! The parser walks through the input a token at a time, maintaining a list
//! of threads consistent with the current position in the input string: `cur_mps`.
//!
//! As it processes them, it fills up `eof_mps` with threads that would be valid if
//! the macro invocation is now over, `bb_mps` with threads that are waiting on
//! a Rust non-terminal like `$e:expr`, and `next_mps` with threads that are waiting
//! on a particular token. Most of the logic concerns moving the · through the
//! repetitions indicated by Kleene stars. The rules for moving the · without
//! consuming any input are called epsilon transitions. It only advances or calls
//! out to the real Rust parser when no `cur_mps` threads remain.
//!
//! Example:
//!
//! ```text, ignore
//! Start parsing a a a a b against [· a $( a )* a b].
//!
//! Remaining input: a a a a b
//! next: [· a $( a )* a b]
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a a b
//! cur: [a · $( a )* a b]
//! Descend/Skip (first position).
//! next: [a $( · a )* a b]  [a $( a )* · a b].
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first position)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first position)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first position)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over a b. - - -
//!
//! Remaining input: ''
//! eof: [a $( a )* a b ·]
//! ```

use std::borrow::Cow;
use std::fmt::Display;
use std::rc::Rc;

pub(crate) use NamedMatch::*;
pub(crate) use ParseResult::*;
use rustc_ast::token::{self, DocComment, NonterminalKind, Token, TokenKind};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Diag, ErrorGuaranteed};
use rustc_middle::span_bug;
use rustc_parse::parser::{ParseNtResult, Parser, token_descr};
use rustc_span::{Ident, MacroRulesNormalizedIdent, Span};

use crate::mbe::macro_rules::Tracker;
use crate::mbe::{KleeneOp, TokenTree};

/// A unit within a matcher that a `MatcherPos` can refer to. Similar to (and derived from)
/// `mbe::TokenTree`, but designed specifically for fast and easy traversal during matching.
/// Notable differences to `mbe::TokenTree`:
/// - It is non-recursive, i.e. there is no nesting.
/// - The end pieces of each sequence (the separator, if present, and the Kleene op) are
///   represented explicitly, as is the very end of the matcher.
///
/// This means a matcher can be represented by `&[MatcherLoc]`, and traversal mostly involves
/// simply incrementing the current matcher position index by one.
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum MatcherLoc {
    Token {
        token: Token,
    },
    Delimited,
    Sequence {
        op: KleeneOp,
        num_metavar_decls: usize,
        idx_first_after: usize,
        next_metavar: usize,
        seq_depth: usize,
    },
    SequenceKleeneOpNoSep {
        op: KleeneOp,
        idx_first: usize,
    },
    SequenceSep {
        separator: Token,
    },
    SequenceKleeneOpAfterSep {
        idx_first: usize,
    },
    MetaVarDecl {
        span: Span,
        bind: Ident,
        kind: NonterminalKind,
        next_metavar: usize,
        seq_depth: usize,
    },
    Eof,
}

impl MatcherLoc {
    pub(super) fn span(&self) -> Option<Span> {
        match self {
            MatcherLoc::Token { token } => Some(token.span),
            MatcherLoc::Delimited => None,
            MatcherLoc::Sequence { .. } => None,
            MatcherLoc::SequenceKleeneOpNoSep { .. } => None,
            MatcherLoc::SequenceSep { .. } => None,
            MatcherLoc::SequenceKleeneOpAfterSep { .. } => None,
            MatcherLoc::MetaVarDecl { span, .. } => Some(*span),
            MatcherLoc::Eof => None,
        }
    }
}

impl Display for MatcherLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatcherLoc::Token { token } | MatcherLoc::SequenceSep { separator: token } => {
                write!(f, "{}", token_descr(token))
            }
            MatcherLoc::MetaVarDecl { bind, kind, .. } => {
                write!(f, "meta-variable `${bind}:{kind}`")
            }
            MatcherLoc::Eof => f.write_str("end of macro"),

            // FIXME: A prior comment noted that the following variants should not be printed in
            // diagnostics. "while trying to match sequence end" appears in several stderrs in the
            // ui tests. Other variants might be reachable too.
            MatcherLoc::Delimited => f.write_str("delimiter"),
            MatcherLoc::Sequence { .. } => f.write_str("sequence start"),
            MatcherLoc::SequenceKleeneOpNoSep { .. } => f.write_str("sequence end"),
            MatcherLoc::SequenceKleeneOpAfterSep { .. } => f.write_str("sequence end"),
        }
    }
}

pub(super) fn compute_locs(matcher: &[TokenTree]) -> Vec<MatcherLoc> {
    fn inner(
        tts: &[TokenTree],
        locs: &mut Vec<MatcherLoc>,
        next_metavar: &mut usize,
        seq_depth: usize,
    ) {
        for tt in tts {
            match tt {
                TokenTree::Token(token) => {
                    locs.push(MatcherLoc::Token { token: *token });
                }
                TokenTree::Delimited(span, _, delimited) => {
                    let open_token = Token::new(delimited.delim.as_open_token_kind(), span.open);
                    let close_token = Token::new(delimited.delim.as_close_token_kind(), span.close);

                    locs.push(MatcherLoc::Delimited);
                    locs.push(MatcherLoc::Token { token: open_token });
                    inner(&delimited.tts, locs, next_metavar, seq_depth);
                    locs.push(MatcherLoc::Token { token: close_token });
                }
                TokenTree::Sequence(_, seq) => {
                    // We can't determine `idx_first_after` and construct the final
                    // `MatcherLoc::Sequence` until after `inner()` is called and the sequence end
                    // pieces are processed. So we push a dummy value (`Eof` is cheapest to
                    // construct) now, and overwrite it with the proper value below.
                    let dummy = MatcherLoc::Eof;
                    locs.push(dummy);

                    let next_metavar_orig = *next_metavar;
                    let op = seq.kleene.op;
                    let idx_first = locs.len();
                    let idx_seq = idx_first - 1;
                    inner(&seq.tts, locs, next_metavar, seq_depth + 1);

                    if let Some(separator) = seq.separator {
                        locs.push(MatcherLoc::SequenceSep { separator });
                        locs.push(MatcherLoc::SequenceKleeneOpAfterSep { idx_first });
                    } else {
                        locs.push(MatcherLoc::SequenceKleeneOpNoSep { op, idx_first });
                    }

                    // Overwrite the dummy value pushed above with the proper value.
                    locs[idx_seq] = MatcherLoc::Sequence {
                        op,
                        num_metavar_decls: seq.num_captures,
                        idx_first_after: locs.len(),
                        next_metavar: next_metavar_orig,
                        seq_depth,
                    };
                }
                &TokenTree::MetaVarDecl { span, name: bind, kind } => {
                    locs.push(MatcherLoc::MetaVarDecl {
                        span,
                        bind,
                        kind,
                        next_metavar: *next_metavar,
                        seq_depth,
                    });
                    *next_metavar += 1;
                }
                TokenTree::MetaVar(..) | TokenTree::MetaVarExpr(..) => unreachable!(),
            }
        }
    }

    let mut locs = vec![];
    let mut next_metavar = 0;
    inner(matcher, &mut locs, &mut next_metavar, /* seq_depth */ 0);

    // A final entry is needed for eof.
    locs.push(MatcherLoc::Eof);

    locs
}

/// A single matcher position, representing the state of matching.
#[derive(Debug)]
struct MatcherPos {
    /// The index into `TtParser::locs`, which represents the "dot".
    idx: u32,

    /// The input position being targeted.
    ///
    /// This is an index into or to the end of `seen_tokens`; in the latter case, it then refers
    /// to the latest token from `parser`.
    input_pos: u32,

    /// The matches made against metavar decls so far. On a successful match, this vector ends up
    /// with one element per metavar decl in the matcher. Each element records token trees matched
    /// against the relevant metavar by the black box parser. An element will be a `MatchedSeq` if
    /// the corresponding metavar decl is within a sequence.
    ///
    /// It is critical to performance that this is an `Rc`, because it gets cloned frequently when
    /// processing sequences. Mostly for sequence-ending possibilities that must be tried but end
    /// up failing.
    matches: Rc<Vec<NamedMatch>>,
}

// This type is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(MatcherPos, 16);

impl MatcherPos {
    /// Adds `m` as a named match for the `metavar_idx`-th metavar. There are only two call sites,
    /// and both are hot enough to be always worth inlining.
    #[inline(always)]
    fn push_match(&mut self, metavar_idx: usize, seq_depth: usize, m: NamedMatch) {
        let matches = Rc::make_mut(&mut self.matches);
        match seq_depth {
            0 => {
                // We are not within a sequence. Just append `m`.
                assert_eq!(metavar_idx, matches.len());
                matches.push(m);
            }
            _ => {
                // We are within a sequence. Find the final `MatchedSeq` at the appropriate depth
                // and append `m` to its vector.
                let mut curr = &mut matches[metavar_idx];
                for _ in 0..seq_depth - 1 {
                    match curr {
                        MatchedSeq(seq) => curr = seq.last_mut().unwrap(),
                        _ => unreachable!(),
                    }
                }
                match curr {
                    MatchedSeq(seq) => seq.push(m),
                    _ => unreachable!(),
                }
            }
        }
    }
}

/// Represents the possible results of an attempted parse.
#[derive(Debug)]
pub(crate) enum ParseResult<T> {
    /// Parsed successfully.
    Success(T),
    /// Arm failed to match.
    ///
    /// [`Tracker::failure()`] will be called beforehand.
    Failure,
    /// The input could be parsed in multiple distinct ways.
    ///
    /// [`Tracker::ambiguity()`] will be called beforehand.
    Ambiguity,
    ErrorReported(ErrorGuaranteed),
}

/// A `ParseResult` where the `Success` variant contains a mapping of
/// `MacroRulesNormalizedIdent`s to `NamedMatch`es. This represents the mapping
/// of metavars to the token trees they bind to.
pub(crate) type NamedParseResult = ParseResult<NamedMatches>;

/// Contains a mapping of `MacroRulesNormalizedIdent`s to `NamedMatch`es.
/// This represents the mapping of metavars to the token trees they bind to.
pub(crate) type NamedMatches = FxHashMap<MacroRulesNormalizedIdent, NamedMatch>;

/// Count how many metavars declarations are in `matcher`.
pub(super) fn count_metavar_decls(matcher: &[TokenTree]) -> usize {
    matcher
        .iter()
        .map(|tt| match tt {
            TokenTree::MetaVarDecl { .. } => 1,
            TokenTree::Sequence(_, seq) => seq.num_captures,
            TokenTree::Delimited(.., delim) => count_metavar_decls(&delim.tts),
            TokenTree::Token(..) => 0,
            TokenTree::MetaVar(..) | TokenTree::MetaVarExpr(..) => unreachable!(),
        })
        .sum()
}

/// `NamedMatch` is a pattern-match result for a single metavar. All
/// `MatchedNonterminal`s in the `NamedMatch` have the same non-terminal type
/// (expr, item, etc).
///
/// The in-memory structure of a particular `NamedMatch` represents the match
/// that occurred when a particular subset of a matcher was applied to a
/// particular token tree.
///
/// The width of each `MatchedSeq` in the `NamedMatch`, and the identity of
/// the `MatchedNtNonTts`s, will depend on the token tree it was applied
/// to: each `MatchedSeq` corresponds to a single repetition in the originating
/// token tree. The depth of the `NamedMatch` structure will therefore depend
/// only on the nesting depth of repetitions in the originating token tree it
/// was derived from.
///
/// In layperson's terms: `NamedMatch` will form a tree representing nested matches of a particular
/// meta variable. For example, if we are matching the following macro against the following
/// invocation...
///
/// ```rust
/// macro_rules! foo {
///   ($($($x:ident),+);+) => {}
/// }
///
/// foo!(a, b, c, d; a, b, c, d, e);
/// ```
///
/// Then, the tree will have the following shape:
///
/// ```ignore (private-internal)
/// # use NamedMatch::*;
/// MatchedSeq([
///   MatchedSeq([
///     MatchedNonterminal(a),
///     MatchedNonterminal(b),
///     MatchedNonterminal(c),
///     MatchedNonterminal(d),
///   ]),
///   MatchedSeq([
///     MatchedNonterminal(a),
///     MatchedNonterminal(b),
///     MatchedNonterminal(c),
///     MatchedNonterminal(d),
///     MatchedNonterminal(e),
///   ])
/// ])
/// ```
#[derive(Debug, Clone)]
pub(crate) enum NamedMatch {
    MatchedSeq(Vec<NamedMatch>),
    MatchedSingle(ParseNtResult),
}

impl NamedMatch {
    pub(super) fn is_repeatable(&self) -> bool {
        match self {
            NamedMatch::MatchedSeq(_) => true,
            NamedMatch::MatchedSingle(_) => false,
        }
    }
}

/// Performs a token equality check, ignoring syntax context (that is, an unhygienic comparison)
fn token_name_eq(t1: &Token, t2: &Token) -> bool {
    if let (Some((ident1, is_raw1)), Some((ident2, is_raw2))) = (t1.ident(), t2.ident()) {
        ident1.name == ident2.name && is_raw1 == is_raw2
    } else if let (Some((ident1, is_raw1)), Some((ident2, is_raw2))) =
        (t1.lifetime(), t2.lifetime())
    {
        ident1.name == ident2.name && is_raw1 == is_raw2
    } else {
        // Note: we SHOULD NOT use `t1.kind == t2.kind` here, and we should instead compare the
        // tokens using the special comparison logic below.
        // It makes sure that variants containing `InvisibleOrigin` will
        // never compare equal to one another.
        //
        // When we had AST-based nonterminals we couldn't compare them, and the
        // old `Nonterminal` type had an `eq` that always returned false,
        // resulting in this restriction:
        // <https://doc.rust-lang.org/nightly/reference/macros-by-example.html#forwarding-a-matched-fragment>
        // This comparison logic emulates that behaviour. We could consider lifting this
        // restriction now but there are still cases involving invisible
        // delimiters that make it harder than it first appears.
        match (t1.kind, t2.kind) {
            (TokenKind::OpenInvisible(_) | TokenKind::CloseInvisible(_), _)
            | (_, TokenKind::OpenInvisible(_) | TokenKind::CloseInvisible(_)) => false,
            (a, b) => a == b,
        }
    }
}

// Note: the vectors could be created and dropped within `parse_tt`, but to avoid excess
// allocations we have a single vector for each kind that is cleared and reused repeatedly.
pub(crate) struct TtParser {
    /// mps at older input positions that are yet to be explored.
    ///
    /// Invariant: `backtrack.iter().is_sorted_by_key(|mp| mp.input_pos)`.
    backtrack: Vec<MatcherPos>,

    /// Previously seen tokens from the parser.
    ///
    /// Tokens after the latest meta-variable that have been matched against a fixed token and
    /// [`Parser::bump()`]-ed past are stored here. This list is cleared every time a meta-variable
    /// is parsed.
    seen_tokens: Vec<Token>,

    /// Pre-allocate an empty match array, so it can be cloned cheaply for macros with many rules
    /// that have no metavars.
    empty_matches: Rc<Vec<NamedMatch>>,

    /// A potentially-ambiguous mp waiting to be parsed.
    ///
    /// This is an mp that has been successfully matched, and that is unambiguous iff no other mps
    /// match at the same input position. It is stored here until all other mps have been exhausted.
    /// If another mp conflicts with this, this is left untouched and [`Self::found_ambiguity`] is
    /// set.
    maybe_ambig_mp: Option<MatcherPos>,

    /// Whether an ambiguity error has occurred.
    found_ambiguity: bool,
}

impl TtParser {
    pub(super) fn new() -> TtParser {
        TtParser {
            backtrack: vec![],
            seen_tokens: vec![],
            empty_matches: Rc::new(vec![]),
            maybe_ambig_mp: None,
            found_ambiguity: false,
        }
    }

    /// Process the matcher positions of `cur_mps` until it is empty. In the process, this will
    /// produce more mps in `next_mps` and `bb_mps`.
    ///
    /// # Returns
    ///
    /// `Some(result)` if everything is finished, `None` otherwise. Note that matches are kept
    /// track of through the mps generated.
    fn parse_tt_inner<'matcher, T: Tracker<'matcher>>(
        &mut self,
        parser: &mut Cow<'_, Parser<'_>>,
        matcher: &'matcher [MatcherLoc],
        track: &mut T,
    ) -> Option<NamedParseResult> {
        while let Some(mut mp) = self.backtrack.pop() {
            while let Some(next_mp) = self.match_one(parser, matcher, mp, track) {
                mp = next_mp;
            }

            debug_assert!(self.backtrack.iter().is_sorted_by_key(|mp| mp.input_pos));
        }

        if let Some(mp) = self.maybe_ambig_mp.take() {
            if self.found_ambiguity || parser.approx_token_stream_pos() > mp.input_pos {
                // Either:
                // - A second maybe-ambig mp was found, setting `found_ambiguity`
                // - Something else was parsed successfully, advancing `parser` past `mp`
                // - `mp` was matched while backtracking
                track.ambiguity();
                return Some(Ambiguity);
            }

            return self.process_special(parser, matcher, mp);
        }

        // There are no possible next positions: syntax error.
        track.failure(parser);
        Some(Failure)
    }

    /// Match a single [`MatcherPos`].
    #[inline(always)] // must be inlined in `parse_tt_inner()`
    fn match_one<'matcher, T: Tracker<'matcher>>(
        &mut self,
        parser: &mut Cow<'_, Parser<'_>>,
        matcher: &'matcher [MatcherLoc],
        mut mp: MatcherPos,
        track: &mut T,
    ) -> Option<MatcherPos> {
        let matcher_loc = &matcher[mp.idx as usize];
        // How far from the latest token are we.
        let age = parser.approx_token_stream_pos() - mp.input_pos;
        let token = if age == 0 {
            &parser.token
        } else {
            &self.seen_tokens[self.seen_tokens.len() - age as usize]
        };
        track.trying_match(mp.input_pos, token, mp.idx);

        match matcher_loc {
            MatcherLoc::Token { token: t } => {
                // If it's a doc comment, we just ignore it and move on to the next tt in the
                // matcher. This is a bug, but #95267 showed that existing programs rely on this
                // behaviour, and changing it would require some care and a transition period.
                //
                // If the token matches, we can just advance the parser.
                //
                // Otherwise, this match has failed, there is nothing to do, and hopefully another
                // mp in `cur_mps` will match.
                if matches!(t, Token { kind: DocComment(..), .. }) {
                    // skip
                } else if token_name_eq(t, token) {
                    track.matched_one(mp.input_pos, mp.idx);
                    mp.input_pos += 1;
                    if mp.input_pos > parser.approx_token_stream_pos() {
                        self.seen_tokens.push(parser.token);
                        parser.to_mut().bump();
                        debug_assert_eq!(mp.input_pos, parser.approx_token_stream_pos());
                    }
                } else {
                    return None;
                }
                mp.idx += 1;
                Some(mp)
            }
            MatcherLoc::Delimited => {
                // Entering the delimiter is trivial.
                mp.idx += 1;
                Some(mp)
            }
            &MatcherLoc::Sequence {
                op,
                num_metavar_decls,
                idx_first_after,
                next_metavar,
                seq_depth,
            } => {
                // Install an empty vec for each metavar within the sequence.
                for metavar_idx in next_metavar..next_metavar + num_metavar_decls {
                    mp.push_match(metavar_idx, seq_depth, MatchedSeq(vec![]));
                }

                if matches!(op, KleeneOp::ZeroOrMore | KleeneOp::ZeroOrOne) {
                    // Try zero matches of this sequence, by skipping over it.
                    let idx = idx_first_after.try_into().unwrap();
                    self.backtrack.push(MatcherPos {
                        idx,
                        input_pos: mp.input_pos,
                        matches: Rc::clone(&mp.matches),
                    });
                }

                // Try one or more matches of this sequence, by entering it.
                mp.idx += 1;
                Some(mp)
            }
            &MatcherLoc::SequenceKleeneOpNoSep { op, idx_first } => {
                if op != KleeneOp::ZeroOrOne {
                    // Try another repetition.
                    let repeating_mp = MatcherPos {
                        idx: idx_first.try_into().unwrap(),
                        input_pos: mp.input_pos,
                        matches: Rc::clone(&mp.matches),
                    };
                    self.backtrack.push(repeating_mp);
                }

                // Try ending the sequence.
                mp.idx += 1;
                Some(mp)
            }
            MatcherLoc::SequenceSep { separator } => {
                // We are past the end of a sequence with a separator but we haven't seen the
                // separator yet. Try ending the sequence.
                let ending_mp = MatcherPos {
                    idx: mp.idx + 2, // +2 skips the separator and the Kleene op
                    input_pos: mp.input_pos,
                    matches: Rc::clone(&mp.matches),
                };

                if token_name_eq(token, separator) {
                    // The separator matches the current token. Advance past it.
                    track.matched_one(mp.input_pos, mp.idx);
                    mp.idx += 1;
                    mp.input_pos += 1;
                    if mp.input_pos > parser.approx_token_stream_pos() {
                        self.seen_tokens.push(parser.token);
                        parser.to_mut().bump();
                        debug_assert_eq!(mp.input_pos, parser.approx_token_stream_pos());
                    }
                    self.backtrack.push(ending_mp);
                    Some(mp)
                } else {
                    Some(ending_mp)
                }
            }
            &MatcherLoc::SequenceKleeneOpAfterSep { idx_first } => {
                // We are past the sequence separator. This can't be a `?` Kleene op, because they
                // don't permit separators. Try another repetition.
                mp.idx = idx_first.try_into().unwrap();
                Some(mp)
            }
            &MatcherLoc::MetaVarDecl { kind, .. } => {
                // Built-in nonterminals never start with these tokens, so we can eliminate them
                // from consideration. We use the span of the metavariable declaration to determine
                // any edition-specific matching behavior for non-terminals.
                if !Parser::nonterminal_may_begin_with(kind, token) {
                    return None;
                }

                // EOF tokens would cause unexpected processing in `match_one()`.
                debug_assert!(parser.token != token::Eof, "{kind:?} should not accept EOF tokens");

                track.matched_one(mp.input_pos, mp.idx);

                if self.maybe_ambig_mp.is_some() {
                    self.found_ambiguity = true;
                } else {
                    self.maybe_ambig_mp = Some(mp);
                }

                None
            }
            MatcherLoc::Eof => {
                // We are past the matcher's end, and not in a sequence. Try to end things.
                debug_assert_eq!(mp.idx as usize, matcher.len() - 1);

                if *token != token::Eof {
                    return None;
                }

                debug_assert_eq!(mp.input_pos, parser.approx_token_stream_pos());

                track.matched_one(mp.input_pos, mp.idx);

                if self.maybe_ambig_mp.is_some() {
                    self.found_ambiguity = true;
                } else {
                    self.maybe_ambig_mp = Some(mp);
                }

                None
            }
        }
    }

    /// Finish processing a matched special [`MatcherPos`].
    fn process_special(
        &mut self,
        parser: &mut Cow<'_, Parser<'_>>,
        matcher: &[MatcherLoc],
        mut mp: MatcherPos,
    ) -> Option<NamedParseResult> {
        let matcher_loc = &matcher[mp.idx as usize];
        match matcher_loc {
            &MatcherLoc::MetaVarDecl { kind, next_metavar, seq_depth, .. } => {
                // We use the span of the metavariable declaration to determine any
                // edition-specific matching behavior for non-terminals.
                let nt = match parser.to_mut().parse_nonterminal(kind) {
                    Err(err) => return Some(self.nt_parsing_error(matcher_loc, err)),
                    Ok(nt) => nt,
                };
                mp.push_match(next_metavar, seq_depth, MatchedSingle(nt));

                mp.idx += 1;
                mp.input_pos = parser.approx_token_stream_pos();
                self.seen_tokens.clear();
                self.backtrack.push(mp);
                None
            }

            MatcherLoc::Eof => {
                self.seen_tokens.clear();
                let matches = Rc::unwrap_or_clone(mp.matches).into_iter();
                Some(Success(self.nameize(matcher, matches)))
            }

            _ => unreachable!(),
        }
    }

    /// Match the token stream from `parser` against `matcher`.
    pub(super) fn parse_tt<'matcher, T: Tracker<'matcher>>(
        &mut self,
        parser: &mut Cow<'_, Parser<'_>>,
        matcher: &'matcher [MatcherLoc],
        track: &mut T,
    ) -> NamedParseResult {
        // A queue of possible matcher positions. We initialize it with the matcher position in
        // which the "dot" is before the first token of the first token tree in `matcher`.
        // `parse_tt_inner` then processes all of these possible matcher positions and produces
        // possible next positions into `next_mps`. After some post-processing, the contents of
        // `next_mps` replenish `cur_mps` and we start over again.
        self.backtrack.clear();
        self.seen_tokens.clear();
        self.backtrack.push(MatcherPos {
            idx: 0,
            input_pos: parser.approx_token_stream_pos(),
            matches: Rc::clone(&self.empty_matches),
        });

        loop {
            let res = self.parse_tt_inner(parser, matcher, track);

            if let Some(res) = res {
                return res;
            }
        }
    }

    fn nt_parsing_error<R>(&self, loc: &MatcherLoc, err: Diag<'_>) -> ParseResult<R> {
        let &MatcherLoc::MetaVarDecl { span, kind, .. } = loc else { unreachable!() };
        let guarantee = err
            .with_span_label(
                span,
                format!("while parsing argument for this `{kind}` macro fragment"),
            )
            .emit();
        ErrorReported(guarantee)
    }

    fn nameize<I: Iterator<Item = NamedMatch>>(
        &self,
        matcher: &[MatcherLoc],
        mut res: I,
    ) -> NamedMatches {
        // Make that each metavar has _exactly one_ binding. If so, insert the binding into the
        // `NamedParseResult`. Otherwise, it's an error.
        let mut ret_val = FxHashMap::default();
        for loc in matcher {
            if let &MatcherLoc::MetaVarDecl { span, bind, .. } = loc
                && ret_val
                    .insert(MacroRulesNormalizedIdent::new(bind), res.next().unwrap())
                    .is_some()
            {
                // Duplicate binds are checked for when the macro definition is processed,
                // and should have prevented the definition from ever being used.
                span_bug!(
                    span,
                    "duplicate meta-variable binding went undetected at macro definition"
                )
            }
        }
        ret_val
    }
}
