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

pub(crate) use NamedMatch::*;
pub(crate) use ParseResult::*;

use crate::mbe::{macro_rules::Tracker, KleeneOp, TokenTree};

use rustc_ast::token::{self, DocComment, Nonterminal, NonterminalKind, Token};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_errors::ErrorGuaranteed;
use rustc_lint_defs::pluralize;
use rustc_parse::parser::{NtOrTt, Parser};
use rustc_span::symbol::Ident;
use rustc_span::symbol::MacroRulesNormalizedIdent;
use rustc_span::Span;
use std::borrow::Cow;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::fmt::Display;
use std::rc::Rc;

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
        kind: Option<NonterminalKind>,
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
                write!(f, "`{}`", pprust::token_to_string(token))
            }
            MatcherLoc::MetaVarDecl { bind, kind, .. } => {
                write!(f, "meta-variable `${bind}")?;
                if let Some(kind) = kind {
                    write!(f, ":{}", kind)?;
                }
                write!(f, "`")?;
                Ok(())
            }
            MatcherLoc::Eof => f.write_str("end of macro"),

            // These are not printed in the diagnostic
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
                    locs.push(MatcherLoc::Token { token: token.clone() });
                }
                TokenTree::Delimited(span, delimited) => {
                    let open_token = Token::new(token::OpenDelim(delimited.delim), span.open);
                    let close_token = Token::new(token::CloseDelim(delimited.delim), span.close);

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

                    if let Some(separator) = &seq.separator {
                        locs.push(MatcherLoc::SequenceSep { separator: separator.clone() });
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
                &TokenTree::MetaVarDecl(span, bind, kind) => {
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
    idx: usize,

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
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
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

enum EofMatcherPositions {
    None,
    One(MatcherPos),
    Multiple,
}

/// Represents the possible results of an attempted parse.
pub(crate) enum ParseResult<T, F> {
    /// Parsed successfully.
    Success(T),
    /// Arm failed to match. If the second parameter is `token::Eof`, it indicates an unexpected
    /// end of macro invocation. Otherwise, it indicates that no rules expected the given token.
    /// The usize is the approximate position of the token in the input token stream.
    Failure(F),
    /// Fatal error (malformed macro?). Abort compilation.
    Error(rustc_span::Span, String),
    ErrorReported(ErrorGuaranteed),
}

/// A `ParseResult` where the `Success` variant contains a mapping of
/// `MacroRulesNormalizedIdent`s to `NamedMatch`es. This represents the mapping
/// of metavars to the token trees they bind to.
pub(crate) type NamedParseResult<F> = ParseResult<NamedMatches, F>;

/// Contains a mapping of `MacroRulesNormalizedIdent`s to `NamedMatch`es.
/// This represents the mapping of metavars to the token trees they bind to.
pub(crate) type NamedMatches = FxHashMap<MacroRulesNormalizedIdent, NamedMatch>;

/// Count how many metavars declarations are in `matcher`.
pub(super) fn count_metavar_decls(matcher: &[TokenTree]) -> usize {
    matcher
        .iter()
        .map(|tt| match tt {
            TokenTree::MetaVarDecl(..) => 1,
            TokenTree::Sequence(_, seq) => seq.num_captures,
            TokenTree::Delimited(_, delim) => count_metavar_decls(&delim.tts),
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

    // A metavar match of type `tt`.
    MatchedTokenTree(rustc_ast::tokenstream::TokenTree),

    // A metavar match of any type other than `tt`.
    MatchedNonterminal(Lrc<Nonterminal>),
}

/// Performs a token equality check, ignoring syntax context (that is, an unhygienic comparison)
fn token_name_eq(t1: &Token, t2: &Token) -> bool {
    if let (Some((ident1, is_raw1)), Some((ident2, is_raw2))) = (t1.ident(), t2.ident()) {
        ident1.name == ident2.name && is_raw1 == is_raw2
    } else if let (Some(ident1), Some(ident2)) = (t1.lifetime(), t2.lifetime()) {
        ident1.name == ident2.name
    } else {
        t1.kind == t2.kind
    }
}

// Note: the vectors could be created and dropped within `parse_tt`, but to avoid excess
// allocations we have a single vector for each kind that is cleared and reused repeatedly.
pub struct TtParser {
    macro_name: Ident,

    /// The set of current mps to be processed. This should be empty by the end of a successful
    /// execution of `parse_tt_inner`.
    cur_mps: Vec<MatcherPos>,

    /// The set of newly generated mps. These are used to replenish `cur_mps` in the function
    /// `parse_tt`.
    next_mps: Vec<MatcherPos>,

    /// The set of mps that are waiting for the black-box parser.
    bb_mps: Vec<MatcherPos>,

    /// Pre-allocate an empty match array, so it can be cloned cheaply for macros with many rules
    /// that have no metavars.
    empty_matches: Rc<Vec<NamedMatch>>,
}

impl TtParser {
    pub(super) fn new(macro_name: Ident) -> TtParser {
        TtParser {
            macro_name,
            cur_mps: vec![],
            next_mps: vec![],
            bb_mps: vec![],
            empty_matches: Rc::new(vec![]),
        }
    }

    pub(super) fn has_no_remaining_items_for_step(&self) -> bool {
        self.cur_mps.is_empty()
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
        matcher: &'matcher [MatcherLoc],
        token: &Token,
        approx_position: usize,
        track: &mut T,
    ) -> Option<NamedParseResult<T::Failure>> {
        // Matcher positions that would be valid if the macro invocation was over now. Only
        // modified if `token == Eof`.
        let mut eof_mps = EofMatcherPositions::None;

        while let Some(mut mp) = self.cur_mps.pop() {
            let matcher_loc = &matcher[mp.idx];
            track.before_match_loc(self, matcher_loc);

            match matcher_loc {
                MatcherLoc::Token { token: t } => {
                    // If it's a doc comment, we just ignore it and move on to the next tt in the
                    // matcher. This is a bug, but #95267 showed that existing programs rely on
                    // this behaviour, and changing it would require some care and a transition
                    // period.
                    //
                    // If the token matches, we can just advance the parser.
                    //
                    // Otherwise, this match has failed, there is nothing to do, and hopefully
                    // another mp in `cur_mps` will match.
                    if matches!(t, Token { kind: DocComment(..), .. }) {
                        mp.idx += 1;
                        self.cur_mps.push(mp);
                    } else if token_name_eq(&t, token) {
                        mp.idx += 1;
                        self.next_mps.push(mp);
                    }
                }
                MatcherLoc::Delimited => {
                    // Entering the delimiter is trivial.
                    mp.idx += 1;
                    self.cur_mps.push(mp);
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
                        self.cur_mps.push(MatcherPos {
                            idx: idx_first_after,
                            matches: Rc::clone(&mp.matches),
                        });
                    }

                    // Try one or more matches of this sequence, by entering it.
                    mp.idx += 1;
                    self.cur_mps.push(mp);
                }
                &MatcherLoc::SequenceKleeneOpNoSep { op, idx_first } => {
                    // We are past the end of a sequence with no separator. Try ending the
                    // sequence. If that's not possible, `ending_mp` will fail quietly when it is
                    // processed next time around the loop.
                    let ending_mp = MatcherPos {
                        idx: mp.idx + 1, // +1 skips the Kleene op
                        matches: Rc::clone(&mp.matches),
                    };
                    self.cur_mps.push(ending_mp);

                    if op != KleeneOp::ZeroOrOne {
                        // Try another repetition.
                        mp.idx = idx_first;
                        self.cur_mps.push(mp);
                    }
                }
                MatcherLoc::SequenceSep { separator } => {
                    // We are past the end of a sequence with a separator but we haven't seen the
                    // separator yet. Try ending the sequence. If that's not possible, `ending_mp`
                    // will fail quietly when it is processed next time around the loop.
                    let ending_mp = MatcherPos {
                        idx: mp.idx + 2, // +2 skips the separator and the Kleene op
                        matches: Rc::clone(&mp.matches),
                    };
                    self.cur_mps.push(ending_mp);

                    if token_name_eq(token, separator) {
                        // The separator matches the current token. Advance past it.
                        mp.idx += 1;
                        self.next_mps.push(mp);
                    }
                }
                &MatcherLoc::SequenceKleeneOpAfterSep { idx_first } => {
                    // We are past the sequence separator. This can't be a `?` Kleene op, because
                    // they don't permit separators. Try another repetition.
                    mp.idx = idx_first;
                    self.cur_mps.push(mp);
                }
                &MatcherLoc::MetaVarDecl { span, kind, .. } => {
                    // Built-in nonterminals never start with these tokens, so we can eliminate
                    // them from consideration. We use the span of the metavariable declaration
                    // to determine any edition-specific matching behavior for non-terminals.
                    if let Some(kind) = kind {
                        if Parser::nonterminal_may_begin_with(kind, token) {
                            self.bb_mps.push(mp);
                        }
                    } else {
                        // E.g. `$e` instead of `$e:expr`, reported as a hard error if actually used.
                        // Both this check and the one in `nameize` are necessary, surprisingly.
                        return Some(Error(span, "missing fragment specifier".to_string()));
                    }
                }
                MatcherLoc::Eof => {
                    // We are past the matcher's end, and not in a sequence. Try to end things.
                    debug_assert_eq!(mp.idx, matcher.len() - 1);
                    if *token == token::Eof {
                        eof_mps = match eof_mps {
                            EofMatcherPositions::None => EofMatcherPositions::One(mp),
                            EofMatcherPositions::One(_) | EofMatcherPositions::Multiple => {
                                EofMatcherPositions::Multiple
                            }
                        }
                    }
                }
            }
        }

        // If we reached the end of input, check that there is EXACTLY ONE possible matcher.
        // Otherwise, either the parse is ambiguous (which is an error) or there is a syntax error.
        if *token == token::Eof {
            Some(match eof_mps {
                EofMatcherPositions::One(mut eof_mp) => {
                    // Need to take ownership of the matches from within the `Rc`.
                    Rc::make_mut(&mut eof_mp.matches);
                    let matches = Rc::try_unwrap(eof_mp.matches).unwrap().into_iter();
                    self.nameize(matcher, matches)
                }
                EofMatcherPositions::Multiple => {
                    Error(token.span, "ambiguity: multiple successful parses".to_string())
                }
                EofMatcherPositions::None => Failure(T::build_failure(
                    Token::new(
                        token::Eof,
                        if token.span.is_dummy() { token.span } else { token.span.shrink_to_hi() },
                    ),
                    approx_position,
                    "missing tokens in macro arguments",
                )),
            })
        } else {
            None
        }
    }

    /// Match the token stream from `parser` against `matcher`.
    pub(super) fn parse_tt<'matcher, T: Tracker<'matcher>>(
        &mut self,
        parser: &mut Cow<'_, Parser<'_>>,
        matcher: &'matcher [MatcherLoc],
        track: &mut T,
    ) -> NamedParseResult<T::Failure> {
        // A queue of possible matcher positions. We initialize it with the matcher position in
        // which the "dot" is before the first token of the first token tree in `matcher`.
        // `parse_tt_inner` then processes all of these possible matcher positions and produces
        // possible next positions into `next_mps`. After some post-processing, the contents of
        // `next_mps` replenish `cur_mps` and we start over again.
        self.cur_mps.clear();
        self.cur_mps.push(MatcherPos { idx: 0, matches: self.empty_matches.clone() });

        loop {
            self.next_mps.clear();
            self.bb_mps.clear();

            // Process `cur_mps` until either we have finished the input or we need to get some
            // parsing from the black-box parser done.
            let res = self.parse_tt_inner(
                matcher,
                &parser.token,
                parser.approx_token_stream_pos(),
                track,
            );
            if let Some(res) = res {
                return res;
            }

            // `parse_tt_inner` handled all of `cur_mps`, so it's empty.
            assert!(self.cur_mps.is_empty());

            // Error messages here could be improved with links to original rules.
            match (self.next_mps.len(), self.bb_mps.len()) {
                (0, 0) => {
                    // There are no possible next positions AND we aren't waiting for the black-box
                    // parser: syntax error.
                    return Failure(T::build_failure(
                        parser.token.clone(),
                        parser.approx_token_stream_pos(),
                        "no rules expected this token in macro call",
                    ));
                }

                (_, 0) => {
                    // Dump all possible `next_mps` into `cur_mps` for the next iteration. Then
                    // process the next token.
                    self.cur_mps.append(&mut self.next_mps);
                    parser.to_mut().bump();
                }

                (0, 1) => {
                    // We need to call the black-box parser to get some nonterminal.
                    let mut mp = self.bb_mps.pop().unwrap();
                    let loc = &matcher[mp.idx];
                    if let &MatcherLoc::MetaVarDecl {
                        span,
                        kind: Some(kind),
                        next_metavar,
                        seq_depth,
                        ..
                    } = loc
                    {
                        // We use the span of the metavariable declaration to determine any
                        // edition-specific matching behavior for non-terminals.
                        let nt = match parser.to_mut().parse_nonterminal(kind) {
                            Err(mut err) => {
                                let guarantee = err.span_label(
                                    span,
                                    format!(
                                        "while parsing argument for this `{kind}` macro fragment"
                                    ),
                                )
                                .emit();
                                return ErrorReported(guarantee);
                            }
                            Ok(nt) => nt,
                        };
                        let m = match nt {
                            NtOrTt::Nt(nt) => MatchedNonterminal(Lrc::new(nt)),
                            NtOrTt::Tt(tt) => MatchedTokenTree(tt),
                        };
                        mp.push_match(next_metavar, seq_depth, m);
                        mp.idx += 1;
                    } else {
                        unreachable!()
                    }
                    self.cur_mps.push(mp);
                }

                (_, _) => {
                    // Too many possibilities!
                    return self.ambiguity_error(matcher, parser.token.span);
                }
            }

            assert!(!self.cur_mps.is_empty());
        }
    }

    fn ambiguity_error<F>(
        &self,
        matcher: &[MatcherLoc],
        token_span: rustc_span::Span,
    ) -> NamedParseResult<F> {
        let nts = self
            .bb_mps
            .iter()
            .map(|mp| match &matcher[mp.idx] {
                MatcherLoc::MetaVarDecl { bind, kind: Some(kind), .. } => {
                    format!("{} ('{}')", kind, bind)
                }
                _ => unreachable!(),
            })
            .collect::<Vec<String>>()
            .join(" or ");

        Error(
            token_span,
            format!(
                "local ambiguity when calling macro `{}`: multiple parsing options: {}",
                self.macro_name,
                match self.next_mps.len() {
                    0 => format!("built-in NTs {}.", nts),
                    n => format!("built-in NTs {} or {n} other option{s}.", nts, s = pluralize!(n)),
                }
            ),
        )
    }

    fn nameize<I: Iterator<Item = NamedMatch>, F>(
        &self,
        matcher: &[MatcherLoc],
        mut res: I,
    ) -> NamedParseResult<F> {
        // Make that each metavar has _exactly one_ binding. If so, insert the binding into the
        // `NamedParseResult`. Otherwise, it's an error.
        let mut ret_val = FxHashMap::default();
        for loc in matcher {
            if let &MatcherLoc::MetaVarDecl { span, bind, kind, .. } = loc {
                if kind.is_some() {
                    match ret_val.entry(MacroRulesNormalizedIdent::new(bind)) {
                        Vacant(spot) => spot.insert(res.next().unwrap()),
                        Occupied(..) => {
                            return Error(span, format!("duplicated bind name: {}", bind));
                        }
                    };
                } else {
                    // E.g. `$e` instead of `$e:expr`, reported as a hard error if actually used.
                    // Both this check and the one in `parse_tt_inner` are necessary, surprisingly.
                    return Error(span, "missing fragment specifier".to_string());
                }
            }
        }
        Success(ret_val)
    }
}
