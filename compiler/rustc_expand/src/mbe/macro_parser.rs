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
//! The parser walks through the input a character at a time, maintaining a list
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

crate use NamedMatch::*;
crate use ParseResult::*;

use crate::mbe::{self, SequenceRepetition, TokenTree};

use rustc_ast::token::{self, DocComment, Nonterminal, Token};
use rustc_parse::parser::{NtOrTt, Parser};
use rustc_session::parse::ParseSess;
use rustc_span::symbol::MacroRulesNormalizedIdent;

use smallvec::{smallvec, SmallVec};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_span::symbol::Ident;
use std::borrow::Cow;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::mem;

/// This is used by `parse_tt_inner` to keep track of delimited submatchers that we have
/// descended into.
#[derive(Clone)]
struct MatcherPosFrame<'tt> {
    /// The "parent" matcher that we have descended from.
    tts: &'tt [TokenTree],
    /// The position of the "dot" in `tt` at the time we descended.
    idx: usize,
}

// One element is enough to cover 95-99% of vectors for most benchmarks. Also,
// vectors longer than one frequently have many elements, not just two or
// three.
type NamedMatchVec = SmallVec<[NamedMatch; 1]>;

// This type is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(NamedMatchVec, 48);

/// A single matcher position, which could be within the top-level matcher, a submatcher, a
/// subsubmatcher, etc. For example:
/// ```text
/// macro_rules! m { $id:ident ( $($e:expr),* ) } => { ... }
///                              <---------->     second submatcher; one tt, one metavar
///                            <-------------->   first submatcher; three tts, zero metavars
///                  <--------------------------> top-level matcher; two tts, one metavar
/// ```
#[derive(Clone)]
struct MatcherPos<'tt> {
    /// The tokens that make up the current matcher. When we are within a `Sequence` or `Delimited`
    /// submatcher, this is just the contents of that submatcher.
    tts: &'tt [TokenTree],

    /// The "dot" position within the current submatcher, i.e. the index into `tts`.
    idx: usize,

    /// This vector ends up with one element per metavar in the *top-level* matcher, even when this
    /// `MatcherPos` is for a submatcher. Each element records token trees matched against the
    /// relevant metavar by the black box parser. The element will be a `MatchedSeq` if the
    /// corresponding metavar is within a sequence.
    matches: Lrc<NamedMatchVec>,

    /// The number of sequences this mp is within.
    seq_depth: usize,

    /// The position in `matches` of the first metavar in this (sub)matcher. Zero if there are
    /// no metavars.
    match_lo: usize,

    /// The position in `matches` of the next metavar to be matched against the source token
    /// stream. Should not be used if there are no metavars.
    match_cur: usize,

    /// This field is only used if we are matching a sequence.
    sequence: Option<MatcherPosSequence<'tt>>,

    /// When we are within a `Delimited` submatcher (or subsubmatcher), this tracks the parent
    /// matcher(s). The bottom of the stack is the top-level matcher.
    stack: SmallVec<[MatcherPosFrame<'tt>; 1]>,
}

// This type is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(MatcherPos<'_>, 104);

impl<'tt> MatcherPos<'tt> {
    fn top_level(matcher: &'tt [TokenTree], empty_matches: Lrc<NamedMatchVec>) -> Self {
        MatcherPos {
            tts: matcher,
            idx: 0,
            matches: empty_matches,
            seq_depth: 0,
            match_lo: 0,
            match_cur: 0,
            stack: smallvec![],
            sequence: None,
        }
    }

    fn sequence(
        parent: Box<MatcherPos<'tt>>,
        seq: &'tt SequenceRepetition,
        empty_matches: Lrc<NamedMatchVec>,
    ) -> Self {
        let mut mp = MatcherPos {
            tts: &seq.tts,
            idx: 0,
            matches: parent.matches.clone(),
            seq_depth: parent.seq_depth,
            match_lo: parent.match_cur,
            match_cur: parent.match_cur,
            sequence: Some(MatcherPosSequence { parent, seq }),
            stack: smallvec![],
        };
        // Start with an empty vec for each metavar within the sequence. Note that `mp.seq_depth`
        // must have the parent's depth at this point for these `push_match` calls to work.
        for idx in mp.match_lo..mp.match_lo + seq.num_captures {
            mp.push_match(idx, MatchedSeq(empty_matches.clone()));
        }
        mp.seq_depth += 1;
        mp
    }

    /// Adds `m` as a named match for the `idx`-th metavar.
    fn push_match(&mut self, idx: usize, m: NamedMatch) {
        let matches = Lrc::make_mut(&mut self.matches);
        match self.seq_depth {
            0 => {
                // We are not within a sequence. Just append `m`.
                assert_eq!(idx, matches.len());
                matches.push(m);
            }
            _ => {
                // We are within a sequence. Find the final `MatchedSeq` at the appropriate depth
                // and append `m` to its vector.
                let mut curr = &mut matches[idx];
                for _ in 0..self.seq_depth - 1 {
                    match curr {
                        MatchedSeq(seq) => {
                            let seq = Lrc::make_mut(seq);
                            curr = seq.last_mut().unwrap();
                        }
                        _ => unreachable!(),
                    }
                }
                match curr {
                    MatchedSeq(seq) => {
                        let seq = Lrc::make_mut(seq);
                        seq.push(m);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[derive(Clone)]
struct MatcherPosSequence<'tt> {
    /// The parent matcher position. Effectively gives a linked list of matches all the way to the
    /// top-level matcher.
    parent: Box<MatcherPos<'tt>>,

    /// The sequence itself.
    seq: &'tt SequenceRepetition,
}

enum EofMatcherPositions<'tt> {
    None,
    One(Box<MatcherPos<'tt>>),
    Multiple,
}

/// Represents the possible results of an attempted parse.
crate enum ParseResult<T> {
    /// Parsed successfully.
    Success(T),
    /// Arm failed to match. If the second parameter is `token::Eof`, it indicates an unexpected
    /// end of macro invocation. Otherwise, it indicates that no rules expected the given token.
    Failure(Token, &'static str),
    /// Fatal error (malformed macro?). Abort compilation.
    Error(rustc_span::Span, String),
    ErrorReported,
}

/// A `ParseResult` where the `Success` variant contains a mapping of
/// `MacroRulesNormalizedIdent`s to `NamedMatch`es. This represents the mapping
/// of metavars to the token trees they bind to.
crate type NamedParseResult = ParseResult<FxHashMap<MacroRulesNormalizedIdent, NamedMatch>>;

/// Count how many metavars declarations are in `matcher`.
pub(super) fn count_metavar_decls(matcher: &[TokenTree]) -> usize {
    matcher
        .iter()
        .map(|tt| {
            match tt {
                TokenTree::Delimited(_, delim) => count_metavar_decls(delim.inner_tts()),
                TokenTree::MetaVar(..) => 0,
                TokenTree::MetaVarDecl(..) => 1,
                // RHS meta-variable expressions eventually end-up here. `0` is returned to inform
                // that no meta-variable was found, because "meta-variables" != "meta-variable
                // expressions".
                TokenTree::MetaVarExpr(..) => 0,
                TokenTree::Sequence(_, seq) => seq.num_captures,
                TokenTree::Token(..) => 0,
            }
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
/// In layman's terms: `NamedMatch` will form a tree representing nested matches of a particular
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
/// ```rust
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
crate enum NamedMatch {
    MatchedSeq(Lrc<NamedMatchVec>),

    // A metavar match of type `tt`.
    MatchedTokenTree(rustc_ast::tokenstream::TokenTree),

    // A metavar match of any type other than `tt`.
    MatchedNonterminal(Lrc<Nonterminal>),
}

fn nameize<I: Iterator<Item = NamedMatch>>(
    sess: &ParseSess,
    matcher: &[TokenTree],
    mut res: I,
) -> NamedParseResult {
    // Recursively descend into each type of matcher (e.g., sequences, delimited, metavars) and make
    // sure that each metavar has _exactly one_ binding. If a metavar does not have exactly one
    // binding, then there is an error. If it does, then we insert the binding into the
    // `NamedParseResult`.
    fn n_rec<I: Iterator<Item = NamedMatch>>(
        sess: &ParseSess,
        tt: &TokenTree,
        res: &mut I,
        ret_val: &mut FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    ) -> Result<(), (rustc_span::Span, String)> {
        match *tt {
            TokenTree::Sequence(_, ref seq) => {
                for next_m in &seq.tts {
                    n_rec(sess, next_m, res.by_ref(), ret_val)?
                }
            }
            TokenTree::Delimited(_, ref delim) => {
                for next_m in delim.inner_tts() {
                    n_rec(sess, next_m, res.by_ref(), ret_val)?;
                }
            }
            TokenTree::MetaVarDecl(span, _, None) => {
                if sess.missing_fragment_specifiers.borrow_mut().remove(&span).is_some() {
                    return Err((span, "missing fragment specifier".to_string()));
                }
            }
            TokenTree::MetaVarDecl(sp, bind_name, _) => match ret_val
                .entry(MacroRulesNormalizedIdent::new(bind_name))
            {
                Vacant(spot) => {
                    spot.insert(res.next().unwrap());
                }
                Occupied(..) => return Err((sp, format!("duplicated bind name: {}", bind_name))),
            },
            TokenTree::Token(..) => (),
            TokenTree::MetaVar(..) | TokenTree::MetaVarExpr(..) => unreachable!(),
        }

        Ok(())
    }

    let mut ret_val = FxHashMap::default();
    for tt in matcher {
        match n_rec(sess, tt, res.by_ref(), &mut ret_val) {
            Ok(_) => {}
            Err((sp, msg)) => return Error(sp, msg),
        }
    }

    Success(ret_val)
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

// Note: the position vectors could be created and dropped within `parse_tt`, but to avoid excess
// allocations we have a single vector fo each kind that is cleared and reused repeatedly.
pub struct TtParser<'tt> {
    macro_name: Ident,

    /// The set of current mps to be processed. This should be empty by the end of a successful
    /// execution of `parse_tt_inner`.
    cur_mps: Vec<Box<MatcherPos<'tt>>>,

    /// The set of newly generated mps. These are used to replenish `cur_mps` in the function
    /// `parse_tt`.
    next_mps: Vec<Box<MatcherPos<'tt>>>,

    /// The set of mps that are waiting for the black-box parser.
    bb_mps: Vec<Box<MatcherPos<'tt>>>,

    /// Pre-allocate an empty match array, so it can be cloned cheaply for macros with many rules
    /// that have no metavars.
    empty_matches: Lrc<NamedMatchVec>,
}

impl<'tt> TtParser<'tt> {
    pub(super) fn new(macro_name: Ident) -> TtParser<'tt> {
        TtParser {
            macro_name,
            cur_mps: vec![],
            next_mps: vec![],
            bb_mps: vec![],
            empty_matches: Lrc::new(smallvec![]),
        }
    }

    /// Process the matcher positions of `cur_mps` until it is empty. In the process, this will
    /// produce more mps in `next_mps` and `bb_mps`.
    ///
    /// # Returns
    ///
    /// `Some(result)` if everything is finished, `None` otherwise. Note that matches are kept
    /// track of through the mps generated.
    fn parse_tt_inner(
        &mut self,
        sess: &ParseSess,
        matcher: &[TokenTree],
        token: &Token,
    ) -> Option<NamedParseResult> {
        // Matcher positions that would be valid if the macro invocation was over now. Only
        // modified if `token == Eof`.
        let mut eof_mps = EofMatcherPositions::None;

        while let Some(mut mp) = self.cur_mps.pop() {
            // Backtrack out of delimited submatcher when necessary. When backtracking out again,
            // we need to advance the "dot" past the delimiters in the parent matcher(s).
            while mp.idx >= mp.tts.len() {
                match mp.stack.pop() {
                    Some(MatcherPosFrame { tts, idx }) => {
                        mp.tts = tts;
                        mp.idx = idx + 1;
                    }
                    None => break,
                }
            }

            // Get the current position of the "dot" (`idx`) in `mp` and the number of token
            // trees in the matcher (`len`).
            let idx = mp.idx;
            let len = mp.tts.len();

            if idx < len {
                // We are in the middle of a matcher. Compare the matcher's current tt against
                // `token`.
                match &mp.tts[idx] {
                    TokenTree::Sequence(_sp, seq) => {
                        let op = seq.kleene.op;
                        if op == mbe::KleeneOp::ZeroOrMore || op == mbe::KleeneOp::ZeroOrOne {
                            // Allow for the possibility of zero matches of this sequence.
                            let mut new_mp = mp.clone();
                            new_mp.match_cur += seq.num_captures;
                            new_mp.idx += 1;
                            for idx in mp.match_cur..mp.match_cur + seq.num_captures {
                                new_mp.push_match(idx, MatchedSeq(self.empty_matches.clone()));
                            }
                            self.cur_mps.push(new_mp);
                        }

                        // Allow for the possibility of one or more matches of this sequence.
                        self.cur_mps.push(box MatcherPos::sequence(
                            mp,
                            &seq,
                            self.empty_matches.clone(),
                        ));
                    }

                    &TokenTree::MetaVarDecl(span, _, None) => {
                        // E.g. `$e` instead of `$e:expr`.
                        if sess.missing_fragment_specifiers.borrow_mut().remove(&span).is_some() {
                            return Some(Error(span, "missing fragment specifier".to_string()));
                        }
                    }

                    &TokenTree::MetaVarDecl(_, _, Some(kind)) => {
                        // Built-in nonterminals never start with these tokens, so we can eliminate
                        // them from consideration.
                        //
                        // We use the span of the metavariable declaration to determine any
                        // edition-specific matching behavior for non-terminals.
                        if Parser::nonterminal_may_begin_with(kind, token) {
                            self.bb_mps.push(mp);
                        }
                    }

                    TokenTree::Delimited(_, delimited) => {
                        // To descend into a delimited submatcher, we push the current matcher onto
                        // a stack and push a new mp containing the submatcher onto `cur_mps`.
                        //
                        // At the beginning of the loop, if we reach the end of the delimited
                        // submatcher, we pop the stack to backtrack out of the descent. Note that
                        // we use `all_tts` to include the open and close delimiter tokens.
                        let tts = mem::replace(&mut mp.tts, &delimited.all_tts);
                        let idx = mp.idx;
                        mp.stack.push(MatcherPosFrame { tts, idx });
                        mp.idx = 0;
                        self.cur_mps.push(mp);
                    }

                    TokenTree::Token(t) => {
                        // If it's a doc comment, we just ignore it and move on to the next tt in
                        // the matcher. This is a bug, but #95267 showed that existing programs
                        // rely on this behaviour, and changing it would require some care and a
                        // transition period.
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

                    // These cannot appear in a matcher.
                    TokenTree::MetaVar(..) | TokenTree::MetaVarExpr(..) => unreachable!(),
                }
            } else if let Some(sequence) = &mp.sequence {
                // We are past the end of a sequence.
                debug_assert!(idx <= len + 1);

                if idx == len {
                    // Add all matches from the sequence to `parent`, and move the "dot" past the
                    // sequence in `parent`. This allows for the case where the sequence matching
                    // is finished.
                    let mut new_mp = sequence.parent.clone();
                    new_mp.matches = mp.matches.clone();
                    new_mp.match_cur = mp.match_lo + sequence.seq.num_captures;
                    new_mp.idx += 1;
                    self.cur_mps.push(new_mp);
                }

                if idx == len && sequence.seq.separator.is_some() {
                    if sequence
                        .seq
                        .separator
                        .as_ref()
                        .map_or(false, |sep| token_name_eq(token, sep))
                    {
                        // The matcher has a separator, and it matches the current token. We can
                        // advance past the separator token.
                        mp.idx += 1;
                        self.next_mps.push(mp);
                    }
                } else if sequence.seq.kleene.op != mbe::KleeneOp::ZeroOrOne {
                    // We don't need a separator. Move the "dot" back to the beginning of the
                    // matcher and try to match again UNLESS we are only allowed to have _one_
                    // repetition.
                    mp.match_cur = mp.match_lo;
                    mp.idx = 0;
                    self.cur_mps.push(mp);
                }
            } else {
                // We are past the end of the matcher, and not in a sequence. Look for end of
                // input.
                debug_assert_eq!(idx, len);
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

        // If we reached the end of input, check that there is EXACTLY ONE possible matcher.
        // Otherwise, either the parse is ambiguous (which is an error) or there is a syntax error.
        if *token == token::Eof {
            Some(match eof_mps {
                EofMatcherPositions::One(mut eof_mp) => {
                    assert_eq!(eof_mp.matches.len(), count_metavar_decls(matcher));
                    // Need to take ownership of the matches from within the `Lrc`.
                    Lrc::make_mut(&mut eof_mp.matches);
                    let matches = Lrc::try_unwrap(eof_mp.matches).unwrap().into_iter();
                    nameize(sess, matcher, matches)
                }
                EofMatcherPositions::Multiple => {
                    Error(token.span, "ambiguity: multiple successful parses".to_string())
                }
                EofMatcherPositions::None => Failure(
                    Token::new(
                        token::Eof,
                        if token.span.is_dummy() { token.span } else { token.span.shrink_to_hi() },
                    ),
                    "missing tokens in macro arguments",
                ),
            })
        } else {
            None
        }
    }

    /// Match the token stream from `parser` against `matcher`.
    pub(super) fn parse_tt(
        &mut self,
        parser: &mut Cow<'_, Parser<'_>>,
        matcher: &'tt [TokenTree],
    ) -> NamedParseResult {
        // A queue of possible matcher positions. We initialize it with the matcher position in
        // which the "dot" is before the first token of the first token tree in `matcher`.
        // `parse_tt_inner` then processes all of these possible matcher positions and produces
        // possible next positions into `next_mps`. After some post-processing, the contents of
        // `next_mps` replenish `cur_mps` and we start over again.
        self.cur_mps.clear();
        self.cur_mps.push(box MatcherPos::top_level(matcher, self.empty_matches.clone()));

        loop {
            self.next_mps.clear();
            self.bb_mps.clear();

            // Process `cur_mps` until either we have finished the input or we need to get some
            // parsing from the black-box parser done.
            if let Some(result) = self.parse_tt_inner(parser.sess, matcher, &parser.token) {
                return result;
            }

            // `parse_tt_inner` handled all of `cur_mps`, so it's empty.
            assert!(self.cur_mps.is_empty());

            // Error messages here could be improved with links to original rules.
            match (self.next_mps.len(), self.bb_mps.len()) {
                (0, 0) => {
                    // There are no possible next positions AND we aren't waiting for the black-box
                    // parser: syntax error.
                    return Failure(
                        parser.token.clone(),
                        "no rules expected this token in macro call",
                    );
                }

                (_, 0) => {
                    // Dump all possible `next_mps` into `cur_mps` for the next iteration. Then
                    // process the next token.
                    self.cur_mps.extend(self.next_mps.drain(..));
                    parser.to_mut().bump();
                }

                (0, 1) => {
                    // We need to call the black-box parser to get some nonterminal.
                    let mut mp = self.bb_mps.pop().unwrap();
                    if let TokenTree::MetaVarDecl(span, _, Some(kind)) = mp.tts[mp.idx] {
                        let match_cur = mp.match_cur;
                        // We use the span of the metavariable declaration to determine any
                        // edition-specific matching behavior for non-terminals.
                        let nt = match parser.to_mut().parse_nonterminal(kind) {
                            Err(mut err) => {
                                err.span_label(
                                    span,
                                    format!(
                                        "while parsing argument for this `{kind}` macro fragment"
                                    ),
                                )
                                .emit();
                                return ErrorReported;
                            }
                            Ok(nt) => nt,
                        };
                        let m = match nt {
                            NtOrTt::Nt(nt) => MatchedNonterminal(Lrc::new(nt)),
                            NtOrTt::Tt(tt) => MatchedTokenTree(tt),
                        };
                        mp.push_match(match_cur, m);
                        mp.idx += 1;
                        mp.match_cur += 1;
                    } else {
                        unreachable!()
                    }
                    self.cur_mps.push(mp);
                }

                (_, _) => {
                    // Too many possibilities!
                    return self.ambiguity_error(parser.token.span);
                }
            }

            assert!(!self.cur_mps.is_empty());
        }
    }

    fn ambiguity_error(&self, token_span: rustc_span::Span) -> NamedParseResult {
        let nts = self
            .bb_mps
            .iter()
            .map(|mp| match mp.tts[mp.idx] {
                TokenTree::MetaVarDecl(_, bind, Some(kind)) => {
                    format!("{} ('{}')", kind, bind)
                }
                _ => panic!(),
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
                    1 => format!("built-in NTs {} or 1 other option.", nts),
                    n => format!("built-in NTs {} or {} other options.", nts, n),
                }
            ),
        )
    }
}
