//! This is an NFA-based parser, which calls out to the main Rust parser for named non-terminals
//! (which it commits to fully when it hits one in a grammar). There's a set of current NFA threads
//! and a set of next ones. Instead of NTs, we have a special case for Kleene star. The big-O, in
//! pathological cases, is worse than traditional use of NFA or Earley parsing, but it's an easier
//! fit for Macro-by-Example-style rules.
//!
//! (In order to prevent the pathological case, we'd need to lazily construct the resulting
//! `NamedMatch`es at the very end. It'd be a pain, and require more memory to keep around old
//! items, but it would also save overhead)
//!
//! We don't say this parser uses the Earley algorithm, because it's unnecessarily inaccurate.
//! The macro parser restricts itself to the features of finite state automata. Earley parsers
//! can be described as an extension of NFAs with completion rules, prediction rules, and recursion.
//!
//! Quick intro to how the parser works:
//!
//! A 'position' is a dot in the middle of a matcher, usually represented as a
//! dot. For example `· a $( a )* a b` is a position, as is `a $( · a )* a b`.
//!
//! The parser walks through the input a character at a time, maintaining a list
//! of threads consistent with the current position in the input string: `cur_items`.
//!
//! As it processes them, it fills up `eof_items` with threads that would be valid if
//! the macro invocation is now over, `bb_items` with threads that are waiting on
//! a Rust non-terminal like `$e:expr`, and `next_items` with threads that are waiting
//! on a particular token. Most of the logic concerns moving the · through the
//! repetitions indicated by Kleene stars. The rules for moving the · without
//! consuming any input are called epsilon transitions. It only advances or calls
//! out to the real Rust parser when no `cur_items` threads remain.
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
//! Descend/Skip (first item).
//! next: [a $( · a )* a b]  [a $( a )* · a b].
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over a b. - - -
//!
//! Remaining input: ''
//! eof: [a $( a )* a b ·]
//! ```

crate use NamedMatch::*;
crate use ParseResult::*;
use TokenTreeOrTokenTreeSlice::*;

use crate::mbe::{self, DelimSpan, SequenceRepetition, TokenTree};

use rustc_ast::token::{self, DocComment, Nonterminal, Token};
use rustc_parse::parser::Parser;
use rustc_session::parse::ParseSess;
use rustc_span::symbol::MacroRulesNormalizedIdent;

use smallvec::{smallvec, SmallVec};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_span::symbol::Ident;
use std::borrow::Cow;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::mem;
use std::ops::{Deref, DerefMut};

// To avoid costly uniqueness checks, we require that `MatchSeq` always has a nonempty body.

/// Either a slice of token trees or a single one. This is used as the representation of the
/// token trees that make up a matcher.
#[derive(Clone)]
enum TokenTreeOrTokenTreeSlice<'tt> {
    Tt(TokenTree),
    TtSlice(&'tt [TokenTree]),
}

impl<'tt> TokenTreeOrTokenTreeSlice<'tt> {
    /// Returns the number of constituent top-level token trees of `self` (top-level in that it
    /// will not recursively descend into subtrees).
    fn len(&self) -> usize {
        match *self {
            TtSlice(ref v) => v.len(),
            Tt(ref tt) => tt.len(),
        }
    }

    /// The `index`-th token tree of `self`.
    fn get_tt(&self, index: usize) -> TokenTree {
        match *self {
            TtSlice(ref v) => v[index].clone(),
            Tt(ref tt) => tt.get_tt(index),
        }
    }
}

/// An unzipping of `TokenTree`s... see the `stack` field of `MatcherPos`.
///
/// This is used by `parse_tt_inner` to keep track of delimited submatchers that we have
/// descended into.
#[derive(Clone)]
struct MatcherTtFrame<'tt> {
    /// The "parent" matcher that we are descending into.
    elts: TokenTreeOrTokenTreeSlice<'tt>,
    /// The position of the "dot" in `elts` at the time we descended.
    idx: usize,
}

type NamedMatchVec = SmallVec<[NamedMatch; 4]>;

/// Represents a single "position" (aka "matcher position", aka "item"), as
/// described in the module documentation.
///
/// Here:
///
/// - `'root` represents the lifetime of the stack slot that holds the root
///   `MatcherPos`. As described in `MatcherPosHandle`, the root `MatcherPos`
///   structure is stored on the stack, but subsequent instances are put into
///   the heap.
/// - `'tt` represents the lifetime of the token trees that this matcher
///   position refers to.
///
/// It is important to distinguish these two lifetimes because we have a
/// `SmallVec<TokenTreeOrTokenTreeSlice<'tt>>` below, and the destructor of
/// that is considered to possibly access the data from its elements (it lacks
/// a `#[may_dangle]` attribute). As a result, the compiler needs to know that
/// all the elements in that `SmallVec` strictly outlive the root stack slot
/// lifetime. By separating `'tt` from `'root`, we can show that.
#[derive(Clone)]
struct MatcherPos<'root, 'tt> {
    /// The token or slice of tokens that make up the matcher. `elts` is short for "elements".
    top_elts: TokenTreeOrTokenTreeSlice<'tt>,

    /// The position of the "dot" in this matcher
    idx: usize,

    /// For each named metavar in the matcher, we keep track of token trees matched against the
    /// metavar by the black box parser. In particular, there may be more than one match per
    /// metavar if we are in a repetition (each repetition matches each of the variables).
    /// Moreover, matchers and repetitions can be nested; the `matches` field is shared (hence the
    /// `Rc`) among all "nested" matchers. `match_lo`, `match_cur`, and `match_hi` keep track of
    /// the current position of the `self` matcher position in the shared `matches` list.
    ///
    /// Also, note that while we are descending into a sequence, matchers are given their own
    /// `matches` vector. Only once we reach the end of a full repetition of the sequence do we add
    /// all bound matches from the submatcher into the shared top-level `matches` vector. If `sep`
    /// and `up` are `Some`, then `matches` is _not_ the shared top-level list. Instead, if one
    /// wants the shared `matches`, one should use `up.matches`.
    matches: Box<[Lrc<NamedMatchVec>]>,
    /// The position in `matches` corresponding to the first metavar in this matcher's sequence of
    /// token trees. In other words, the first metavar in the first token of `top_elts` corresponds
    /// to `matches[match_lo]`.
    match_lo: usize,
    /// The position in `matches` corresponding to the metavar we are currently trying to match
    /// against the source token stream. `match_lo <= match_cur <= match_hi`.
    match_cur: usize,
    /// Similar to `match_lo` except `match_hi` is the position in `matches` of the _last_ metavar
    /// in this matcher.
    match_hi: usize,

    /// This field is only used if we are matching a repetition.
    repetition: Option<MatcherPosRepetition<'root, 'tt>>,

    /// Specifically used to "unzip" token trees. By "unzip", we mean to unwrap the delimiters from
    /// a delimited token tree (e.g., something wrapped in `(` `)`) or to get the contents of a doc
    /// comment...
    ///
    /// When matching against matchers with nested delimited submatchers (e.g., `pat ( pat ( .. )
    /// pat ) pat`), we need to keep track of the matchers we are descending into. This stack does
    /// that where the bottom of the stack is the outermost matcher.
    /// Also, throughout the comments, this "descent" is often referred to as "unzipping"...
    stack: SmallVec<[MatcherTtFrame<'tt>; 1]>,
}

// This type is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(MatcherPos<'_, '_>, 240);

impl<'root, 'tt> MatcherPos<'root, 'tt> {
    /// `len` `Vec`s (initially shared and empty) that will store matches of metavars.
    fn create_matches(len: usize) -> Box<[Lrc<NamedMatchVec>]> {
        if len == 0 {
            vec![]
        } else {
            let empty_matches = Lrc::new(SmallVec::new());
            vec![empty_matches; len]
        }
        .into_boxed_slice()
    }

    /// Generates the top-level matcher position in which the "dot" is before the first token of
    /// the matcher `ms`.
    fn new(ms: &'tt [TokenTree]) -> Self {
        let match_idx_hi = count_names(ms);
        MatcherPos {
            // Start with the top level matcher given to us.
            top_elts: TtSlice(ms),

            // The "dot" is before the first token of the matcher.
            idx: 0,

            // Initialize `matches` to a bunch of empty `Vec`s -- one for each metavar in
            // `top_elts`. `match_lo` for `top_elts` is 0 and `match_hi` is `match_idx_hi`.
            // `match_cur` is 0 since we haven't actually matched anything yet.
            matches: Self::create_matches(match_idx_hi),
            match_lo: 0,
            match_cur: 0,
            match_hi: match_idx_hi,

            // Haven't descended into any delimiters, so this is empty.
            stack: smallvec![],

            // Haven't descended into any sequences, so this is `None`.
            repetition: None,
        }
    }

    fn repetition(
        up: MatcherPosHandle<'root, 'tt>,
        sp: DelimSpan,
        seq: Lrc<SequenceRepetition>,
    ) -> Self {
        MatcherPos {
            stack: smallvec![],
            idx: 0,
            matches: Self::create_matches(up.matches.len()),
            match_lo: up.match_cur,
            match_cur: up.match_cur,
            match_hi: up.match_cur + seq.num_captures,
            repetition: Some(MatcherPosRepetition {
                up,
                sep: seq.separator.clone(),
                seq_op: seq.kleene.op,
            }),
            top_elts: Tt(TokenTree::Sequence(sp, seq)),
        }
    }

    /// Adds `m` as a named match for the `idx`-th metavar.
    fn push_match(&mut self, idx: usize, m: NamedMatch) {
        let matches = Lrc::make_mut(&mut self.matches[idx]);
        matches.push(m);
    }
}

#[derive(Clone)]
struct MatcherPosRepetition<'root, 'tt> {
    /// The KleeneOp of this sequence.
    seq_op: mbe::KleeneOp,

    /// The separator.
    sep: Option<Token>,

    /// The "parent" matcher position. That is, the matcher position just before we enter the
    /// sequence.
    up: MatcherPosHandle<'root, 'tt>,
}

// Lots of MatcherPos instances are created at runtime. Allocating them on the
// heap is slow. Furthermore, using SmallVec<MatcherPos> to allocate them all
// on the stack is also slow, because MatcherPos is quite a large type and
// instances get moved around a lot between vectors, which requires lots of
// slow memcpy calls.
//
// Therefore, the initial MatcherPos is always allocated on the stack,
// subsequent ones (of which there aren't that many) are allocated on the heap,
// and this type is used to encapsulate both cases.
enum MatcherPosHandle<'root, 'tt> {
    Ref(&'root mut MatcherPos<'root, 'tt>),
    Box(Box<MatcherPos<'root, 'tt>>),
}

impl<'root, 'tt> Clone for MatcherPosHandle<'root, 'tt> {
    // This always produces a new Box.
    fn clone(&self) -> Self {
        MatcherPosHandle::Box(match *self {
            MatcherPosHandle::Ref(ref r) => Box::new((**r).clone()),
            MatcherPosHandle::Box(ref b) => b.clone(),
        })
    }
}

impl<'root, 'tt> Deref for MatcherPosHandle<'root, 'tt> {
    type Target = MatcherPos<'root, 'tt>;
    fn deref(&self) -> &Self::Target {
        match *self {
            MatcherPosHandle::Ref(ref r) => r,
            MatcherPosHandle::Box(ref b) => b,
        }
    }
}

impl<'root, 'tt> DerefMut for MatcherPosHandle<'root, 'tt> {
    fn deref_mut(&mut self) -> &mut MatcherPos<'root, 'tt> {
        match *self {
            MatcherPosHandle::Ref(ref mut r) => r,
            MatcherPosHandle::Box(ref mut b) => b,
        }
    }
}

enum EofItems<'root, 'tt> {
    None,
    One(MatcherPosHandle<'root, 'tt>),
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

/// Count how many metavars are named in the given matcher `ms`.
pub(super) fn count_names(ms: &[TokenTree]) -> usize {
    ms.iter().fold(0, |count, elt| {
        count
            + match *elt {
                TokenTree::Delimited(_, ref delim) => count_names(&delim.tts),
                TokenTree::MetaVar(..) => 0,
                TokenTree::MetaVarDecl(..) => 1,
                // Panicking here would abort execution because `parse_tree` makes use of this
                // function. In other words, RHS meta-variable expressions eventually end-up here.
                //
                // `0` is still returned to inform that no meta-variable was found. `Meta-variables
                // != Meta-variable expressions`
                TokenTree::MetaVarExpr(..) => 0,
                TokenTree::Sequence(_, ref seq) => seq.num_captures,
                TokenTree::Token(..) => 0,
            }
    })
}

/// `NamedMatch` is a pattern-match result for a single `token::MATCH_NONTERMINAL`:
/// so it is associated with a single ident in a parse, and all
/// `MatchedNonterminal`s in the `NamedMatch` have the same non-terminal type
/// (expr, item, etc). Each leaf in a single `NamedMatch` corresponds to a
/// single `token::MATCH_NONTERMINAL` in the `TokenTree` that produced it.
///
/// The in-memory structure of a particular `NamedMatch` represents the match
/// that occurred when a particular subset of a matcher was applied to a
/// particular token tree.
///
/// The width of each `MatchedSeq` in the `NamedMatch`, and the identity of
/// the `MatchedNonterminal`s, will depend on the token tree it was applied
/// to: each `MatchedSeq` corresponds to a single `TTSeq` in the originating
/// token tree. The depth of the `NamedMatch` structure will therefore depend
/// only on the nesting depth of `ast::TTSeq`s in the originating
/// token tree it was derived from.
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
    MatchedNonterminal(Lrc<Nonterminal>),
}

/// Takes a slice of token trees `ms` representing a matcher which successfully matched input
/// and an iterator of items that matched input and produces a `NamedParseResult`.
fn nameize<I: Iterator<Item = NamedMatch>>(
    sess: &ParseSess,
    ms: &[TokenTree],
    mut res: I,
) -> NamedParseResult {
    // Recursively descend into each type of matcher (e.g., sequences, delimited, metavars) and make
    // sure that each metavar has _exactly one_ binding. If a metavar does not have exactly one
    // binding, then there is an error. If it does, then we insert the binding into the
    // `NamedParseResult`.
    fn n_rec<I: Iterator<Item = NamedMatch>>(
        sess: &ParseSess,
        m: &TokenTree,
        res: &mut I,
        ret_val: &mut FxHashMap<MacroRulesNormalizedIdent, NamedMatch>,
    ) -> Result<(), (rustc_span::Span, String)> {
        match *m {
            TokenTree::Sequence(_, ref seq) => {
                for next_m in &seq.tts {
                    n_rec(sess, next_m, res.by_ref(), ret_val)?
                }
            }
            TokenTree::Delimited(_, ref delim) => {
                for next_m in &delim.tts {
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
    for m in ms {
        match n_rec(sess, m, res.by_ref(), &mut ret_val) {
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

/// Process the matcher positions of `cur_items` until it is empty. In the process, this will
/// produce more items in `next_items` and `bb_items`.
///
/// For more info about the how this happens, see the module-level doc comments and the inline
/// comments of this function.
///
/// # Parameters
///
/// - `cur_items`: the set of current items to be processed. This should be empty by the end of a
///   successful execution of this function.
/// - `next_items`: the set of newly generated items. These are used to replenish `cur_items` in
///   the function `parse`.
/// - `bb_items`: the set of items that are waiting for the black-box parser.
/// - `token`: the current token of the parser.
///
/// # Returns
///
/// `Some(result)` if everything is finished, `None` otherwise. Note that matches are kept track of
/// through the items generated.
fn parse_tt_inner<'root, 'tt>(
    sess: &ParseSess,
    ms: &[TokenTree],
    cur_items: &mut SmallVec<[MatcherPosHandle<'root, 'tt>; 1]>,
    next_items: &mut SmallVec<[MatcherPosHandle<'root, 'tt>; 1]>,
    bb_items: &mut SmallVec<[MatcherPosHandle<'root, 'tt>; 1]>,
    token: &Token,
) -> Option<NamedParseResult> {
    // Matcher positions that would be valid if the macro invocation was over now. Only modified if
    // `token == Eof`.
    let mut eof_items = EofItems::None;

    while let Some(mut item) = cur_items.pop() {
        // When unzipped trees end, remove them. This corresponds to backtracking out of a
        // delimited submatcher into which we already descended. When backtracking out again, we
        // need to advance the "dot" past the delimiters in the outer matcher.
        while item.idx >= item.top_elts.len() {
            match item.stack.pop() {
                Some(MatcherTtFrame { elts, idx }) => {
                    item.top_elts = elts;
                    item.idx = idx + 1;
                }
                None => break,
            }
        }

        // Get the current position of the "dot" (`idx`) in `item` and the number of token trees in
        // the matcher (`len`).
        let idx = item.idx;
        let len = item.top_elts.len();

        if idx < len {
            // We are in the middle of a matcher. Compare the matcher's current tt against `token`.
            match item.top_elts.get_tt(idx) {
                TokenTree::Sequence(sp, seq) => {
                    let op = seq.kleene.op;
                    if op == mbe::KleeneOp::ZeroOrMore || op == mbe::KleeneOp::ZeroOrOne {
                        // Allow for the possibility of zero matches of this sequence.
                        let mut new_item = item.clone();
                        new_item.match_cur += seq.num_captures;
                        new_item.idx += 1;
                        for idx in item.match_cur..item.match_cur + seq.num_captures {
                            new_item.push_match(idx, MatchedSeq(Lrc::new(smallvec![])));
                        }
                        cur_items.push(new_item);
                    }

                    // Allow for the possibility of one or more matches of this sequence.
                    cur_items.push(MatcherPosHandle::Box(Box::new(MatcherPos::repetition(
                        item, sp, seq,
                    ))));
                }

                TokenTree::MetaVarDecl(span, _, None) => {
                    // E.g. `$e` instead of `$e:expr`.
                    if sess.missing_fragment_specifiers.borrow_mut().remove(&span).is_some() {
                        return Some(Error(span, "missing fragment specifier".to_string()));
                    }
                }

                TokenTree::MetaVarDecl(_, _, Some(kind)) => {
                    // Built-in nonterminals never start with these tokens, so we can eliminate
                    // them from consideration.
                    //
                    // We use the span of the metavariable declaration to determine any
                    // edition-specific matching behavior for non-terminals.
                    if Parser::nonterminal_may_begin_with(kind, token) {
                        bb_items.push(item);
                    }
                }

                seq @ (TokenTree::Delimited(..)
                | TokenTree::Token(Token { kind: DocComment(..), .. })) => {
                    // To descend into a delimited submatcher or a doc comment, we push the current
                    // matcher onto a stack and push a new item containing the submatcher onto
                    // `cur_items`.
                    //
                    // At the beginning of the loop, if we reach the end of the delimited
                    // submatcher, we pop the stack to backtrack out of the descent.
                    let lower_elts = mem::replace(&mut item.top_elts, Tt(seq));
                    let idx = item.idx;
                    item.stack.push(MatcherTtFrame { elts: lower_elts, idx });
                    item.idx = 0;
                    cur_items.push(item);
                }

                TokenTree::Token(t) => {
                    // If the token matches, we can just advance the parser. Otherwise, this match
                    // hash failed, there is nothing to do, and hopefully another item in
                    // `cur_items` will match.
                    if token_name_eq(&t, token) {
                        item.idx += 1;
                        next_items.push(item);
                    }
                }

                // These cannot appear in a matcher.
                TokenTree::MetaVar(..) | TokenTree::MetaVarExpr(..) => unreachable!(),
            }
        } else if let Some(repetition) = &item.repetition {
            // We are past the end of a repetition.
            debug_assert!(idx <= len + 1);
            debug_assert!(matches!(item.top_elts, Tt(TokenTree::Sequence(..))));

            if idx == len {
                // Add all matches from the sequence to `up`, and move the "dot" past the
                // repetition in `up`. This allows for the case where the sequence matching is
                // finished.
                let mut new_pos = repetition.up.clone();
                for idx in item.match_lo..item.match_hi {
                    let sub = item.matches[idx].clone();
                    new_pos.push_match(idx, MatchedSeq(sub));
                }
                new_pos.match_cur = item.match_hi;
                new_pos.idx += 1;
                cur_items.push(new_pos);
            }

            if idx == len && repetition.sep.is_some() {
                if repetition.sep.as_ref().map_or(false, |sep| token_name_eq(token, sep)) {
                    // The matcher has a separator, and it matches the current token. We can
                    // advance past the separator token.
                    item.idx += 1;
                    next_items.push(item);
                }
            } else if repetition.seq_op != mbe::KleeneOp::ZeroOrOne {
                // We don't need a separator. Move the "dot" back to the beginning of the
                // matcher and try to match again UNLESS we are only allowed to have _one_
                // repetition.
                item.match_cur = item.match_lo;
                item.idx = 0;
                cur_items.push(item);
            }
        } else {
            // We are past the end of the matcher, and not in a repetition. Look for end of input.
            debug_assert_eq!(idx, len);
            if *token == token::Eof {
                eof_items = match eof_items {
                    EofItems::None => EofItems::One(item),
                    EofItems::One(_) | EofItems::Multiple => EofItems::Multiple,
                }
            }
        }
    }

    // If we reached the end of input, check that there is EXACTLY ONE possible matcher. Otherwise,
    // either the parse is ambiguous (which is an error) or there is a syntax error.
    if *token == token::Eof {
        Some(match eof_items {
            EofItems::One(mut eof_item) => {
                let matches =
                    eof_item.matches.iter_mut().map(|dv| Lrc::make_mut(dv).pop().unwrap());
                nameize(sess, ms, matches)
            }
            EofItems::Multiple => {
                Error(token.span, "ambiguity: multiple successful parses".to_string())
            }
            EofItems::None => Failure(
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

/// Use the given slice of token trees (`ms`) as a matcher. Match the token stream from the given
/// `parser` against it and return the match.
pub(super) fn parse_tt(
    parser: &mut Cow<'_, Parser<'_>>,
    ms: &[TokenTree],
    macro_name: Ident,
) -> NamedParseResult {
    // A queue of possible matcher positions. We initialize it with the matcher position in which
    // the "dot" is before the first token of the first token tree in `ms`. `parse_tt_inner` then
    // processes all of these possible matcher positions and produces possible next positions into
    // `next_items`. After some post-processing, the contents of `next_items` replenish `cur_items`
    // and we start over again.
    //
    // This MatcherPos instance is allocated on the stack. All others -- and there are frequently
    // *no* others! -- are allocated on the heap.
    let mut initial = MatcherPos::new(ms);
    let mut cur_items = smallvec![MatcherPosHandle::Ref(&mut initial)];

    loop {
        let mut next_items = SmallVec::new();

        // Matcher positions black-box parsed by `Parser`.
        let mut bb_items = SmallVec::new();

        // Process `cur_items` until either we have finished the input or we need to get some
        // parsing from the black-box parser done.
        if let Some(result) = parse_tt_inner(
            parser.sess,
            ms,
            &mut cur_items,
            &mut next_items,
            &mut bb_items,
            &parser.token,
        ) {
            return result;
        }

        // `parse_tt_inner` handled all cur_items, so it's empty.
        assert!(cur_items.is_empty());

        // Error messages here could be improved with links to original rules.
        match (next_items.len(), bb_items.len()) {
            (0, 0) => {
                // There are no possible next positions AND we aren't waiting for the black-box
                // parser: syntax error.
                return Failure(parser.token.clone(), "no rules expected this token in macro call");
            }

            (_, 0) => {
                // Dump all possible `next_items` into `cur_items` for the next iteration. Then
                // process the next token.
                cur_items.extend(next_items.drain(..));
                parser.to_mut().bump();
            }

            (0, 1) => {
                // We need to call the black-box parser to get some nonterminal.
                let mut item = bb_items.pop().unwrap();
                if let TokenTree::MetaVarDecl(span, _, Some(kind)) = item.top_elts.get_tt(item.idx)
                {
                    let match_cur = item.match_cur;
                    // We use the span of the metavariable declaration to determine any
                    // edition-specific matching behavior for non-terminals.
                    let nt = match parser.to_mut().parse_nonterminal(kind) {
                        Err(mut err) => {
                            err.span_label(
                                span,
                                format!("while parsing argument for this `{kind}` macro fragment"),
                            )
                            .emit();
                            return ErrorReported;
                        }
                        Ok(nt) => nt,
                    };
                    item.push_match(match_cur, MatchedNonterminal(Lrc::new(nt)));
                    item.idx += 1;
                    item.match_cur += 1;
                } else {
                    unreachable!()
                }
                cur_items.push(item);
            }

            (_, _) => {
                // Too many possibilities!
                return bb_items_ambiguity_error(
                    macro_name,
                    next_items,
                    bb_items,
                    parser.token.span,
                );
            }
        }

        assert!(!cur_items.is_empty());
    }
}

fn bb_items_ambiguity_error<'root, 'tt>(
    macro_name: Ident,
    next_items: SmallVec<[MatcherPosHandle<'root, 'tt>; 1]>,
    bb_items: SmallVec<[MatcherPosHandle<'root, 'tt>; 1]>,
    token_span: rustc_span::Span,
) -> NamedParseResult {
    let nts = bb_items
        .iter()
        .map(|item| match item.top_elts.get_tt(item.idx) {
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
            "local ambiguity when calling macro `{macro_name}`: multiple parsing options: {}",
            match next_items.len() {
                0 => format!("built-in NTs {}.", nts),
                1 => format!("built-in NTs {} or 1 other option.", nts),
                n => format!("built-in NTs {} or {} other options.", nts, n),
            }
        ),
    )
}
