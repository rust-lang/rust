//! This module takes a (parsed) definition of `macro_rules` invocation, a
//! `tt::TokenTree` representing an argument of macro invocation, and produces a
//! `tt::TokenTree` for the result of the expansion.

mod matcher;
mod transcriber;

use intern::Symbol;
use rustc_hash::FxHashMap;
use span::{Edition, Span};

use crate::{ExpandError, ExpandErrorKind, ExpandResult, MatchedArmIndex, parser::MetaVarKind};

pub(crate) fn expand_rules(
    rules: &[crate::Rule],
    input: &tt::TopSubtree<Span>,
    marker: impl Fn(&mut Span) + Copy,
    call_site: Span,
    def_site_edition: Edition,
) -> ExpandResult<(tt::TopSubtree<Span>, MatchedArmIndex)> {
    let mut match_: Option<(matcher::Match<'_>, &crate::Rule, usize)> = None;
    for (idx, rule) in rules.iter().enumerate() {
        let new_match = matcher::match_(&rule.lhs, input, def_site_edition);

        if new_match.err.is_none() {
            // If we find a rule that applies without errors, we're done.
            // Unconditionally returning the transcription here makes the
            // `test_repeat_bad_var` test fail.
            let ExpandResult { value, err: transcribe_err } =
                transcriber::transcribe(&rule.rhs, &new_match.bindings, marker, call_site);
            if transcribe_err.is_none() {
                return ExpandResult::ok((value, Some(idx as u32)));
            }
        }
        // Use the rule if we matched more tokens, or bound variables count
        if let Some((prev_match, _, _)) = &match_ {
            if (new_match.unmatched_tts, -(new_match.bound_count as i32))
                < (prev_match.unmatched_tts, -(prev_match.bound_count as i32))
            {
                match_ = Some((new_match, rule, idx));
            }
        } else {
            match_ = Some((new_match, rule, idx));
        }
    }
    if let Some((match_, rule, idx)) = match_ {
        // if we got here, there was no match without errors
        let ExpandResult { value, err: transcribe_err } =
            transcriber::transcribe(&rule.rhs, &match_.bindings, marker, call_site);
        ExpandResult { value: (value, idx.try_into().ok()), err: match_.err.or(transcribe_err) }
    } else {
        ExpandResult::new(
            (tt::TopSubtree::empty(tt::DelimSpan::from_single(call_site)), None),
            ExpandError::new(call_site, ExpandErrorKind::NoMatchingRule),
        )
    }
}

/// The actual algorithm for expansion is not too hard, but is pretty tricky.
/// `Bindings` structure is the key to understanding what we are doing here.
///
/// On the high level, it stores mapping from meta variables to the bits of
/// syntax it should be substituted with. For example, if `$e:expr` is matched
/// with `1 + 1` by macro_rules, the `Binding` will store `$e -> 1 + 1`.
///
/// The tricky bit is dealing with repetitions (`$()*`). Consider this example:
///
/// ```not_rust
/// macro_rules! foo {
///     ($($ i:ident $($ e:expr),*);*) => {
///         $(fn $ i() { $($ e);*; })*
///     }
/// }
/// foo! { foo 1,2,3; bar 4,5,6 }
/// ```
///
/// Here, the `$i` meta variable is matched first with `foo` and then with
/// `bar`, and `$e` is matched in turn with `1`, `2`, `3`, `4`, `5`, `6`.
///
/// To represent such "multi-mappings", we use a recursive structures: we map
/// variables not to values, but to *lists* of values or other lists (that is,
/// to the trees).
///
/// For the above example, the bindings would store
///
/// ```not_rust
/// i -> [foo, bar]
/// e -> [[1, 2, 3], [4, 5, 6]]
/// ```
///
/// We construct `Bindings` in the `match_lhs`. The interesting case is
/// `TokenTree::Repeat`, where we use `push_nested` to create the desired
/// nesting structure.
///
/// The other side of the puzzle is `expand_subtree`, where we use the bindings
/// to substitute meta variables in the output template. When expanding, we
/// maintain a `nesting` stack of indices which tells us which occurrence from
/// the `Bindings` we should take. We push to the stack when we enter a
/// repetition.
///
/// In other words, `Bindings` is a *multi* mapping from `Symbol` to
/// `tt::TokenTree`, where the index to select a particular `TokenTree` among
/// many is not a plain `usize`, but a `&[usize]`.
#[derive(Debug, Default, Clone)]
struct Bindings<'a> {
    inner: FxHashMap<Symbol, Binding<'a>>,
}

#[derive(Debug, Clone)]
enum Binding<'a> {
    Fragment(Fragment<'a>),
    Nested(Vec<Binding<'a>>),
    Empty,
    Missing(MetaVarKind),
}

#[derive(Debug, Default, Clone)]
enum Fragment<'a> {
    #[default]
    Empty,
    /// token fragments are just copy-pasted into the output
    Tokens(tt::TokenTreesView<'a, Span>),
    /// Expr ast fragments are surrounded with `()` on transcription to preserve precedence.
    /// Note that this impl is different from the one currently in `rustc` --
    /// `rustc` doesn't translate fragments into token trees at all.
    ///
    /// At one point in time, we tried to use "fake" delimiters here à la
    /// proc-macro delimiter=none. As we later discovered, "none" delimiters are
    /// tricky to handle in the parser, and rustc doesn't handle those either.
    ///
    /// The span of the outer delimiters is marked on transcription.
    Expr(tt::TokenTreesView<'a, Span>),
    /// There are roughly two types of paths: paths in expression context, where a
    /// separator `::` between an identifier and its following generic argument list
    /// is mandatory, and paths in type context, where `::` can be omitted.
    ///
    /// Unlike rustc, we need to transform the parsed fragments back into tokens
    /// during transcription. When the matched path fragment is a type-context path
    /// and is trasncribed as an expression-context path, verbatim transcription
    /// would cause a syntax error. We need to fix it up just before transcribing;
    /// see `transcriber::fix_up_and_push_path_tt()`.
    Path(tt::TokenTreesView<'a, Span>),
    TokensOwned(tt::TopSubtree<Span>),
}

impl Fragment<'_> {
    fn is_empty(&self) -> bool {
        match self {
            Fragment::Empty => true,
            Fragment::Tokens(it) => it.len() == 0,
            Fragment::Expr(it) => it.len() == 0,
            Fragment::Path(it) => it.len() == 0,
            Fragment::TokensOwned(it) => it.0.is_empty(),
        }
    }
}
