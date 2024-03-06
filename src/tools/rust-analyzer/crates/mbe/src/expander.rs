//! This module takes a (parsed) definition of `macro_rules` invocation, a
//! `tt::TokenTree` representing an argument of macro invocation, and produces a
//! `tt::TokenTree` for the result of the expansion.

mod matcher;
mod transcriber;

use rustc_hash::FxHashMap;
use syntax::SmolStr;
use tt::Span;

use crate::{parser::MetaVarKind, ExpandError, ExpandResult};

pub(crate) fn expand_rules<S: Span>(
    rules: &[crate::Rule<S>],
    input: &tt::Subtree<S>,
    marker: impl Fn(&mut S) + Copy,
    is_2021: bool,
    new_meta_vars: bool,
    call_site: S,
) -> ExpandResult<tt::Subtree<S>> {
    let mut match_: Option<(matcher::Match<S>, &crate::Rule<S>)> = None;
    for rule in rules {
        let new_match = matcher::match_(&rule.lhs, input, is_2021);

        if new_match.err.is_none() {
            // If we find a rule that applies without errors, we're done.
            // Unconditionally returning the transcription here makes the
            // `test_repeat_bad_var` test fail.
            let ExpandResult { value, err: transcribe_err } = transcriber::transcribe(
                &rule.rhs,
                &new_match.bindings,
                marker,
                new_meta_vars,
                call_site,
            );
            if transcribe_err.is_none() {
                return ExpandResult::ok(value);
            }
        }
        // Use the rule if we matched more tokens, or bound variables count
        if let Some((prev_match, _)) = &match_ {
            if (new_match.unmatched_tts, -(new_match.bound_count as i32))
                < (prev_match.unmatched_tts, -(prev_match.bound_count as i32))
            {
                match_ = Some((new_match, rule));
            }
        } else {
            match_ = Some((new_match, rule));
        }
    }
    if let Some((match_, rule)) = match_ {
        // if we got here, there was no match without errors
        let ExpandResult { value, err: transcribe_err } =
            transcriber::transcribe(&rule.rhs, &match_.bindings, marker, new_meta_vars, call_site);
        ExpandResult { value, err: match_.err.or(transcribe_err) }
    } else {
        ExpandResult::new(
            tt::Subtree {
                delimiter: tt::Delimiter::invisible_spanned(call_site),
                token_trees: Box::new([]),
            },
            ExpandError::NoMatchingRule,
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
/// In other words, `Bindings` is a *multi* mapping from `SmolStr` to
/// `tt::TokenTree`, where the index to select a particular `TokenTree` among
/// many is not a plain `usize`, but a `&[usize]`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Bindings<S> {
    inner: FxHashMap<SmolStr, Binding<S>>,
}

impl<S> Default for Bindings<S> {
    fn default() -> Self {
        Self { inner: Default::default() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Binding<S> {
    Fragment(Fragment<S>),
    Nested(Vec<Binding<S>>),
    Empty,
    Missing(MetaVarKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Fragment<S> {
    Empty,
    /// token fragments are just copy-pasted into the output
    Tokens(tt::TokenTree<S>),
    /// Expr ast fragments are surrounded with `()` on insertion to preserve
    /// precedence. Note that this impl is different from the one currently in
    /// `rustc` -- `rustc` doesn't translate fragments into token trees at all.
    ///
    /// At one point in time, we tried to use "fake" delimiters here à la
    /// proc-macro delimiter=none. As we later discovered, "none" delimiters are
    /// tricky to handle in the parser, and rustc doesn't handle those either.
    Expr(tt::Subtree<S>),
    /// There are roughly two types of paths: paths in expression context, where a
    /// separator `::` between an identifier and its following generic argument list
    /// is mandatory, and paths in type context, where `::` can be omitted.
    ///
    /// Unlike rustc, we need to transform the parsed fragments back into tokens
    /// during transcription. When the matched path fragment is a type-context path
    /// and is trasncribed as an expression-context path, verbatim transcription
    /// would cause a syntax error. We need to fix it up just before transcribing;
    /// see `transcriber::fix_up_and_push_path_tt()`.
    Path(tt::Subtree<S>),
}
