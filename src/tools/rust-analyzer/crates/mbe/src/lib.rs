//! `mbe` (short for Macro By Example) crate contains code for handling
//! `macro_rules` macros. It uses `TokenTree` (from `tt` package) as the
//! interface, although it contains some code to bridge `SyntaxNode`s and
//! `TokenTree`s as well!
//!
//! The tests for this functionality live in another crate:
//! `hir_def::macro_expansion_tests::mbe`.

#![warn(rust_2018_idioms, unused_lifetimes)]

mod expander;
mod parser;
mod syntax_bridge;
mod to_parser_input;
mod tt_iter;

#[cfg(test)]
mod benchmark;

use stdx::impl_from;
use tt::Span;

use std::fmt;

use crate::{
    parser::{MetaTemplate, MetaVarKind, Op},
    tt_iter::TtIter,
};

// FIXME: we probably should re-think  `token_tree_to_syntax_node` interfaces
pub use ::parser::TopEntryPoint;
pub use tt::{Delimiter, DelimiterKind, Punct};

pub use crate::syntax_bridge::{
    parse_exprs_with_sep, parse_to_token_tree, parse_to_token_tree_static_span,
    syntax_node_to_token_tree, syntax_node_to_token_tree_modified, token_tree_to_syntax_node,
    SpanMapper,
};

pub use crate::syntax_bridge::dummy_test_span_utils::*;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ParseError {
    UnexpectedToken(Box<str>),
    Expected(Box<str>),
    InvalidRepeat,
    RepetitionEmptyTokenTree,
}

impl ParseError {
    fn expected(e: &str) -> ParseError {
        ParseError::Expected(e.into())
    }

    fn unexpected(e: &str) -> ParseError {
        ParseError::UnexpectedToken(e.into())
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken(it) => f.write_str(it),
            ParseError::Expected(it) => f.write_str(it),
            ParseError::InvalidRepeat => f.write_str("invalid repeat"),
            ParseError::RepetitionEmptyTokenTree => f.write_str("empty token tree in repetition"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ExpandError {
    BindingError(Box<Box<str>>),
    UnresolvedBinding(Box<Box<str>>),
    LeftoverTokens,
    ConversionError,
    LimitExceeded,
    NoMatchingRule,
    UnexpectedToken,
    CountError(CountError),
}

impl_from!(CountError for ExpandError);

impl ExpandError {
    fn binding_error(e: impl Into<Box<str>>) -> ExpandError {
        ExpandError::BindingError(Box::new(e.into()))
    }
}

impl fmt::Display for ExpandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpandError::NoMatchingRule => f.write_str("no rule matches input tokens"),
            ExpandError::UnexpectedToken => f.write_str("unexpected token in input"),
            ExpandError::BindingError(e) => f.write_str(e),
            ExpandError::UnresolvedBinding(binding) => {
                f.write_str("could not find binding ")?;
                f.write_str(binding)
            }
            ExpandError::ConversionError => f.write_str("could not convert tokens"),
            ExpandError::LimitExceeded => f.write_str("Expand exceed limit"),
            ExpandError::LeftoverTokens => f.write_str("leftover tokens"),
            ExpandError::CountError(e) => e.fmt(f),
        }
    }
}

// FIXME: Showing these errors could be nicer.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum CountError {
    OutOfBounds,
    Misplaced,
}

impl fmt::Display for CountError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CountError::OutOfBounds => f.write_str("${count} out of bounds"),
            CountError::Misplaced => f.write_str("${count} misplaced"),
        }
    }
}

/// This struct contains AST for a single `macro_rules` definition. What might
/// be very confusing is that AST has almost exactly the same shape as
/// `tt::TokenTree`, but there's a crucial difference: in macro rules, `$ident`
/// and `$()*` have special meaning (see `Var` and `Repeat` data structures)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclarativeMacro<S> {
    rules: Box<[Rule<S>]>,
    // This is used for correctly determining the behavior of the pat fragment
    // FIXME: This should be tracked by hygiene of the fragment identifier!
    is_2021: bool,
    err: Option<Box<ParseError>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Rule<S> {
    lhs: MetaTemplate<S>,
    rhs: MetaTemplate<S>,
}

impl<S: Span> DeclarativeMacro<S> {
    pub fn from_err(err: ParseError, is_2021: bool) -> DeclarativeMacro<S> {
        DeclarativeMacro { rules: Box::default(), is_2021, err: Some(Box::new(err)) }
    }

    /// The old, `macro_rules! m {}` flavor.
    pub fn parse_macro_rules(
        tt: &tt::Subtree<S>,
        is_2021: bool,
        // FIXME: Remove this once we drop support for rust 1.76 (defaults to true then)
        new_meta_vars: bool,
    ) -> DeclarativeMacro<S> {
        // Note: this parsing can be implemented using mbe machinery itself, by
        // matching against `$($lhs:tt => $rhs:tt);*` pattern, but implementing
        // manually seems easier.
        let mut src = TtIter::new(tt);
        let mut rules = Vec::new();
        let mut err = None;

        while src.len() > 0 {
            let rule = match Rule::parse(&mut src, true, new_meta_vars) {
                Ok(it) => it,
                Err(e) => {
                    err = Some(Box::new(e));
                    break;
                }
            };
            rules.push(rule);
            if let Err(()) = src.expect_char(';') {
                if src.len() > 0 {
                    err = Some(Box::new(ParseError::expected("expected `;`")));
                }
                break;
            }
        }

        for Rule { lhs, .. } in &rules {
            if let Err(e) = validate(lhs) {
                err = Some(Box::new(e));
                break;
            }
        }

        DeclarativeMacro { rules: rules.into_boxed_slice(), is_2021, err }
    }

    /// The new, unstable `macro m {}` flavor.
    pub fn parse_macro2(
        tt: &tt::Subtree<S>,
        is_2021: bool,
        // FIXME: Remove this once we drop support for rust 1.76 (defaults to true then)
        new_meta_vars: bool,
    ) -> DeclarativeMacro<S> {
        let mut src = TtIter::new(tt);
        let mut rules = Vec::new();
        let mut err = None;

        if tt::DelimiterKind::Brace == tt.delimiter.kind {
            cov_mark::hit!(parse_macro_def_rules);
            while src.len() > 0 {
                let rule = match Rule::parse(&mut src, true, new_meta_vars) {
                    Ok(it) => it,
                    Err(e) => {
                        err = Some(Box::new(e));
                        break;
                    }
                };
                rules.push(rule);
                if let Err(()) = src.expect_any_char(&[';', ',']) {
                    if src.len() > 0 {
                        err = Some(Box::new(ParseError::expected(
                            "expected `;` or `,` to delimit rules",
                        )));
                    }
                    break;
                }
            }
        } else {
            cov_mark::hit!(parse_macro_def_simple);
            match Rule::parse(&mut src, false, new_meta_vars) {
                Ok(rule) => {
                    if src.len() != 0 {
                        err = Some(Box::new(ParseError::expected("remaining tokens in macro def")));
                    }
                    rules.push(rule);
                }
                Err(e) => {
                    err = Some(Box::new(e));
                }
            }
        }

        for Rule { lhs, .. } in &rules {
            if let Err(e) = validate(lhs) {
                err = Some(Box::new(e));
                break;
            }
        }

        DeclarativeMacro { rules: rules.into_boxed_slice(), is_2021, err }
    }

    pub fn err(&self) -> Option<&ParseError> {
        self.err.as_deref()
    }

    pub fn expand(
        &self,
        tt: &tt::Subtree<S>,
        marker: impl Fn(&mut S) + Copy,
        new_meta_vars: bool,
        call_site: S,
    ) -> ExpandResult<tt::Subtree<S>> {
        expander::expand_rules(&self.rules, tt, marker, self.is_2021, new_meta_vars, call_site)
    }
}

impl<S: Span> Rule<S> {
    fn parse(
        src: &mut TtIter<'_, S>,
        expect_arrow: bool,
        new_meta_vars: bool,
    ) -> Result<Self, ParseError> {
        let lhs = src.expect_subtree().map_err(|()| ParseError::expected("expected subtree"))?;
        if expect_arrow {
            src.expect_char('=').map_err(|()| ParseError::expected("expected `=`"))?;
            src.expect_char('>').map_err(|()| ParseError::expected("expected `>`"))?;
        }
        let rhs = src.expect_subtree().map_err(|()| ParseError::expected("expected subtree"))?;

        let lhs = MetaTemplate::parse_pattern(lhs)?;
        let rhs = MetaTemplate::parse_template(rhs, new_meta_vars)?;

        Ok(crate::Rule { lhs, rhs })
    }
}

fn validate<S: Span>(pattern: &MetaTemplate<S>) -> Result<(), ParseError> {
    for op in pattern.iter() {
        match op {
            Op::Subtree { tokens, .. } => validate(tokens)?,
            Op::Repeat { tokens: subtree, separator, .. } => {
                // Checks that no repetition which could match an empty token
                // https://github.com/rust-lang/rust/blob/a58b1ed44f5e06976de2bdc4d7dc81c36a96934f/src/librustc_expand/mbe/macro_rules.rs#L558
                let lsh_is_empty_seq = separator.is_none() && subtree.iter().all(|child_op| {
                    match *child_op {
                        // vis is optional
                        Op::Var { kind: Some(kind), .. } => kind == MetaVarKind::Vis,
                        Op::Repeat {
                            kind: parser::RepeatKind::ZeroOrMore | parser::RepeatKind::ZeroOrOne,
                            ..
                        } => true,
                        _ => false,
                    }
                });
                if lsh_is_empty_seq {
                    return Err(ParseError::RepetitionEmptyTokenTree);
                }
                validate(subtree)?
            }
            _ => (),
        }
    }
    Ok(())
}

pub type ExpandResult<T> = ValueResult<T, ExpandError>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ValueResult<T, E> {
    pub value: T,
    pub err: Option<E>,
}

impl<T, E> ValueResult<T, E> {
    pub fn new(value: T, err: E) -> Self {
        Self { value, err: Some(err) }
    }

    pub fn ok(value: T) -> Self {
        Self { value, err: None }
    }

    pub fn only_err(err: E) -> Self
    where
        T: Default,
    {
        Self { value: Default::default(), err: Some(err) }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> ValueResult<U, E> {
        ValueResult { value: f(self.value), err: self.err }
    }

    pub fn map_err<E2>(self, f: impl FnOnce(E) -> E2) -> ValueResult<T, E2> {
        ValueResult { value: self.value, err: self.err.map(f) }
    }

    pub fn result(self) -> Result<T, E> {
        self.err.map_or(Ok(self.value), Err)
    }
}

impl<T: Default, E> From<Result<T, E>> for ValueResult<T, E> {
    fn from(result: Result<T, E>) -> Self {
        result.map_or_else(Self::only_err, Self::ok)
    }
}
