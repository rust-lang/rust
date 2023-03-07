//! `mbe` (short for Macro By Example) crate contains code for handling
//! `macro_rules` macros. It uses `TokenTree` (from `tt` package) as the
//! interface, although it contains some code to bridge `SyntaxNode`s and
//! `TokenTree`s as well!
//!
//! The tes for this functionality live in another crate:
//! `hir_def::macro_expansion_tests::mbe`.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod parser;
mod expander;
mod syntax_bridge;
mod tt_iter;
mod to_parser_input;

#[cfg(test)]
mod benchmark;
mod token_map;

use ::tt::token_id as tt;

use std::fmt;

use crate::{
    parser::{MetaTemplate, MetaVarKind, Op},
    tt_iter::TtIter,
};

// FIXME: we probably should re-think  `token_tree_to_syntax_node` interfaces
pub use self::tt::{Delimiter, DelimiterKind, Punct};
pub use ::parser::TopEntryPoint;

pub use crate::{
    syntax_bridge::{
        parse_exprs_with_sep, parse_to_token_tree, syntax_node_to_token_tree,
        syntax_node_to_token_tree_with_modifications, token_tree_to_syntax_node, SyntheticToken,
        SyntheticTokenId,
    },
    token_map::TokenMap,
};

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

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ExpandError {
    BindingError(Box<Box<str>>),
    LeftoverTokens,
    ConversionError,
    LimitExceeded,
    NoMatchingRule,
    UnexpectedToken,
}

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
            ExpandError::ConversionError => f.write_str("could not convert tokens"),
            ExpandError::LimitExceeded => f.write_str("Expand exceed limit"),
            ExpandError::LeftoverTokens => f.write_str("leftover tokens"),
        }
    }
}

/// This struct contains AST for a single `macro_rules` definition. What might
/// be very confusing is that AST has almost exactly the same shape as
/// `tt::TokenTree`, but there's a crucial difference: in macro rules, `$ident`
/// and `$()*` have special meaning (see `Var` and `Repeat` data structures)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclarativeMacro {
    rules: Vec<Rule>,
    /// Highest id of the token we have in TokenMap
    shift: Shift,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Rule {
    lhs: MetaTemplate,
    rhs: MetaTemplate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shift(u32);

impl Shift {
    pub fn new(tt: &tt::Subtree) -> Shift {
        // Note that TokenId is started from zero,
        // We have to add 1 to prevent duplication.
        let value = max_id(tt).map_or(0, |it| it + 1);
        return Shift(value);

        // Find the max token id inside a subtree
        fn max_id(subtree: &tt::Subtree) -> Option<u32> {
            let filter =
                |tt: &_| match tt {
                    tt::TokenTree::Subtree(subtree) => {
                        let tree_id = max_id(subtree);
                        if subtree.delimiter.open != tt::TokenId::unspecified() {
                            Some(tree_id.map_or(subtree.delimiter.open.0, |t| {
                                t.max(subtree.delimiter.open.0)
                            }))
                        } else {
                            tree_id
                        }
                    }
                    tt::TokenTree::Leaf(leaf) => {
                        let &(tt::Leaf::Ident(tt::Ident { span, .. })
                        | tt::Leaf::Punct(tt::Punct { span, .. })
                        | tt::Leaf::Literal(tt::Literal { span, .. })) = leaf;

                        (span != tt::TokenId::unspecified()).then_some(span.0)
                    }
                };
            subtree.token_trees.iter().filter_map(filter).max()
        }
    }

    /// Shift given TokenTree token id
    pub fn shift_all(self, tt: &mut tt::Subtree) {
        for t in &mut tt.token_trees {
            match t {
                tt::TokenTree::Leaf(
                    tt::Leaf::Ident(tt::Ident { span, .. })
                    | tt::Leaf::Punct(tt::Punct { span, .. })
                    | tt::Leaf::Literal(tt::Literal { span, .. }),
                ) => *span = self.shift(*span),
                tt::TokenTree::Subtree(tt) => {
                    tt.delimiter.open = self.shift(tt.delimiter.open);
                    tt.delimiter.close = self.shift(tt.delimiter.close);
                    self.shift_all(tt)
                }
            }
        }
    }

    pub fn shift(self, id: tt::TokenId) -> tt::TokenId {
        if id == tt::TokenId::unspecified() {
            id
        } else {
            tt::TokenId(id.0 + self.0)
        }
    }

    pub fn unshift(self, id: tt::TokenId) -> Option<tt::TokenId> {
        id.0.checked_sub(self.0).map(tt::TokenId)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Origin {
    Def,
    Call,
}

impl DeclarativeMacro {
    /// The old, `macro_rules! m {}` flavor.
    pub fn parse_macro_rules(tt: &tt::Subtree) -> Result<DeclarativeMacro, ParseError> {
        // Note: this parsing can be implemented using mbe machinery itself, by
        // matching against `$($lhs:tt => $rhs:tt);*` pattern, but implementing
        // manually seems easier.
        let mut src = TtIter::new(tt);
        let mut rules = Vec::new();
        while src.len() > 0 {
            let rule = Rule::parse(&mut src, true)?;
            rules.push(rule);
            if let Err(()) = src.expect_char(';') {
                if src.len() > 0 {
                    return Err(ParseError::expected("expected `;`"));
                }
                break;
            }
        }

        for Rule { lhs, .. } in &rules {
            validate(lhs)?;
        }

        Ok(DeclarativeMacro { rules, shift: Shift::new(tt) })
    }

    /// The new, unstable `macro m {}` flavor.
    pub fn parse_macro2(tt: &tt::Subtree) -> Result<DeclarativeMacro, ParseError> {
        let mut src = TtIter::new(tt);
        let mut rules = Vec::new();

        if tt::DelimiterKind::Brace == tt.delimiter.kind {
            cov_mark::hit!(parse_macro_def_rules);
            while src.len() > 0 {
                let rule = Rule::parse(&mut src, true)?;
                rules.push(rule);
                if let Err(()) = src.expect_any_char(&[';', ',']) {
                    if src.len() > 0 {
                        return Err(ParseError::expected("expected `;` or `,` to delimit rules"));
                    }
                    break;
                }
            }
        } else {
            cov_mark::hit!(parse_macro_def_simple);
            let rule = Rule::parse(&mut src, false)?;
            if src.len() != 0 {
                return Err(ParseError::expected("remaining tokens in macro def"));
            }
            rules.push(rule);
        }

        for Rule { lhs, .. } in &rules {
            validate(lhs)?;
        }

        Ok(DeclarativeMacro { rules, shift: Shift::new(tt) })
    }

    pub fn expand(&self, tt: &tt::Subtree) -> ExpandResult<tt::Subtree> {
        // apply shift
        let mut tt = tt.clone();
        self.shift.shift_all(&mut tt);
        expander::expand_rules(&self.rules, &tt)
    }

    pub fn map_id_down(&self, id: tt::TokenId) -> tt::TokenId {
        self.shift.shift(id)
    }

    pub fn map_id_up(&self, id: tt::TokenId) -> (tt::TokenId, Origin) {
        match self.shift.unshift(id) {
            Some(id) => (id, Origin::Call),
            None => (id, Origin::Def),
        }
    }

    pub fn shift(&self) -> Shift {
        self.shift
    }
}

impl Rule {
    fn parse(src: &mut TtIter<'_>, expect_arrow: bool) -> Result<Self, ParseError> {
        let lhs = src.expect_subtree().map_err(|()| ParseError::expected("expected subtree"))?;
        if expect_arrow {
            src.expect_char('=').map_err(|()| ParseError::expected("expected `=`"))?;
            src.expect_char('>').map_err(|()| ParseError::expected("expected `>`"))?;
        }
        let rhs = src.expect_subtree().map_err(|()| ParseError::expected("expected subtree"))?;

        let lhs = MetaTemplate::parse_pattern(lhs)?;
        let rhs = MetaTemplate::parse_template(rhs)?;

        Ok(crate::Rule { lhs, rhs })
    }
}

fn validate(pattern: &MetaTemplate) -> Result<(), ParseError> {
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
    pub fn ok(value: T) -> Self {
        Self { value, err: None }
    }

    pub fn with_err(value: T, err: E) -> Self {
        Self { value, err: Some(err) }
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
