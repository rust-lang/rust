//! Ungrammar -- a DSL for specifying concrete syntax tree grammar.
//!
//! Producing a parser is an explicit non-goal -- it's ok for this grammar to be
//! ambiguous, non LL, non LR, etc.
//!
//! See this
//! [introductory post](https://rust-analyzer.github.io/blog/2020/10/24/introducing-ungrammar.html)
//! for details.

#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(rust_2018_idioms)]

mod error;
mod lexer;
mod parser;

use std::{ops, str::FromStr};

pub use error::{Error, Result};

/// Returns a Rust grammar.
pub fn rust_grammar() -> Grammar {
    let src = include_str!("../rust.ungram");
    src.parse().unwrap()
}

/// A node, like `A = 'b' | 'c'`.
///
/// Indexing into a [`Grammar`] with a [`Node`] returns a reference to a
/// [`NodeData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node(usize);

/// A token, denoted with single quotes, like `'+'` or `'struct'`.
///
/// Indexing into a [`Grammar`] with a [`Token`] returns a reference to a
/// [`TokenData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token(usize);

/// An Ungrammar grammar.
#[derive(Default, Debug)]
pub struct Grammar {
    nodes: Vec<NodeData>,
    tokens: Vec<TokenData>,
}

impl FromStr for Grammar {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let tokens = lexer::tokenize(s)?;
        parser::parse(tokens)
    }
}

impl Grammar {
    /// Returns an iterator over all nodes in the grammar.
    pub fn iter(&self) -> impl Iterator<Item = Node> + '_ {
        (0..self.nodes.len()).map(Node)
    }

    /// Returns an iterator over all tokens in the grammar.
    pub fn tokens(&self) -> impl Iterator<Item = Token> + '_ {
        (0..self.tokens.len()).map(Token)
    }
}

impl ops::Index<Node> for Grammar {
    type Output = NodeData;
    fn index(&self, Node(index): Node) -> &NodeData {
        &self.nodes[index]
    }
}

impl ops::Index<Token> for Grammar {
    type Output = TokenData;
    fn index(&self, Token(index): Token) -> &TokenData {
        &self.tokens[index]
    }
}

/// Data about a node.
#[derive(Debug)]
pub struct NodeData {
    /// The name of the node.
    ///
    /// In the rule `A = 'b' | 'c'`, this is `"A"`.
    pub name: String,
    /// The rule for this node.
    ///
    /// In the rule `A = 'b' | 'c'`, this represents `'b' | 'c'`.
    pub rule: Rule,
}

/// Data about a token.
#[derive(Debug)]
pub struct TokenData {
    /// The name of the token.
    pub name: String,
}

/// A production rule.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Rule {
    /// A labeled rule, like `a:B` (`"a"` is the label, `B` is the rule).
    Labeled {
        /// The label.
        label: String,
        /// The rule.
        rule: Box<Rule>,
    },
    /// A node, like `A`.
    Node(Node),
    /// A token, like `'struct'`.
    Token(Token),
    /// A sequence of rules, like `'while' '(' Expr ')' Stmt`.
    Seq(Vec<Rule>),
    /// An alternative between many rules, like `'+' | '-' | '*' | '/'`.
    Alt(Vec<Rule>),
    /// An optional rule, like `A?`.
    Opt(Box<Rule>),
    /// A repeated rule, like `A*`.
    Rep(Box<Rule>),
}

#[test]
fn smoke() {
    let grammar = include_str!("../ungrammar.ungram");
    let grammar = grammar.parse::<Grammar>().unwrap();
    drop(grammar)
}

#[test]
fn test_rust_grammar() {
    let _ = rust_grammar();
}
