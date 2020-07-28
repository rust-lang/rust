//! Ungrammar -- a DSL for specifying concrete syntax tree grammar.
//!
//! Producing a parser is an explicit non-goal -- it's ok for this grammar to be
//! ambiguous, non LL, non LR, etc.
mod error;
mod lexer;
mod parser;

use std::{ops, str::FromStr};

pub use error::{Error, Result};

pub fn rust_grammar() -> Grammar {
    let src = include_str!("../rust.ungram");
    src.parse().unwrap()
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct Node(usize);
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct Token(usize);

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
    pub fn iter(&self) -> impl Iterator<Item = Node> + '_ {
        (0..self.nodes.len()).map(Node)
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

#[derive(Debug)]
pub struct NodeData {
    pub name: String,
    pub rule: Rule,
}

#[derive(Debug)]
pub struct TokenData {
    pub name: String,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Rule {
    Labeled { label: String, rule: Box<Rule> },
    Node(Node),
    Token(Token),
    Seq(Vec<Rule>),
    Alt(Vec<Rule>),
    Opt(Box<Rule>),
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
