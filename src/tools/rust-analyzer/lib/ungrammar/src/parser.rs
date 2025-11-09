//! Simple hand-written ungrammar parser.
#![allow(clippy::disallowed_types)]
use std::collections::HashMap;

use crate::{
    Grammar, Node, NodeData, Rule, Token, TokenData,
    error::{Result, bail, format_err},
    lexer::{self, TokenKind},
};

macro_rules! bail {
    ($loc:expr, $($tt:tt)*) => {{
        let err = $crate::error::format_err!($($tt)*)
            .with_location($loc);
        return Err(err);
    }};
}

pub(crate) fn parse(tokens: Vec<lexer::Token>) -> Result<Grammar> {
    let mut p = Parser::new(tokens);
    while !p.is_eof() {
        node(&mut p)?;
    }
    p.finish()
}

#[derive(Default)]
struct Parser {
    grammar: Grammar,
    tokens: Vec<lexer::Token>,
    node_table: HashMap<String, Node>,
    token_table: HashMap<String, Token>,
}

const DUMMY_RULE: Rule = Rule::Node(Node(!0));

impl Parser {
    fn new(mut tokens: Vec<lexer::Token>) -> Parser {
        tokens.reverse();
        Parser { tokens, ..Parser::default() }
    }

    fn peek(&self) -> Option<&lexer::Token> {
        self.peek_n(0)
    }
    fn peek_n(&self, n: usize) -> Option<&lexer::Token> {
        self.tokens.iter().nth_back(n)
    }
    fn bump(&mut self) -> Result<lexer::Token> {
        self.tokens.pop().ok_or_else(|| format_err!("unexpected EOF"))
    }
    fn expect(&mut self, kind: TokenKind, what: &str) -> Result<()> {
        let token = self.bump()?;
        if token.kind != kind {
            bail!(token.loc, "unexpected token, expected `{}`", what);
        }
        Ok(())
    }
    fn is_eof(&self) -> bool {
        self.tokens.is_empty()
    }
    fn finish(self) -> Result<Grammar> {
        for node_data in &self.grammar.nodes {
            if matches!(node_data.rule, DUMMY_RULE) {
                crate::error::bail!("Undefined node: {}", node_data.name)
            }
        }
        Ok(self.grammar)
    }
    fn intern_node(&mut self, name: String) -> Node {
        let len = self.node_table.len();
        let grammar = &mut self.grammar;
        *self.node_table.entry(name.clone()).or_insert_with(|| {
            grammar.nodes.push(NodeData { name, rule: DUMMY_RULE });
            Node(len)
        })
    }
    fn intern_token(&mut self, name: String) -> Token {
        let len = self.token_table.len();
        let grammar = &mut self.grammar;
        *self.token_table.entry(name.clone()).or_insert_with(|| {
            grammar.tokens.push(TokenData { name });
            Token(len)
        })
    }
}

fn node(p: &mut Parser) -> Result<()> {
    let token = p.bump()?;
    let node = match token.kind {
        TokenKind::Node(it) => p.intern_node(it),
        _ => bail!(token.loc, "expected ident"),
    };
    p.expect(TokenKind::Eq, "=")?;
    if !matches!(p.grammar[node].rule, DUMMY_RULE) {
        bail!(token.loc, "duplicate rule: `{}`", p.grammar[node].name)
    }

    let rule = rule(p)?;
    p.grammar.nodes[node.0].rule = rule;
    Ok(())
}

fn rule(p: &mut Parser) -> Result<Rule> {
    if let Some(lexer::Token { kind: TokenKind::Pipe, loc }) = p.peek() {
        bail!(
            *loc,
            "The first element in a sequence of productions or alternatives \
            must not have a leading pipe (`|`)"
        );
    }

    let lhs = seq_rule(p)?;
    let mut alt = vec![lhs];
    while let Some(token) = p.peek() {
        if token.kind != TokenKind::Pipe {
            break;
        }
        p.bump()?;
        let rule = seq_rule(p)?;
        alt.push(rule)
    }
    let res = if alt.len() == 1 { alt.pop().unwrap() } else { Rule::Alt(alt) };
    Ok(res)
}

fn seq_rule(p: &mut Parser) -> Result<Rule> {
    let lhs = atom_rule(p)?;

    let mut seq = vec![lhs];
    while let Some(rule) = opt_atom_rule(p)? {
        seq.push(rule)
    }
    let res = if seq.len() == 1 { seq.pop().unwrap() } else { Rule::Seq(seq) };
    Ok(res)
}

fn atom_rule(p: &mut Parser) -> Result<Rule> {
    match opt_atom_rule(p)? {
        Some(it) => Ok(it),
        None => {
            let token = p.bump()?;
            bail!(token.loc, "unexpected token")
        }
    }
}

fn opt_atom_rule(p: &mut Parser) -> Result<Option<Rule>> {
    let token = match p.peek() {
        Some(it) => it,
        None => return Ok(None),
    };
    let mut res = match &token.kind {
        TokenKind::Node(name) => {
            if let Some(lookahead) = p.peek_n(1) {
                match lookahead.kind {
                    TokenKind::Eq => return Ok(None),
                    TokenKind::Colon => {
                        let label = name.clone();
                        p.bump()?;
                        p.bump()?;
                        let rule = atom_rule(p)?;
                        let res = Rule::Labeled { label, rule: Box::new(rule) };
                        return Ok(Some(res));
                    }
                    _ => (),
                }
            }
            match p.peek_n(1) {
                Some(token) if token.kind == TokenKind::Eq => return Ok(None),
                _ => (),
            }
            let name = name.clone();
            p.bump()?;
            let node = p.intern_node(name);
            Rule::Node(node)
        }
        TokenKind::Token(name) => {
            let name = name.clone();
            p.bump()?;
            let token = p.intern_token(name);
            Rule::Token(token)
        }
        TokenKind::LParen => {
            p.bump()?;
            let rule = rule(p)?;
            p.expect(TokenKind::RParen, ")")?;
            rule
        }
        _ => return Ok(None),
    };

    if let Some(token) = p.peek() {
        match &token.kind {
            TokenKind::QMark => {
                p.bump()?;
                res = Rule::Opt(Box::new(res));
            }
            TokenKind::Star => {
                p.bump()?;
                res = Rule::Rep(Box::new(res));
            }
            _ => (),
        }
    }
    Ok(Some(res))
}
