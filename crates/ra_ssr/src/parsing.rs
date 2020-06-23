//! This file contains code for parsing SSR rules, which look something like `foo($a) ==>> bar($b)`.
//! We first split everything before and after the separator `==>>`. Next, both the search pattern
//! and the replacement template get tokenized by the Rust tokenizer. Tokens are then searched for
//! placeholders, which start with `$`. For replacement templates, this is the final form. For
//! search patterns, we go further and parse the pattern as each kind of thing that we can match.
//! e.g. expressions, type references etc.

use crate::errors::bail;
use crate::{SsrError, SsrPattern, SsrRule};
use ra_syntax::{ast, AstNode, SmolStr, SyntaxKind, T};
use rustc_hash::{FxHashMap, FxHashSet};
use std::str::FromStr;

#[derive(Clone, Debug)]
pub(crate) struct SsrTemplate {
    pub(crate) tokens: Vec<PatternElement>,
}

#[derive(Debug)]
pub(crate) struct RawSearchPattern {
    tokens: Vec<PatternElement>,
}

// Part of a search or replace pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PatternElement {
    Token(Token),
    Placeholder(Placeholder),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Placeholder {
    /// The name of this placeholder. e.g. for "$a", this would be "a"
    pub(crate) ident: SmolStr,
    /// A unique name used in place of this placeholder when we parse the pattern as Rust code.
    stand_in_name: String,
    pub(crate) constraints: Vec<Constraint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Constraint {
    Kind(NodeKind),
    Not(Box<Constraint>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NodeKind {
    Literal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Token {
    kind: SyntaxKind,
    pub(crate) text: SmolStr,
}

impl FromStr for SsrRule {
    type Err = SsrError;

    fn from_str(query: &str) -> Result<SsrRule, SsrError> {
        let mut it = query.split("==>>");
        let pattern = it.next().expect("at least empty string").trim();
        let template = it
            .next()
            .ok_or_else(|| SsrError("Cannot find delimiter `==>>`".into()))?
            .trim()
            .to_string();
        if it.next().is_some() {
            return Err(SsrError("More than one delimiter found".into()));
        }
        let rule = SsrRule { pattern: pattern.parse()?, template: template.parse()? };
        validate_rule(&rule)?;
        Ok(rule)
    }
}

impl FromStr for RawSearchPattern {
    type Err = SsrError;

    fn from_str(pattern_str: &str) -> Result<RawSearchPattern, SsrError> {
        Ok(RawSearchPattern { tokens: parse_pattern(pattern_str)? })
    }
}

impl RawSearchPattern {
    /// Returns this search pattern as Rust source code that we can feed to the Rust parser.
    fn as_rust_code(&self) -> String {
        let mut res = String::new();
        for t in &self.tokens {
            res.push_str(match t {
                PatternElement::Token(token) => token.text.as_str(),
                PatternElement::Placeholder(placeholder) => placeholder.stand_in_name.as_str(),
            });
        }
        res
    }

    fn placeholders_by_stand_in(&self) -> FxHashMap<SmolStr, Placeholder> {
        let mut res = FxHashMap::default();
        for t in &self.tokens {
            if let PatternElement::Placeholder(placeholder) = t {
                res.insert(SmolStr::new(placeholder.stand_in_name.clone()), placeholder.clone());
            }
        }
        res
    }
}

impl FromStr for SsrPattern {
    type Err = SsrError;

    fn from_str(pattern_str: &str) -> Result<SsrPattern, SsrError> {
        let raw: RawSearchPattern = pattern_str.parse()?;
        let raw_str = raw.as_rust_code();
        let res = SsrPattern {
            expr: ast::Expr::parse(&raw_str).ok().map(|n| n.syntax().clone()),
            type_ref: ast::TypeRef::parse(&raw_str).ok().map(|n| n.syntax().clone()),
            item: ast::ModuleItem::parse(&raw_str).ok().map(|n| n.syntax().clone()),
            path: ast::Path::parse(&raw_str).ok().map(|n| n.syntax().clone()),
            pattern: ast::Pat::parse(&raw_str).ok().map(|n| n.syntax().clone()),
            placeholders_by_stand_in: raw.placeholders_by_stand_in(),
            raw,
        };
        if res.expr.is_none()
            && res.type_ref.is_none()
            && res.item.is_none()
            && res.path.is_none()
            && res.pattern.is_none()
        {
            bail!("Pattern is not a valid Rust expression, type, item, path or pattern");
        }
        Ok(res)
    }
}

impl FromStr for SsrTemplate {
    type Err = SsrError;

    fn from_str(pattern_str: &str) -> Result<SsrTemplate, SsrError> {
        let tokens = parse_pattern(pattern_str)?;
        // Validate that the template is a valid fragment of Rust code. We reuse the validation
        // logic for search patterns since the only thing that differs is the error message.
        if SsrPattern::from_str(pattern_str).is_err() {
            bail!("Replacement is not a valid Rust expression, type, item, path or pattern");
        }
        // Our actual template needs to preserve whitespace, so we can't reuse `tokens`.
        Ok(SsrTemplate { tokens })
    }
}

/// Returns `pattern_str`, parsed as a search or replace pattern. If `remove_whitespace` is true,
/// then any whitespace tokens will be removed, which we do for the search pattern, but not for the
/// replace pattern.
fn parse_pattern(pattern_str: &str) -> Result<Vec<PatternElement>, SsrError> {
    let mut res = Vec::new();
    let mut placeholder_names = FxHashSet::default();
    let mut tokens = tokenize(pattern_str)?.into_iter();
    while let Some(token) = tokens.next() {
        if token.kind == T![$] {
            let placeholder = parse_placeholder(&mut tokens)?;
            if !placeholder_names.insert(placeholder.ident.clone()) {
                bail!("Name `{}` repeats more than once", placeholder.ident);
            }
            res.push(PatternElement::Placeholder(placeholder));
        } else {
            res.push(PatternElement::Token(token));
        }
    }
    Ok(res)
}

/// Checks for errors in a rule. e.g. the replace pattern referencing placeholders that the search
/// pattern didn't define.
fn validate_rule(rule: &SsrRule) -> Result<(), SsrError> {
    let mut defined_placeholders = FxHashSet::default();
    for p in &rule.pattern.raw.tokens {
        if let PatternElement::Placeholder(placeholder) = p {
            defined_placeholders.insert(&placeholder.ident);
        }
    }
    let mut undefined = Vec::new();
    for p in &rule.template.tokens {
        if let PatternElement::Placeholder(placeholder) = p {
            if !defined_placeholders.contains(&placeholder.ident) {
                undefined.push(format!("${}", placeholder.ident));
            }
            if !placeholder.constraints.is_empty() {
                bail!("Replacement placeholders cannot have constraints");
            }
        }
    }
    if !undefined.is_empty() {
        bail!("Replacement contains undefined placeholders: {}", undefined.join(", "));
    }
    Ok(())
}

fn tokenize(source: &str) -> Result<Vec<Token>, SsrError> {
    let mut start = 0;
    let (raw_tokens, errors) = ra_syntax::tokenize(source);
    if let Some(first_error) = errors.first() {
        bail!("Failed to parse pattern: {}", first_error);
    }
    let mut tokens: Vec<Token> = Vec::new();
    for raw_token in raw_tokens {
        let token_len = usize::from(raw_token.len);
        tokens.push(Token {
            kind: raw_token.kind,
            text: SmolStr::new(&source[start..start + token_len]),
        });
        start += token_len;
    }
    Ok(tokens)
}

fn parse_placeholder(tokens: &mut std::vec::IntoIter<Token>) -> Result<Placeholder, SsrError> {
    let mut name = None;
    let mut constraints = Vec::new();
    if let Some(token) = tokens.next() {
        match token.kind {
            SyntaxKind::IDENT => {
                name = Some(token.text);
            }
            T!['{'] => {
                let token =
                    tokens.next().ok_or_else(|| SsrError::new("Unexpected end of placeholder"))?;
                if token.kind == SyntaxKind::IDENT {
                    name = Some(token.text);
                }
                loop {
                    let token = tokens
                        .next()
                        .ok_or_else(|| SsrError::new("Placeholder is missing closing brace '}'"))?;
                    match token.kind {
                        T![:] => {
                            constraints.push(parse_constraint(tokens)?);
                        }
                        T!['}'] => break,
                        _ => bail!("Unexpected token while parsing placeholder: '{}'", token.text),
                    }
                }
            }
            _ => {
                bail!("Placeholders should either be $name or ${{name:constraints}}");
            }
        }
    }
    let name = name.ok_or_else(|| SsrError::new("Placeholder ($) with no name"))?;
    Ok(Placeholder::new(name, constraints))
}

fn parse_constraint(tokens: &mut std::vec::IntoIter<Token>) -> Result<Constraint, SsrError> {
    let constraint_type = tokens
        .next()
        .ok_or_else(|| SsrError::new("Found end of placeholder while looking for a constraint"))?
        .text
        .to_string();
    match constraint_type.as_str() {
        "kind" => {
            expect_token(tokens, "(")?;
            let t = tokens.next().ok_or_else(|| {
                SsrError::new("Unexpected end of constraint while looking for kind")
            })?;
            if t.kind != SyntaxKind::IDENT {
                bail!("Expected ident, found {:?} while parsing kind constraint", t.kind);
            }
            expect_token(tokens, ")")?;
            Ok(Constraint::Kind(NodeKind::from(&t.text)?))
        }
        "not" => {
            expect_token(tokens, "(")?;
            let sub = parse_constraint(tokens)?;
            expect_token(tokens, ")")?;
            Ok(Constraint::Not(Box::new(sub)))
        }
        x => bail!("Unsupported constraint type '{}'", x),
    }
}

fn expect_token(tokens: &mut std::vec::IntoIter<Token>, expected: &str) -> Result<(), SsrError> {
    if let Some(t) = tokens.next() {
        if t.text == expected {
            return Ok(());
        }
        bail!("Expected {} found {}", expected, t.text);
    }
    bail!("Expected {} found end of stream", expected);
}

impl NodeKind {
    fn from(name: &SmolStr) -> Result<NodeKind, SsrError> {
        Ok(match name.as_str() {
            "literal" => NodeKind::Literal,
            _ => bail!("Unknown node kind '{}'", name),
        })
    }
}

impl Placeholder {
    fn new(name: SmolStr, constraints: Vec<Constraint>) -> Self {
        Self { stand_in_name: format!("__placeholder_{}", name), constraints, ident: name }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_happy_case() {
        fn token(kind: SyntaxKind, text: &str) -> PatternElement {
            PatternElement::Token(Token { kind, text: SmolStr::new(text) })
        }
        fn placeholder(name: &str) -> PatternElement {
            PatternElement::Placeholder(Placeholder::new(SmolStr::new(name), Vec::new()))
        }
        let result: SsrRule = "foo($a, $b) ==>> bar($b, $a)".parse().unwrap();
        assert_eq!(
            result.pattern.raw.tokens,
            vec![
                token(SyntaxKind::IDENT, "foo"),
                token(T!['('], "("),
                placeholder("a"),
                token(T![,], ","),
                token(SyntaxKind::WHITESPACE, " "),
                placeholder("b"),
                token(T![')'], ")"),
            ]
        );
        assert_eq!(
            result.template.tokens,
            vec![
                token(SyntaxKind::IDENT, "bar"),
                token(T!['('], "("),
                placeholder("b"),
                token(T![,], ","),
                token(SyntaxKind::WHITESPACE, " "),
                placeholder("a"),
                token(T![')'], ")"),
            ]
        );
    }
}
