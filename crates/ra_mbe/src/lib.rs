/// `mbe` (short for Macro By Example) crate contains code for handling
/// `macro_rules` macros. It uses `TokenTree` (from `ra_tt` package) as the
/// interface, although it contains some code to bridge `SyntaxNode`s and
/// `TokenTree`s as well!

mod parser;
mod mbe_expander;
mod syntax_bridge;
mod tt_iter;
mod subtree_source;

pub use tt::{Delimiter, Punct};

use crate::{
    parser::{parse_pattern, Op},
    tt_iter::TtIter,
};

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    Expected(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExpandError {
    NoMatchingRule,
    UnexpectedToken,
    BindingError(String),
    ConversionError,
    InvalidRepeat,
}

pub use crate::syntax_bridge::{
    ast_to_token_tree, syntax_node_to_token_tree, token_tree_to_expr, token_tree_to_items,
    token_tree_to_macro_stmts, token_tree_to_pat, token_tree_to_ty,
};

/// This struct contains AST for a single `macro_rules` definition. What might
/// be very confusing is that AST has almost exactly the same shape as
/// `tt::TokenTree`, but there's a crucial difference: in macro rules, `$ident`
/// and `$()*` have special meaning (see `Var` and `Repeat` data structures)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroRules {
    pub(crate) rules: Vec<Rule>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rule {
    pub(crate) lhs: tt::Subtree,
    pub(crate) rhs: tt::Subtree,
}

impl MacroRules {
    pub fn parse(tt: &tt::Subtree) -> Result<MacroRules, ParseError> {
        let mut src = TtIter::new(tt);
        let mut rules = Vec::new();
        while src.len() > 0 {
            let rule = Rule::parse(&mut src)?;
            rules.push(rule);
            if let Err(()) = src.expect_char(';') {
                if src.len() > 0 {
                    return Err(ParseError::Expected("expected `:`".to_string()));
                }
                break;
            }
        }
        Ok(MacroRules { rules })
    }
    pub fn expand(&self, tt: &tt::Subtree) -> Result<tt::Subtree, ExpandError> {
        mbe_expander::expand(self, tt)
    }
}

impl Rule {
    fn parse(src: &mut TtIter) -> Result<Rule, ParseError> {
        let mut lhs = src
            .expect_subtree()
            .map_err(|()| ParseError::Expected("expected subtree".to_string()))?
            .clone();
        validate(&lhs)?;
        lhs.delimiter = tt::Delimiter::None;
        src.expect_char('=').map_err(|()| ParseError::Expected("expected `=`".to_string()))?;
        src.expect_char('>').map_err(|()| ParseError::Expected("expected `>`".to_string()))?;
        let mut rhs = src
            .expect_subtree()
            .map_err(|()| ParseError::Expected("expected subtree".to_string()))?
            .clone();
        rhs.delimiter = tt::Delimiter::None;
        Ok(crate::Rule { lhs, rhs })
    }
}

fn validate(pattern: &tt::Subtree) -> Result<(), ParseError> {
    for op in parse_pattern(pattern) {
        let op = match op {
            Ok(it) => it,
            Err(e) => {
                let msg = match e {
                    ExpandError::InvalidRepeat => "invalid repeat".to_string(),
                    _ => "invalid macro definition".to_string(),
                };
                return Err(ParseError::Expected(msg));
            }
        };
        match op {
            Op::TokenTree(tt::TokenTree::Subtree(subtree)) | Op::Repeat { subtree, .. } => {
                validate(subtree)?
            }
            _ => (),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
