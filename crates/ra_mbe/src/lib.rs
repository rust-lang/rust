//! `mbe` (short for Macro By Example) crate contains code for handling
//! `macro_rules` macros. It uses `TokenTree` (from `ra_tt` package) as the
//! interface, although it contains some code to bridge `SyntaxNode`s and
//! `TokenTree`s as well!

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
    ast_to_token_tree, syntax_node_to_token_tree, token_tree_to_syntax_node, TokenMap,
};

/// This struct contains AST for a single `macro_rules` definition. What might
/// be very confusing is that AST has almost exactly the same shape as
/// `tt::TokenTree`, but there's a crucial difference: in macro rules, `$ident`
/// and `$()*` have special meaning (see `Var` and `Repeat` data structures)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroRules {
    rules: Vec<Rule>,
    /// Highest id of the token we have in TokenMap
    shift: Shift,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Rule {
    lhs: tt::Subtree,
    rhs: tt::Subtree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shift(u32);

impl Shift {
    fn new(tt: &tt::Subtree) -> Shift {
        // Note that TokenId is started from zero,
        // We have to add 1 to prevent duplication.
        let value = max_id(tt).map_or(0, |it| it + 1);
        return Shift(value);

        // Find the max token id inside a subtree
        fn max_id(subtree: &tt::Subtree) -> Option<u32> {
            subtree
                .token_trees
                .iter()
                .filter_map(|tt| match tt {
                    tt::TokenTree::Subtree(subtree) => {
                        let tree_id = max_id(subtree);
                        match subtree.delimiter {
                            Some(it) if it.id != tt::TokenId::unspecified() => {
                                Some(tree_id.map_or(it.id.0, |t| t.max(it.id.0)))
                            }
                            _ => tree_id,
                        }
                    }
                    tt::TokenTree::Leaf(tt::Leaf::Ident(ident))
                        if ident.id != tt::TokenId::unspecified() =>
                    {
                        Some(ident.id.0)
                    }
                    _ => None,
                })
                .max()
        }
    }

    /// Shift given TokenTree token id
    fn shift_all(self, tt: &mut tt::Subtree) {
        for t in tt.token_trees.iter_mut() {
            match t {
                tt::TokenTree::Leaf(leaf) => match leaf {
                    tt::Leaf::Ident(ident) => ident.id = self.shift(ident.id),
                    tt::Leaf::Punct(punct) => punct.id = self.shift(punct.id),
                    tt::Leaf::Literal(lit) => lit.id = self.shift(lit.id),
                },
                tt::TokenTree::Subtree(tt) => {
                    if let Some(it) = tt.delimiter.as_mut() {
                        it.id = self.shift(it.id);
                    };
                    self.shift_all(tt)
                }
            }
        }
    }

    fn shift(self, id: tt::TokenId) -> tt::TokenId {
        if id == tt::TokenId::unspecified() {
            return id;
        }
        tt::TokenId(id.0 + self.0)
    }

    fn unshift(self, id: tt::TokenId) -> Option<tt::TokenId> {
        id.0.checked_sub(self.0).map(tt::TokenId)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Origin {
    Def,
    Call,
}

impl MacroRules {
    pub fn parse(tt: &tt::Subtree) -> Result<MacroRules, ParseError> {
        // Note: this parsing can be implemented using mbe machinery itself, by
        // matching against `$($lhs:tt => $rhs:tt);*` pattern, but implementing
        // manually seems easier.
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

        for rule in rules.iter() {
            validate(&rule.lhs)?;
        }

        Ok(MacroRules { rules, shift: Shift::new(tt) })
    }

    pub fn expand(&self, tt: &tt::Subtree) -> Result<tt::Subtree, ExpandError> {
        // apply shift
        let mut tt = tt.clone();
        self.shift.shift_all(&mut tt);
        mbe_expander::expand(self, &tt)
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
}

impl Rule {
    fn parse(src: &mut TtIter) -> Result<Rule, ParseError> {
        let mut lhs = src
            .expect_subtree()
            .map_err(|()| ParseError::Expected("expected subtree".to_string()))?
            .clone();
        lhs.delimiter = None;
        src.expect_char('=').map_err(|()| ParseError::Expected("expected `=`".to_string()))?;
        src.expect_char('>').map_err(|()| ParseError::Expected("expected `>`".to_string()))?;
        let mut rhs = src
            .expect_subtree()
            .map_err(|()| ParseError::Expected("expected subtree".to_string()))?
            .clone();
        rhs.delimiter = None;
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
