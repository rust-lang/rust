//! Syntax highlighting for macro_rules!.
use syntax::{SyntaxKind, SyntaxToken, TextRange, T};

use crate::{HlRange, HlTag};

#[derive(Default)]
pub(super) struct MacroHighlighter {
    state: Option<MacroMatcherParseState>,
}

impl MacroHighlighter {
    pub(super) fn init(&mut self) {
        self.state = Some(MacroMatcherParseState::default());
    }

    pub(super) fn advance(&mut self, token: &SyntaxToken) {
        if let Some(state) = self.state.as_mut() {
            update_macro_state(state, token);
        }
    }

    pub(super) fn highlight(&self, token: &SyntaxToken) -> Option<HlRange> {
        if let Some(state) = self.state.as_ref() {
            if matches!(state.rule_state, RuleState::Matcher | RuleState::Expander) {
                if let Some(range) = is_metavariable(token) {
                    return Some(HlRange {
                        range,
                        highlight: HlTag::UnresolvedReference.into(),
                        binding_hash: None,
                    });
                }
            }
        }
        None
    }
}

struct MacroMatcherParseState {
    /// Opening and corresponding closing bracket of the matcher or expander of the current rule
    paren_ty: Option<(SyntaxKind, SyntaxKind)>,
    paren_level: usize,
    rule_state: RuleState,
    /// Whether we are inside the outer `{` `}` macro block that holds the rules
    in_invoc_body: bool,
}

impl Default for MacroMatcherParseState {
    fn default() -> Self {
        MacroMatcherParseState {
            paren_ty: None,
            paren_level: 0,
            in_invoc_body: false,
            rule_state: RuleState::None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum RuleState {
    Matcher,
    Expander,
    Between,
    None,
}

impl RuleState {
    fn transition(&mut self) {
        *self = match self {
            RuleState::Matcher => RuleState::Between,
            RuleState::Expander => RuleState::None,
            RuleState::Between => RuleState::Expander,
            RuleState::None => RuleState::Matcher,
        };
    }
}

fn update_macro_state(state: &mut MacroMatcherParseState, tok: &SyntaxToken) {
    if !state.in_invoc_body {
        if tok.kind() == T!['{'] || tok.kind() == T!['('] {
            state.in_invoc_body = true;
        }
        return;
    }

    match state.paren_ty {
        Some((open, close)) => {
            if tok.kind() == open {
                state.paren_level += 1;
            } else if tok.kind() == close {
                state.paren_level -= 1;
                if state.paren_level == 0 {
                    state.rule_state.transition();
                    state.paren_ty = None;
                }
            }
        }
        None => {
            match tok.kind() {
                T!['('] => {
                    state.paren_ty = Some((T!['('], T![')']));
                }
                T!['{'] => {
                    state.paren_ty = Some((T!['{'], T!['}']));
                }
                T!['['] => {
                    state.paren_ty = Some((T!['['], T![']']));
                }
                _ => (),
            }
            if state.paren_ty.is_some() {
                state.paren_level = 1;
                state.rule_state.transition();
            }
        }
    }
}

fn is_metavariable(token: &SyntaxToken) -> Option<TextRange> {
    match token.kind() {
        kind if kind == SyntaxKind::IDENT || kind.is_keyword() => {
            if let Some(_dollar) = token.prev_token().filter(|t| t.kind() == T![$]) {
                return Some(token.text_range());
            }
        }
        _ => (),
    };
    None
}
