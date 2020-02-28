//! Semantic Tokens helpers

use std::ops;

use lsp_types::{Range, SemanticToken, SemanticTokenModifier, SemanticTokenType};

pub(crate) const ATTRIBUTE: SemanticTokenType = SemanticTokenType::new("attribute");
pub(crate) const BUILTIN_TYPE: SemanticTokenType = SemanticTokenType::new("builtinType");
pub(crate) const ENUM_MEMBER: SemanticTokenType = SemanticTokenType::new("enumMember");
pub(crate) const LIFETIME: SemanticTokenType = SemanticTokenType::new("lifetime");
pub(crate) const TYPE_ALIAS: SemanticTokenType = SemanticTokenType::new("typeAlias");
pub(crate) const UNION: SemanticTokenType = SemanticTokenType::new("union");

pub(crate) const CONSTANT: SemanticTokenModifier = SemanticTokenModifier::new("constant");
pub(crate) const CONTROL: SemanticTokenModifier = SemanticTokenModifier::new("control");
pub(crate) const MUTABLE: SemanticTokenModifier = SemanticTokenModifier::new("mutable");
pub(crate) const UNSAFE: SemanticTokenModifier = SemanticTokenModifier::new("unsafe");

pub(crate) const SUPPORTED_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::COMMENT,
    SemanticTokenType::KEYWORD,
    SemanticTokenType::STRING,
    SemanticTokenType::NUMBER,
    SemanticTokenType::REGEXP,
    SemanticTokenType::OPERATOR,
    SemanticTokenType::NAMESPACE,
    SemanticTokenType::TYPE,
    SemanticTokenType::STRUCT,
    SemanticTokenType::CLASS,
    SemanticTokenType::INTERFACE,
    SemanticTokenType::ENUM,
    SemanticTokenType::TYPE_PARAMETER,
    SemanticTokenType::FUNCTION,
    SemanticTokenType::MEMBER,
    SemanticTokenType::PROPERTY,
    SemanticTokenType::MACRO,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::PARAMETER,
    SemanticTokenType::LABEL,
    ATTRIBUTE,
    BUILTIN_TYPE,
    ENUM_MEMBER,
    LIFETIME,
    TYPE_ALIAS,
    UNION,
];

pub(crate) const SUPPORTED_MODIFIERS: &[SemanticTokenModifier] = &[
    SemanticTokenModifier::DOCUMENTATION,
    SemanticTokenModifier::DECLARATION,
    SemanticTokenModifier::DEFINITION,
    SemanticTokenModifier::REFERENCE,
    SemanticTokenModifier::STATIC,
    SemanticTokenModifier::ABSTRACT,
    SemanticTokenModifier::DEPRECATED,
    SemanticTokenModifier::ASYNC,
    SemanticTokenModifier::VOLATILE,
    SemanticTokenModifier::READONLY,
    CONSTANT,
    MUTABLE,
    UNSAFE,
    CONTROL,
];

#[derive(Default)]
pub(crate) struct ModifierSet(pub(crate) u32);

impl ops::BitOrAssign<SemanticTokenModifier> for ModifierSet {
    fn bitor_assign(&mut self, rhs: SemanticTokenModifier) {
        let idx = SUPPORTED_MODIFIERS.iter().position(|it| it == &rhs).unwrap();
        self.0 |= 1 << idx;
    }
}

/// Tokens are encoded relative to each other.
///
/// This is a direct port of https://github.com/microsoft/vscode-languageserver-node/blob/f425af9de46a0187adb78ec8a46b9b2ce80c5412/server/src/sematicTokens.proposed.ts#L45
#[derive(Default)]
pub(crate) struct SemanticTokensBuilder {
    prev_line: u32,
    prev_char: u32,
    data: Vec<SemanticToken>,
}

impl SemanticTokensBuilder {
    /// Push a new token onto the builder
    pub fn push(&mut self, range: Range, token_index: u32, modifier_bitset: u32) {
        let mut push_line = range.start.line as u32;
        let mut push_char = range.start.character as u32;

        if !self.data.is_empty() {
            push_line -= self.prev_line;
            if push_line == 0 {
                push_char -= self.prev_char;
            }
        }

        // A token cannot be multiline
        let token_len = range.end.character - range.start.character;

        let token = SemanticToken {
            delta_line: push_line,
            delta_start: push_char,
            length: token_len as u32,
            token_type: token_index,
            token_modifiers_bitset: modifier_bitset,
        };

        self.data.push(token);

        self.prev_line = range.start.line as u32;
        self.prev_char = range.start.character as u32;
    }

    pub fn build(self) -> Vec<SemanticToken> {
        self.data
    }
}

pub fn type_index(type_: SemanticTokenType) -> u32 {
    SUPPORTED_TYPES.iter().position(|it| *it == type_).unwrap() as u32
}
