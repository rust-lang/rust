//! Semantic Tokens helpers

use std::{fmt, ops, slice::Iter};

use lsp_types::{Range, SemanticTokenModifiers, SemanticTokens, SemanticTokensEdit};

macro_rules! declare_enum {
    (
        $(#[$attrs:meta])*
        $visibility:vis enum $name:ident {
            $($variant:ident),* $(,)?
        }
    ) => {
        $(#[$attrs])*
        $visibility enum $name {
            $($variant,)*
        }

        impl $name {
            pub(crate) fn iter() -> Iter<'static, Self> {
                static ITEMS: &[$name] = &[
                    $(
                        $name::$variant,
                    )*
                ];
                ITEMS.iter()
            }
        }
    };
}

declare_enum! {
    #[repr(u32)]
    #[derive(Debug, PartialEq, Clone, Copy)]
    pub(crate) enum SupportedType {
        Comment,
        Decorator,
        EnumMember,
        Enum,
        Function,
        Interface,
        Keyword,
        Macro,
        Method,
        Namespace,
        Number,
        Operator,
        Parameter,
        Property,
        String,
        Struct,
        TypeParameter,
        Variable,
        Type,
        Label,
        Angle,
        Arithmetic,
        AttributeBracket,
        Attribute,
        Bitwise,
        Boolean,
        Brace,
        Bracket,
        BuiltinAttribute,
        BuiltinType,
        Char,
        Colon,
        Comma,
        Comparison,
        ConstParameter,
        Const,
        DeriveHelper,
        Derive,
        Dot,
        EscapeSequence,
        FormatSpecifier,
        Generic,
        InvalidEscapeSequence,
        Lifetime,
        Logical,
        MacroBang,
        Negation,
        Parenthesis,
        ProcMacro,
        Punctuation,
        SelfKeyword,
        SelfTypeKeyword,
        Semicolon,
        Static,
        ToolModule,
        TypeAlias,
        Union,
        UnresolvedReference,
    }
}

declare_enum! {
    #[repr(u32)]
    #[derive(Debug, PartialEq, Clone, Copy)]
    pub(crate) enum SupportedModifiers {
        Async,
        Documentation,
        Declaration,
        Static,
        DefaultLibrary,
        Deprecated,
        Associated,
        AttributeModifier,
        Callable,
        Constant,
        Consuming,
        ControlFlow,
        CrateRoot,
        Injected,
        IntraDocLink,
        Library,
        MacroModifier,
        Mutable,
        ProcMacroModifier,
        Public,
        Reference,
        TraitModifier,
        Unsafe,
    }
}

impl fmt::Display for SupportedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string = match self {
            SupportedType::Comment => ::lsp_types::SemanticTokenTypes::Comment.as_str(),
            SupportedType::Decorator => ::lsp_types::SemanticTokenTypes::Decorator.as_str(),
            SupportedType::EnumMember => ::lsp_types::SemanticTokenTypes::EnumMember.as_str(),
            SupportedType::Enum => ::lsp_types::SemanticTokenTypes::Enum.as_str(),
            SupportedType::Function => ::lsp_types::SemanticTokenTypes::Function.as_str(),
            SupportedType::Interface => ::lsp_types::SemanticTokenTypes::Interface.as_str(),
            SupportedType::Keyword => ::lsp_types::SemanticTokenTypes::Keyword.as_str(),
            SupportedType::Macro => ::lsp_types::SemanticTokenTypes::Macro.as_str(),
            SupportedType::Method => ::lsp_types::SemanticTokenTypes::Method.as_str(),
            SupportedType::Namespace => ::lsp_types::SemanticTokenTypes::Namespace.as_str(),
            SupportedType::Number => ::lsp_types::SemanticTokenTypes::Number.as_str(),
            SupportedType::Operator => ::lsp_types::SemanticTokenTypes::Operator.as_str(),
            SupportedType::Parameter => ::lsp_types::SemanticTokenTypes::Parameter.as_str(),
            SupportedType::Property => ::lsp_types::SemanticTokenTypes::Property.as_str(),
            SupportedType::String => ::lsp_types::SemanticTokenTypes::String.as_str(),
            SupportedType::Struct => ::lsp_types::SemanticTokenTypes::Struct.as_str(),
            SupportedType::TypeParameter => ::lsp_types::SemanticTokenTypes::TypeParameter.as_str(),
            SupportedType::Variable => ::lsp_types::SemanticTokenTypes::Variable.as_str(),
            SupportedType::Type => ::lsp_types::SemanticTokenTypes::Type.as_str(),
            SupportedType::Label => ::lsp_types::SemanticTokenTypes::Label.as_str(),
            SupportedType::Angle => "angle",
            SupportedType::Arithmetic => "arithmetic",
            SupportedType::AttributeBracket => "attributeBracket",
            SupportedType::Attribute => "attribute",
            SupportedType::Bitwise => "bitwise",
            SupportedType::Boolean => "boolean",
            SupportedType::Brace => "brace",
            SupportedType::Bracket => "bracket",
            SupportedType::BuiltinAttribute => "builtinAttribute",
            SupportedType::BuiltinType => "builtinType",
            SupportedType::Char => "character",
            SupportedType::Colon => "colon",
            SupportedType::Comma => "comma",
            SupportedType::Comparison => "comparison",
            SupportedType::ConstParameter => "constParameter",
            SupportedType::Const => "const",
            SupportedType::DeriveHelper => "deriveHelper",
            SupportedType::Derive => "derive",
            SupportedType::Dot => "dot",
            SupportedType::EscapeSequence => "escapeSequence",
            SupportedType::FormatSpecifier => "formatSpecifier",
            SupportedType::Generic => "generic",
            SupportedType::InvalidEscapeSequence => "invalidEscapeSequence",
            SupportedType::Lifetime => "lifetime",
            SupportedType::Logical => "logical",
            SupportedType::MacroBang => "macroBang",
            SupportedType::Negation => "negation",
            SupportedType::Parenthesis => "parenthesis",
            SupportedType::ProcMacro => "procMacro",
            SupportedType::Punctuation => "punctuation",
            SupportedType::SelfKeyword => "selfKeyword",
            SupportedType::SelfTypeKeyword => "selfTypeKeyword",
            SupportedType::Semicolon => "semicolon",
            SupportedType::Static => "static",
            SupportedType::ToolModule => "toolModule",
            SupportedType::TypeAlias => "typeAlias",
            SupportedType::Union => "union",
            SupportedType::UnresolvedReference => "unresolvedReference",
        };
        f.write_str(string)
    }
}

pub(crate) fn standard_fallback_type(token: SupportedType) -> Option<SupportedType> {
    Some(match token {
        SupportedType::Comment => SupportedType::Comment,
        SupportedType::Decorator => SupportedType::Decorator,
        SupportedType::EnumMember => SupportedType::EnumMember,
        SupportedType::Enum => SupportedType::Enum,
        SupportedType::Function => SupportedType::Function,
        SupportedType::Interface => SupportedType::Interface,
        SupportedType::Keyword => SupportedType::Keyword,
        SupportedType::Macro => SupportedType::Macro,
        SupportedType::Method => SupportedType::Method,
        SupportedType::Namespace => SupportedType::Namespace,
        SupportedType::Number => SupportedType::Number,
        SupportedType::Operator => SupportedType::Operator,
        SupportedType::Parameter => SupportedType::Parameter,
        SupportedType::Property => SupportedType::Property,
        SupportedType::String => SupportedType::String,
        SupportedType::Struct => SupportedType::Struct,
        SupportedType::TypeParameter => SupportedType::TypeParameter,
        SupportedType::Variable => SupportedType::Variable,
        SupportedType::Type => SupportedType::Type,
        SupportedType::Label => SupportedType::Label,
        _ => return None,
    })
}

impl fmt::Display for SupportedModifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string = match self {
            SupportedModifiers::Async => SemanticTokenModifiers::Async.as_str(),
            SupportedModifiers::Documentation => SemanticTokenModifiers::Documentation.as_str(),
            SupportedModifiers::Declaration => SemanticTokenModifiers::Declaration.as_str(),
            SupportedModifiers::Static => SemanticTokenModifiers::Static.as_str(),
            SupportedModifiers::DefaultLibrary => SemanticTokenModifiers::DefaultLibrary.as_str(),
            SupportedModifiers::Deprecated => SemanticTokenModifiers::Deprecated.as_str(),
            SupportedModifiers::Associated => "associated",
            SupportedModifiers::AttributeModifier => "attribute",
            SupportedModifiers::Callable => "callable",
            SupportedModifiers::Constant => "constant",
            SupportedModifiers::Consuming => "consuming",
            SupportedModifiers::ControlFlow => "controlFlow",
            SupportedModifiers::CrateRoot => "crateRoot",
            SupportedModifiers::Injected => "injected",
            SupportedModifiers::IntraDocLink => "intraDocLink",
            SupportedModifiers::Library => "library",
            SupportedModifiers::MacroModifier => "macro",
            SupportedModifiers::Mutable => "mutable",
            SupportedModifiers::ProcMacroModifier => "procMacro",
            SupportedModifiers::Public => "public",
            SupportedModifiers::Reference => "reference",
            SupportedModifiers::TraitModifier => "trait",
            SupportedModifiers::Unsafe => "unsafe",
        };
        f.write_str(string)
    }
}

const STANDARD_MOD: [SupportedModifiers; 6] = [
    SupportedModifiers::Async,
    SupportedModifiers::Documentation,
    SupportedModifiers::Declaration,
    SupportedModifiers::Static,
    SupportedModifiers::DefaultLibrary,
    SupportedModifiers::Deprecated,
];
const LAST_STANDARD_MOD: usize = STANDARD_MOD.len() - 1;

#[derive(Default)]
pub(crate) struct ModifierSet(pub(crate) u32);

impl ModifierSet {
    pub(crate) fn standard_fallback(&mut self) {
        // Remove all non standard modifiers
        self.0 &= !(!0u32 << LAST_STANDARD_MOD)
    }
}

impl ops::BitOrAssign<SupportedModifiers> for ModifierSet {
    fn bitor_assign(&mut self, rhs: SupportedModifiers) {
        let idx = SupportedModifiers::iter().position(|it| *it == rhs).unwrap();
        self.0 |= 1 << idx;
    }
}

/// Tokens are encoded relative to each other.
///
/// This is a direct port of <https://github.com/microsoft/vscode-languageserver-node/blob/f425af9de46a0187adb78ec8a46b9b2ce80c5412/server/src/sematicTokens.proposed.ts#L45>
pub(crate) struct SemanticTokensBuilder {
    id: String,
    prev_line: u32,
    prev_char: u32,
    data: Vec<lsp_types::SemanticToken>,
}

impl SemanticTokensBuilder {
    pub(crate) fn new(id: String) -> Self {
        SemanticTokensBuilder { id, prev_line: 0, prev_char: 0, data: Vec::new() }
    }

    /// Push a new token onto the builder
    pub(crate) fn push(&mut self, range: Range, token_index: u32, modifier_bitset: u32) {
        let mut push_line = range.start.line;
        let mut push_char = range.start.character;

        if !self.data.is_empty() {
            push_line -= self.prev_line;
            if push_line == 0 {
                push_char -= self.prev_char;
            }
        }

        // A token cannot be multiline
        let token_len = range.end.character - range.start.character;

        let token = lsp_types::SemanticToken {
            delta_line: push_line,
            delta_start: push_char,
            length: token_len,
            token_type: token_index,
            token_modifiers_bitset: modifier_bitset,
        };

        self.data.push(token);

        self.prev_line = range.start.line;
        self.prev_char = range.start.character;
    }

    pub(crate) fn build(self) -> SemanticTokens {
        SemanticTokens { result_id: Some(self.id), data: self.data }
    }
}

pub(crate) fn diff_tokens(
    old: &[lsp_types::SemanticToken],
    new: &[lsp_types::SemanticToken],
) -> Vec<SemanticTokensEdit> {
    let offset = new.iter().zip(old.iter()).take_while(|&(n, p)| n == p).count();

    let (_, old) = old.split_at(offset);
    let (_, new) = new.split_at(offset);

    let offset_from_end =
        new.iter().rev().zip(old.iter().rev()).take_while(|&(n, p)| n == p).count();

    let (old, _) = old.split_at(old.len() - offset_from_end);
    let (new, _) = new.split_at(new.len() - offset_from_end);

    if old.is_empty() && new.is_empty() {
        vec![]
    } else {
        // The lsp data field is actually a byte-diff but we
        // travel in tokens so `start` and `delete_count` are in multiples of the
        // serialized size of `SemanticToken`.
        vec![SemanticTokensEdit {
            start: 5 * offset as u32,
            delete_count: 5 * old.len() as u32,
            data: Some(new.into()),
        }]
    }
}

pub(crate) fn type_index(kind: SupportedType) -> u32 {
    kind as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from(t: (u32, u32, u32, u32, u32)) -> lsp_types::SemanticToken {
        lsp_types::SemanticToken {
            delta_line: t.0,
            delta_start: t.1,
            length: t.2,
            token_type: t.3,
            token_modifiers_bitset: t.4,
        }
    }

    #[test]
    fn test_diff_insert_at_end() {
        let before = [from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10))];
        let after = [from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10)), from((11, 12, 13, 14, 15))];

        let edits = diff_tokens(&before, &after);
        assert_eq!(
            edits[0],
            SemanticTokensEdit {
                start: 10,
                delete_count: 0,
                data: Some(vec![from((11, 12, 13, 14, 15))])
            }
        );
    }

    #[test]
    fn test_diff_insert_at_beginning() {
        let before = [from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10))];
        let after = [from((11, 12, 13, 14, 15)), from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10))];

        let edits = diff_tokens(&before, &after);
        assert_eq!(
            edits[0],
            SemanticTokensEdit {
                start: 0,
                delete_count: 0,
                data: Some(vec![from((11, 12, 13, 14, 15))])
            }
        );
    }

    #[test]
    fn test_diff_insert_in_middle() {
        let before = [from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10))];
        let after = [
            from((1, 2, 3, 4, 5)),
            from((10, 20, 30, 40, 50)),
            from((60, 70, 80, 90, 100)),
            from((6, 7, 8, 9, 10)),
        ];

        let edits = diff_tokens(&before, &after);
        assert_eq!(
            edits[0],
            SemanticTokensEdit {
                start: 5,
                delete_count: 0,
                data: Some(vec![from((10, 20, 30, 40, 50)), from((60, 70, 80, 90, 100))])
            }
        );
    }

    #[test]
    fn test_diff_remove_from_end() {
        let before = [from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10)), from((11, 12, 13, 14, 15))];
        let after = [from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10))];

        let edits = diff_tokens(&before, &after);
        assert_eq!(edits[0], SemanticTokensEdit { start: 10, delete_count: 5, data: Some(vec![]) });
    }

    #[test]
    fn test_diff_remove_from_beginning() {
        let before = [from((11, 12, 13, 14, 15)), from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10))];
        let after = [from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10))];

        let edits = diff_tokens(&before, &after);
        assert_eq!(edits[0], SemanticTokensEdit { start: 0, delete_count: 5, data: Some(vec![]) });
    }

    #[test]
    fn test_diff_remove_from_middle() {
        let before = [
            from((1, 2, 3, 4, 5)),
            from((10, 20, 30, 40, 50)),
            from((60, 70, 80, 90, 100)),
            from((6, 7, 8, 9, 10)),
        ];
        let after = [from((1, 2, 3, 4, 5)), from((6, 7, 8, 9, 10))];

        let edits = diff_tokens(&before, &after);
        assert_eq!(edits[0], SemanticTokensEdit { start: 5, delete_count: 10, data: Some(vec![]) });
    }
}
