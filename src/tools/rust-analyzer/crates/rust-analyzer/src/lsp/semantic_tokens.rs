//! Semantic Tokens helpers

use std::ops;

use lsp_types::{
    Range, SemanticToken, SemanticTokenModifier, SemanticTokenType, SemanticTokens,
    SemanticTokensEdit,
};

macro_rules! define_semantic_token_types {
    (
        standard {
            $($standard:ident),*$(,)?
        }
        custom {
            $(($custom:ident, $string:literal) $(=> $fallback:ident)?),*$(,)?
        }

    ) => {
        pub(crate) mod types {
            use super::SemanticTokenType;
            $(pub(crate) const $standard: SemanticTokenType = SemanticTokenType::$standard;)*
            $(pub(crate) const $custom: SemanticTokenType = SemanticTokenType::new($string);)*
        }

        pub(crate) const SUPPORTED_TYPES: &[SemanticTokenType] = &[
            $(SemanticTokenType::$standard,)*
            $(self::types::$custom),*
        ];

        pub(crate) fn standard_fallback_type(token: SemanticTokenType) -> Option<SemanticTokenType> {
            use self::types::*;
            $(
                if token == $custom {
                    None $(.or(Some(SemanticTokenType::$fallback)))?
                } else
            )*
            { Some(token )}
        }
    };
}

define_semantic_token_types![
    standard {
        COMMENT,
        DECORATOR,
        ENUM_MEMBER,
        ENUM,
        FUNCTION,
        INTERFACE,
        KEYWORD,
        MACRO,
        METHOD,
        NAMESPACE,
        NUMBER,
        OPERATOR,
        PARAMETER,
        PROPERTY,
        STRING,
        STRUCT,
        TYPE_PARAMETER,
        VARIABLE,
    }

    custom {
        (ANGLE, "angle"),
        (ARITHMETIC, "arithmetic") => OPERATOR,
        (ATTRIBUTE_BRACKET, "attributeBracket") => DECORATOR,
        (ATTRIBUTE, "attribute") => DECORATOR,
        (BITWISE, "bitwise") => OPERATOR,
        (BOOLEAN, "boolean"),
        (BRACE, "brace"),
        (BRACKET, "bracket"),
        (BUILTIN_ATTRIBUTE, "builtinAttribute") => DECORATOR,
        (BUILTIN_TYPE, "builtinType") => TYPE,
        (CHAR, "character") => STRING,
        (COLON, "colon"),
        (COMMA, "comma"),
        (COMPARISON, "comparison") => OPERATOR,
        (CONST_PARAMETER, "constParameter"),
        (CONST, "const") => VARIABLE,
        (DERIVE_HELPER, "deriveHelper") => DECORATOR,
        (DERIVE, "derive") => DECORATOR,
        (DOT, "dot"),
        (ESCAPE_SEQUENCE, "escapeSequence") => STRING,
        (FORMAT_SPECIFIER, "formatSpecifier") => STRING,
        (GENERIC, "generic") => TYPE_PARAMETER,
        (INVALID_ESCAPE_SEQUENCE, "invalidEscapeSequence") => STRING,
        (LABEL, "label"),
        (LIFETIME, "lifetime"),
        (LOGICAL, "logical") => OPERATOR,
        (MACRO_BANG, "macroBang") => MACRO,
        (PARENTHESIS, "parenthesis"),
        (PROC_MACRO, "procMacro") => MACRO,
        (PUNCTUATION, "punctuation"),
        (SELF_KEYWORD, "selfKeyword") => KEYWORD,
        (SELF_TYPE_KEYWORD, "selfTypeKeyword") => KEYWORD,
        (SEMICOLON, "semicolon"),
        (STATIC, "static") => VARIABLE,
        (TOOL_MODULE, "toolModule") => DECORATOR,
        (TYPE_ALIAS, "typeAlias") => TYPE,
        (UNION, "union") => TYPE,
        (UNRESOLVED_REFERENCE, "unresolvedReference"),
    }
];

macro_rules! count_tts {
    () => {0usize};
    ($_head:tt $($tail:tt)*) => {1usize + count_tts!($($tail)*)};
}
macro_rules! define_semantic_token_modifiers {
    (
        standard {
            $($standard:ident),*$(,)?
        }
        custom {
            $(($custom:ident, $string:literal)),*$(,)?
        }

    ) => {
        pub(crate) mod modifiers {
            use super::SemanticTokenModifier;

            $(pub(crate) const $standard: SemanticTokenModifier = SemanticTokenModifier::$standard;)*
            $(pub(crate) const $custom: SemanticTokenModifier = SemanticTokenModifier::new($string);)*
        }

        pub(crate) const SUPPORTED_MODIFIERS: &[SemanticTokenModifier] = &[
            $(SemanticTokenModifier::$standard,)*
            $(self::modifiers::$custom),*
        ];

        const LAST_STANDARD_MOD: usize = count_tts!($($standard)*);
    };
}

define_semantic_token_modifiers![
    standard {
        ASYNC,
        DOCUMENTATION,
        DECLARATION,
        STATIC,
        DEFAULT_LIBRARY,
    }
    custom {
        (ASSOCIATED, "associated"),
        (ATTRIBUTE_MODIFIER, "attribute"),
        (CALLABLE, "callable"),
        (CONSTANT, "constant"),
        (CONSUMING, "consuming"),
        (CONTROL_FLOW, "controlFlow"),
        (CRATE_ROOT, "crateRoot"),
        (INJECTED, "injected"),
        (INTRA_DOC_LINK, "intraDocLink"),
        (LIBRARY, "library"),
        (MACRO_MODIFIER, "macro"),
        (MUTABLE, "mutable"),
        (PROC_MACRO_MODIFIER, "procMacro"),
        (PUBLIC, "public"),
        (REFERENCE, "reference"),
        (TRAIT_MODIFIER, "trait"),
        (UNSAFE, "unsafe"),
    }
];

#[derive(Default)]
pub(crate) struct ModifierSet(pub(crate) u32);

impl ModifierSet {
    pub(crate) fn standard_fallback(&mut self) {
        // Remove all non standard modifiers
        self.0 &= !(!0u32 << LAST_STANDARD_MOD)
    }
}

impl ops::BitOrAssign<SemanticTokenModifier> for ModifierSet {
    fn bitor_assign(&mut self, rhs: SemanticTokenModifier) {
        let idx = SUPPORTED_MODIFIERS.iter().position(|it| it == &rhs).unwrap();
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
    data: Vec<SemanticToken>,
}

impl SemanticTokensBuilder {
    pub(crate) fn new(id: String) -> Self {
        SemanticTokensBuilder { id, prev_line: 0, prev_char: 0, data: Default::default() }
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

        let token = SemanticToken {
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

pub(crate) fn diff_tokens(old: &[SemanticToken], new: &[SemanticToken]) -> Vec<SemanticTokensEdit> {
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

pub(crate) fn type_index(ty: SemanticTokenType) -> u32 {
    SUPPORTED_TYPES.iter().position(|it| *it == ty).unwrap() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from(t: (u32, u32, u32, u32, u32)) -> SemanticToken {
        SemanticToken {
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
