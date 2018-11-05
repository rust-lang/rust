use crate::{
    algo::visit::{visitor_ctx, VisitorCtx},
    ast::{self, AstNode},
    File,
    string_lexing::{self, CharComponentKind},
    yellow::{
        SyntaxError,
        SyntaxErrorKind::*,
    },
};

pub(crate) fn validate(file: &File) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in file.root.borrowed().descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Char, _>(validate_char)
            .accept(node);
    }
    errors
}

fn validate_char(node: ast::Char, errors: &mut Vec<SyntaxError>) {
    let mut components = string_lexing::parse_char_literal(node.text());
    let mut len = 0;
    for component in &mut components {
        len += 1;

        // Validate escapes
        let text = &node.text()[component.range];
        let range = component.range + node.syntax().range().start();
        use self::CharComponentKind::*;
        match component.kind {
            AsciiEscape => {
                if text.len() == 1 {
                    // Escape sequence consists only of leading `\`
                    errors.push(SyntaxError::new(EmptyAsciiEscape, range));
                } else {
                    let escape_code = text.chars().skip(1).next().unwrap();
                    if !is_ascii_escape(escape_code) {
                        errors.push(SyntaxError::new(InvalidAsciiEscape, range));
                    }
                }
            }
            AsciiCodeEscape => {
                // TODO:
                // * First digit is octal
                // * Second digit is hex
            }
            UnicodeEscape => {
                // TODO:
                // * Only hex digits or underscores allowed
                // * Max 6 chars
                // * Within allowed range (must be at most 10FFFF)
            }
            // Code points are always valid
            CodePoint => (),
        }
    }

    if !components.has_closing_quote {
        errors.push(SyntaxError::new(UnclosedChar, node.syntax().range()));
    }

    if len == 0 {
        errors.push(SyntaxError::new(EmptyChar, node.syntax().range()));
    }

    if len > 1 {
        errors.push(SyntaxError::new(LongChar, node.syntax().range()));
    }
}

fn is_ascii_escape(code: char) -> bool {
    match code {
        '\'' | '"' | 'n' | 'r' | 't' | '0' => true,
        _ => false,
    }
}
