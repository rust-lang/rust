use crate::{
    ast::{self, AstNode},
    File,
    string_lexing,
    yellow::{
        SyntaxError,
    },
};

pub(crate) fn validate(file: &File) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for d in file.root.borrowed().descendants() {
        if let Some(c) = ast::Char::cast(d) {
            let components = &mut string_lexing::parse_char_literal(c.text());
            let len = components.count();

            if !components.has_closing_quote {
                errors.push(SyntaxError {
                    msg: "Unclosed char literal".to_string(),
                    offset: d.range().start(),
                });
            }

            if len == 0 {
                errors.push(SyntaxError {
                    msg: "Empty char literal".to_string(),
                    offset: d.range().start(),
                });
            }

            if len > 1 {
                errors.push(SyntaxError {
                    msg: "Character literal should be only one character long".to_string(),
                    offset: d.range().start(),
                });
            }
        }
    }
    errors
}
