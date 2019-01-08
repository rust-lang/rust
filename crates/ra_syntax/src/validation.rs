mod byte;
mod byte_string;
mod char;
mod string;

use crate::{
    SourceFile, yellow::SyntaxError, AstNode,
    ast,
    algo::visit::{visitor_ctx, VisitorCtx},
};

pub(crate) fn validate(file: &SourceFile) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in file.syntax().descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Byte, _>(self::byte::validate_byte_node)
            .visit::<ast::ByteString, _>(self::byte_string::validate_byte_string_node)
            .visit::<ast::Char, _>(self::char::validate_char_node)
            .visit::<ast::String, _>(self::string::validate_string_node)
            .accept(node);
    }
    errors
}
