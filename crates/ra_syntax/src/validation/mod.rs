use crate::{
    algo::visit::{visitor_ctx, VisitorCtx},
    ast,
    SourceFileNode,
    yellow::SyntaxError,
};

mod char;
mod string;

pub(crate) fn validate(file: &SourceFileNode) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in file.syntax().descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Char, _>(self::char::validate_char_node)
            .visit::<ast::String, _>(self::string::validate_string_node)
            .accept(node);
    }
    errors
}
