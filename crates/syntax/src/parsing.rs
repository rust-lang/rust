//! Lexing, bridging to parser (which does the actual parsing) and
//! incremental reparsing.

mod text_tree_sink;
mod reparsing;

use crate::{
    parsing::text_tree_sink::build_tree, syntax_node::GreenNode, AstNode, SyntaxError, SyntaxNode,
};

pub(crate) use crate::parsing::reparsing::incremental_reparse;

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let lexed = parser::LexedStr::new(text);
    let parser_input = lexed.to_input();
    let parser_output = parser::parse_source_file(&parser_input);
    let (node, errors, _eof) = build_tree(lexed, parser_output, false);
    (node, errors)
}

/// Returns `text` parsed as a `T` provided there are no parse errors.
pub(crate) fn parse_text_as<T: AstNode>(
    text: &str,
    entry_point: parser::ParserEntryPoint,
) -> Result<T, ()> {
    let lexed = parser::LexedStr::new(text);
    if lexed.errors().next().is_some() {
        return Err(());
    }
    let parser_input = lexed.to_input();
    let parser_output = parser::parse(&parser_input, entry_point);
    let (node, errors, eof) = build_tree(lexed, parser_output, true);

    if !errors.is_empty() || !eof {
        return Err(());
    }

    SyntaxNode::new_root(node).first_child().and_then(T::cast).ok_or(())
}
