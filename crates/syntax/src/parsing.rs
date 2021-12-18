//! Lexing, bridging to parser (which does the actual parsing) and
//! incremental reparsing.

mod text_tree_sink;
mod reparsing;

use parser::SyntaxKind;
use text_tree_sink::TextTreeSink;

use crate::{syntax_node::GreenNode, AstNode, SyntaxError, SyntaxNode};

pub(crate) use crate::parsing::reparsing::incremental_reparse;

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let lexed = parser::LexedStr::new(text);
    let parser_tokens = lexed.to_tokens();

    let mut tree_sink = TextTreeSink::new(lexed);

    parser::parse_source_file(&parser_tokens, &mut tree_sink);

    let (tree, parser_errors) = tree_sink.finish();

    (tree, parser_errors)
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
    let parser_tokens = lexed.to_tokens();

    let mut tree_sink = TextTreeSink::new(lexed);

    // TextTreeSink assumes that there's at least some root node to which it can attach errors and
    // tokens. We arbitrarily give it a SourceFile.
    use parser::TreeSink;
    tree_sink.start_node(SyntaxKind::SOURCE_FILE);
    parser::parse(&parser_tokens, &mut tree_sink, entry_point);
    tree_sink.finish_node();

    let (tree, parser_errors, eof) = tree_sink.finish_eof();
    if !parser_errors.is_empty() || !eof {
        return Err(());
    }

    SyntaxNode::new_root(tree).first_child().and_then(T::cast).ok_or(())
}
