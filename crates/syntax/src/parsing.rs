//! Lexing, bridging to parser (which does the actual parsing) and
//! incremental reparsing.

pub(crate) mod lexer;
mod text_token_source;
mod text_tree_sink;
mod reparsing;

use parser::SyntaxKind;
use text_token_source::TextTokenSource;
use text_tree_sink::TextTreeSink;

use crate::{syntax_node::GreenNode, AstNode, SyntaxError, SyntaxNode};

pub(crate) use crate::parsing::{lexer::*, reparsing::incremental_reparse};

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let (tokens, lexer_errors) = tokenize(&text);

    let mut token_source = TextTokenSource::new(text, &tokens);
    let mut tree_sink = TextTreeSink::new(text, &tokens);

    parser::parse(&mut token_source, &mut tree_sink);

    let (tree, mut parser_errors) = tree_sink.finish();
    parser_errors.extend(lexer_errors);

    (tree, parser_errors)
}

/// Returns `text` parsed as a `T` provided there are no parse errors.
pub(crate) fn parse_text_fragment<T: AstNode>(
    text: &str,
    fragment_kind: parser::FragmentKind,
) -> Result<T, ()> {
    let (tokens, lexer_errors) = tokenize(&text);
    if !lexer_errors.is_empty() {
        return Err(());
    }

    let mut token_source = TextTokenSource::new(text, &tokens);
    let mut tree_sink = TextTreeSink::new(text, &tokens);

    // TextTreeSink assumes that there's at least some root node to which it can attach errors and
    // tokens. We arbitrarily give it a SourceFile.
    use parser::TreeSink;
    tree_sink.start_node(SyntaxKind::SOURCE_FILE);
    parser::parse_fragment(&mut token_source, &mut tree_sink, fragment_kind);
    tree_sink.finish_node();

    let (tree, parser_errors) = tree_sink.finish();
    use parser::TokenSource;
    if !parser_errors.is_empty() || token_source.current().kind != SyntaxKind::EOF {
        return Err(());
    }

    SyntaxNode::new_root(tree).first_child().and_then(T::cast).ok_or(())
}
