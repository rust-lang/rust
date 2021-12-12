//! Lexing, bridging to parser (which does the actual parsing) and
//! incremental reparsing.

pub(crate) mod lexer;
mod text_tree_sink;
mod reparsing;

use parser::SyntaxKind;
use text_tree_sink::TextTreeSink;

use crate::{syntax_node::GreenNode, AstNode, SyntaxError, SyntaxNode};

pub(crate) use crate::parsing::{lexer::*, reparsing::incremental_reparse};

pub(crate) fn parse_text(text: &str) -> (GreenNode, Vec<SyntaxError>) {
    let (lexer_tokens, lexer_errors) = tokenize(text);
    let parser_tokens = to_parser_tokens(text, &lexer_tokens);

    let mut tree_sink = TextTreeSink::new(text, &lexer_tokens);

    parser::parse_source_file(&parser_tokens, &mut tree_sink);

    let (tree, mut parser_errors) = tree_sink.finish();
    parser_errors.extend(lexer_errors);

    (tree, parser_errors)
}

/// Returns `text` parsed as a `T` provided there are no parse errors.
pub(crate) fn parse_text_as<T: AstNode>(
    text: &str,
    entry_point: parser::ParserEntryPoint,
) -> Result<T, ()> {
    let (lexer_tokens, lexer_errors) = tokenize(text);
    if !lexer_errors.is_empty() {
        return Err(());
    }

    let parser_tokens = to_parser_tokens(text, &lexer_tokens);

    let mut tree_sink = TextTreeSink::new(text, &lexer_tokens);

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

pub(crate) fn to_parser_tokens(text: &str, lexer_tokens: &[lexer::Token]) -> ::parser::Tokens {
    let mut off = 0;
    let mut res = parser::Tokens::default();
    let mut was_joint = true;
    for t in lexer_tokens {
        if t.kind.is_trivia() {
            was_joint = false;
        } else if t.kind == SyntaxKind::IDENT {
            let token_text = &text[off..][..usize::from(t.len)];
            let contextual_kw =
                SyntaxKind::from_contextual_keyword(token_text).unwrap_or(SyntaxKind::IDENT);
            res.push_ident(contextual_kw);
        } else {
            res.was_joint(was_joint);
            res.push(t.kind);
            was_joint = true;
        }
        off += usize::from(t.len);
    }
    res
}
