//! Lexing, bridging to parser (which does the actual parsing) and
//! incremental reparsing.

mod reparsing;

use rowan::TextRange;

use crate::{syntax_node::GreenNode, AstNode, SyntaxError, SyntaxNode, SyntaxTreeBuilder};

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

pub(crate) fn build_tree(
    lexed: parser::LexedStr<'_>,
    parser_output: parser::Output,
    synthetic_root: bool,
) -> (GreenNode, Vec<SyntaxError>, bool) {
    let mut builder = SyntaxTreeBuilder::default();

    let is_eof = lexed.intersperse_trivia(&parser_output, synthetic_root, &mut |step| match step {
        parser::StrStep::Token { kind, text } => builder.token(kind, text),
        parser::StrStep::Enter { kind } => builder.start_node(kind),
        parser::StrStep::Exit => builder.finish_node(),
        parser::StrStep::Error { msg, pos } => {
            builder.error(msg.to_string(), pos.try_into().unwrap())
        }
    });

    let (node, mut errors) = builder.finish_raw();
    for (i, err) in lexed.errors() {
        let text_range = lexed.text_range(i);
        let text_range = TextRange::new(
            text_range.start.try_into().unwrap(),
            text_range.end.try_into().unwrap(),
        );
        errors.push(SyntaxError::new(err, text_range))
    }

    (node, errors, is_eof)
}
