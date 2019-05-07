mod unescape;

mod block;
mod field_expr;

use crate::{
    SourceFile, SyntaxError, AstNode, SyntaxNode, TextUnit,
    SyntaxKind::{L_CURLY, R_CURLY, BYTE, BYTE_STRING, STRING, CHAR},
    ast,
    algo::visit::{visitor_ctx, VisitorCtx},
};

pub(crate) use unescape::EscapeError;

pub(crate) fn validate(file: &SourceFile) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in file.syntax().descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Literal, _>(validate_literal)
            .visit::<ast::Block, _>(block::validate_block_node)
            .visit::<ast::FieldExpr, _>(field_expr::validate_field_expr_node)
            .accept(node);
    }
    errors
}

// FIXME: kill duplication
fn validate_literal(literal: &ast::Literal, acc: &mut Vec<SyntaxError>) {
    let token = literal.token();
    let text = token.text().as_str();
    match token.kind() {
        BYTE => {
            if let Some(end) = text.rfind('\'') {
                if let Some(without_quotes) = text.get(2..end) {
                    if let Err((off, err)) = unescape::unescape_byte(without_quotes) {
                        let off = token.range().start() + TextUnit::from_usize(off + 2);
                        acc.push(SyntaxError::new(err.into(), off))
                    }
                }
            }
        }
        CHAR => {
            if let Some(end) = text.rfind('\'') {
                if let Some(without_quotes) = text.get(1..end) {
                    if let Err((off, err)) = unescape::unescape_char(without_quotes) {
                        let off = token.range().start() + TextUnit::from_usize(off + 1);
                        acc.push(SyntaxError::new(err.into(), off))
                    }
                }
            }
        }
        BYTE_STRING => {
            if let Some(end) = text.rfind('\"') {
                if let Some(without_quotes) = text.get(2..end) {
                    unescape::unescape_byte_str(without_quotes, &mut |range, char| {
                        if let Err(err) = char {
                            let off = range.start;
                            let off = token.range().start() + TextUnit::from_usize(off + 2);
                            acc.push(SyntaxError::new(err.into(), off))
                        }
                    })
                }
            }
        }
        STRING => {
            if let Some(end) = text.rfind('\"') {
                if let Some(without_quotes) = text.get(1..end) {
                    unescape::unescape_str(without_quotes, &mut |range, char| {
                        if let Err(err) = char {
                            let off = range.start;
                            let off = token.range().start() + TextUnit::from_usize(off + 1);
                            acc.push(SyntaxError::new(err.into(), off))
                        }
                    })
                }
            }
        }
        _ => (),
    }
}

pub(crate) fn validate_block_structure(root: &SyntaxNode) {
    let mut stack = Vec::new();
    for node in root.descendants() {
        match node.kind() {
            L_CURLY => stack.push(node),
            R_CURLY => {
                if let Some(pair) = stack.pop() {
                    assert_eq!(
                        node.parent(),
                        pair.parent(),
                        "\nunpaired curleys:\n{}\n{}\n",
                        root.text(),
                        root.debug_dump(),
                    );
                    assert!(
                        node.next_sibling().is_none() && pair.prev_sibling().is_none(),
                        "\nfloating curlys at {:?}\nfile:\n{}\nerror:\n{}\n",
                        node,
                        root.text(),
                        node.text(),
                    );
                }
            }
            _ => (),
        }
    }
}
