//! FIXME: write short doc here

mod block;

use rustc_lexer::unescape;

use crate::{
    ast, match_ast, AstNode, SyntaxError, SyntaxErrorKind,
    SyntaxKind::{BYTE, BYTE_STRING, CHAR, CONST_DEF, FN_DEF, INT_NUMBER, STRING, TYPE_ALIAS_DEF},
    SyntaxNode, SyntaxToken, TextUnit, T,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EscapeError {
    ZeroChars,
    MoreThanOneChar,
    LoneSlash,
    InvalidEscape,
    BareCarriageReturn,
    EscapeOnlyChar,
    TooShortHexEscape,
    InvalidCharInHexEscape,
    OutOfRangeHexEscape,
    NoBraceInUnicodeEscape,
    InvalidCharInUnicodeEscape,
    EmptyUnicodeEscape,
    UnclosedUnicodeEscape,
    LeadingUnderscoreUnicodeEscape,
    OverlongUnicodeEscape,
    LoneSurrogateUnicodeEscape,
    OutOfRangeUnicodeEscape,
    UnicodeEscapeInByte,
    NonAsciiCharInByte,
}

impl From<rustc_lexer::unescape::EscapeError> for EscapeError {
    fn from(err: rustc_lexer::unescape::EscapeError) -> Self {
        match err {
            rustc_lexer::unescape::EscapeError::ZeroChars => EscapeError::ZeroChars,
            rustc_lexer::unescape::EscapeError::MoreThanOneChar => EscapeError::MoreThanOneChar,
            rustc_lexer::unescape::EscapeError::LoneSlash => EscapeError::LoneSlash,
            rustc_lexer::unescape::EscapeError::InvalidEscape => EscapeError::InvalidEscape,
            rustc_lexer::unescape::EscapeError::BareCarriageReturn
            | rustc_lexer::unescape::EscapeError::BareCarriageReturnInRawString => {
                EscapeError::BareCarriageReturn
            }
            rustc_lexer::unescape::EscapeError::EscapeOnlyChar => EscapeError::EscapeOnlyChar,
            rustc_lexer::unescape::EscapeError::TooShortHexEscape => EscapeError::TooShortHexEscape,
            rustc_lexer::unescape::EscapeError::InvalidCharInHexEscape => {
                EscapeError::InvalidCharInHexEscape
            }
            rustc_lexer::unescape::EscapeError::OutOfRangeHexEscape => {
                EscapeError::OutOfRangeHexEscape
            }
            rustc_lexer::unescape::EscapeError::NoBraceInUnicodeEscape => {
                EscapeError::NoBraceInUnicodeEscape
            }
            rustc_lexer::unescape::EscapeError::InvalidCharInUnicodeEscape => {
                EscapeError::InvalidCharInUnicodeEscape
            }
            rustc_lexer::unescape::EscapeError::EmptyUnicodeEscape => {
                EscapeError::EmptyUnicodeEscape
            }
            rustc_lexer::unescape::EscapeError::UnclosedUnicodeEscape => {
                EscapeError::UnclosedUnicodeEscape
            }
            rustc_lexer::unescape::EscapeError::LeadingUnderscoreUnicodeEscape => {
                EscapeError::LeadingUnderscoreUnicodeEscape
            }
            rustc_lexer::unescape::EscapeError::OverlongUnicodeEscape => {
                EscapeError::OverlongUnicodeEscape
            }
            rustc_lexer::unescape::EscapeError::LoneSurrogateUnicodeEscape => {
                EscapeError::LoneSurrogateUnicodeEscape
            }
            rustc_lexer::unescape::EscapeError::OutOfRangeUnicodeEscape => {
                EscapeError::OutOfRangeUnicodeEscape
            }
            rustc_lexer::unescape::EscapeError::UnicodeEscapeInByte => {
                EscapeError::UnicodeEscapeInByte
            }
            rustc_lexer::unescape::EscapeError::NonAsciiCharInByte
            | rustc_lexer::unescape::EscapeError::NonAsciiCharInByteString => {
                EscapeError::NonAsciiCharInByte
            }
        }
    }
}

impl From<rustc_lexer::unescape::EscapeError> for SyntaxErrorKind {
    fn from(err: rustc_lexer::unescape::EscapeError) -> Self {
        SyntaxErrorKind::EscapeError(err.into())
    }
}

pub(crate) fn validate(root: &SyntaxNode) -> Vec<SyntaxError> {
    // FIXME:
    // * Add validation of character literal containing only a single char
    // * Add validation of `crate` keyword not appearing in the middle of the symbol path
    // * Add validation of doc comments are being attached to nodes
    // * Remove validation of unterminated literals (it is already implemented in `tokenize()`)

    let mut errors = Vec::new();
    for node in root.descendants() {
        match_ast! {
            match node {
                ast::Literal(it) => { validate_literal(it, &mut errors) },
                ast::BlockExpr(it) => { block::validate_block_expr(it, &mut errors) },
                ast::FieldExpr(it) => { validate_numeric_name(it.name_ref(), &mut errors) },
                ast::RecordField(it) => { validate_numeric_name(it.name_ref(), &mut errors) },
                ast::Visibility(it) => { validate_visibility(it, &mut errors) },
                ast::RangeExpr(it) => { validate_range_expr(it, &mut errors) },
                _ => (),
            }
        }
    }
    errors
}

fn validate_literal(literal: ast::Literal, acc: &mut Vec<SyntaxError>) {
    fn unquote(text: &str, prefix_len: usize, end_delimiter: char) -> Option<&str> {
        text.rfind(end_delimiter).and_then(|end| text.get(prefix_len..end))
    }

    let token = literal.token();
    let text = token.text().as_str();

    let mut push_err = |prefix_len, (off, err): (usize, unescape::EscapeError)| {
        let off = token.text_range().start() + TextUnit::from_usize(off + prefix_len);
        acc.push(SyntaxError::new(err.into(), off));
    };

    match token.kind() {
        BYTE => {
            if let Some(Err(e)) = unquote(text, 2, '\'').map(unescape::unescape_byte) {
                push_err(2, e);
            }
        }
        CHAR => {
            if let Some(Err(e)) = unquote(text, 1, '\'').map(unescape::unescape_char) {
                push_err(1, e);
            }
        }
        BYTE_STRING => {
            if let Some(without_quotes) = unquote(text, 2, '"') {
                unescape::unescape_byte_str(without_quotes, &mut |range, char| {
                    if let Err(err) = char {
                        push_err(2, (range.start, err));
                    }
                })
            }
        }
        STRING => {
            if let Some(without_quotes) = unquote(text, 1, '"') {
                unescape::unescape_str(without_quotes, &mut |range, char| {
                    if let Err(err) = char {
                        push_err(1, (range.start, err));
                    }
                })
            }
        }
        _ => (),
    }
}

pub(crate) fn validate_block_structure(root: &SyntaxNode) {
    let mut stack = Vec::new();
    for node in root.descendants() {
        match node.kind() {
            T!['{'] => stack.push(node),
            T!['}'] => {
                if let Some(pair) = stack.pop() {
                    assert_eq!(
                        node.parent(),
                        pair.parent(),
                        "\nunpaired curleys:\n{}\n{:#?}\n",
                        root.text(),
                        root,
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

fn validate_numeric_name(name_ref: Option<ast::NameRef>, errors: &mut Vec<SyntaxError>) {
    if let Some(int_token) = int_token(name_ref) {
        if int_token.text().chars().any(|c| !c.is_digit(10)) {
            errors.push(SyntaxError::new(
                SyntaxErrorKind::InvalidTupleIndexFormat,
                int_token.text_range(),
            ));
        }
    }

    fn int_token(name_ref: Option<ast::NameRef>) -> Option<SyntaxToken> {
        name_ref?.syntax().first_child_or_token()?.into_token().filter(|it| it.kind() == INT_NUMBER)
    }
}

fn validate_visibility(vis: ast::Visibility, errors: &mut Vec<SyntaxError>) {
    let parent = match vis.syntax().parent() {
        Some(it) => it,
        None => return,
    };
    match parent.kind() {
        FN_DEF | CONST_DEF | TYPE_ALIAS_DEF => (),
        _ => return,
    }
    let impl_block = match parent.parent().and_then(|it| it.parent()).and_then(ast::ImplBlock::cast)
    {
        Some(it) => it,
        None => return,
    };
    if impl_block.target_trait().is_some() {
        errors
            .push(SyntaxError::new(SyntaxErrorKind::VisibilityNotAllowed, vis.syntax.text_range()))
    }
}

fn validate_range_expr(expr: ast::RangeExpr, errors: &mut Vec<SyntaxError>) {
    if expr.op_kind() == Some(ast::RangeOp::Inclusive) && expr.end().is_none() {
        errors.push(SyntaxError::new(
            SyntaxErrorKind::InclusiveRangeMissingEnd,
            expr.syntax().text_range(),
        ));
    }
}
