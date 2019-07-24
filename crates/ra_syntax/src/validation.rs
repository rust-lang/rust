mod block;
mod field_expr;

use ra_rustc_lexer::unescape;

use crate::{
    algo::visit::{visitor_ctx, VisitorCtx},
    ast, SyntaxError, SyntaxErrorKind,
    SyntaxKind::{BYTE, BYTE_STRING, CHAR, STRING},
    SyntaxNode, TextUnit, T,
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

impl From<ra_rustc_lexer::unescape::EscapeError> for EscapeError {
    fn from(err: ra_rustc_lexer::unescape::EscapeError) -> Self {
        match err {
            ra_rustc_lexer::unescape::EscapeError::ZeroChars => EscapeError::ZeroChars,
            ra_rustc_lexer::unescape::EscapeError::MoreThanOneChar => EscapeError::MoreThanOneChar,
            ra_rustc_lexer::unescape::EscapeError::LoneSlash => EscapeError::LoneSlash,
            ra_rustc_lexer::unescape::EscapeError::InvalidEscape => EscapeError::InvalidEscape,
            ra_rustc_lexer::unescape::EscapeError::BareCarriageReturn
            | ra_rustc_lexer::unescape::EscapeError::BareCarriageReturnInRawString => {
                EscapeError::BareCarriageReturn
            }
            ra_rustc_lexer::unescape::EscapeError::EscapeOnlyChar => EscapeError::EscapeOnlyChar,
            ra_rustc_lexer::unescape::EscapeError::TooShortHexEscape => {
                EscapeError::TooShortHexEscape
            }
            ra_rustc_lexer::unescape::EscapeError::InvalidCharInHexEscape => {
                EscapeError::InvalidCharInHexEscape
            }
            ra_rustc_lexer::unescape::EscapeError::OutOfRangeHexEscape => {
                EscapeError::OutOfRangeHexEscape
            }
            ra_rustc_lexer::unescape::EscapeError::NoBraceInUnicodeEscape => {
                EscapeError::NoBraceInUnicodeEscape
            }
            ra_rustc_lexer::unescape::EscapeError::InvalidCharInUnicodeEscape => {
                EscapeError::InvalidCharInUnicodeEscape
            }
            ra_rustc_lexer::unescape::EscapeError::EmptyUnicodeEscape => {
                EscapeError::EmptyUnicodeEscape
            }
            ra_rustc_lexer::unescape::EscapeError::UnclosedUnicodeEscape => {
                EscapeError::UnclosedUnicodeEscape
            }
            ra_rustc_lexer::unescape::EscapeError::LeadingUnderscoreUnicodeEscape => {
                EscapeError::LeadingUnderscoreUnicodeEscape
            }
            ra_rustc_lexer::unescape::EscapeError::OverlongUnicodeEscape => {
                EscapeError::OverlongUnicodeEscape
            }
            ra_rustc_lexer::unescape::EscapeError::LoneSurrogateUnicodeEscape => {
                EscapeError::LoneSurrogateUnicodeEscape
            }
            ra_rustc_lexer::unescape::EscapeError::OutOfRangeUnicodeEscape => {
                EscapeError::OutOfRangeUnicodeEscape
            }
            ra_rustc_lexer::unescape::EscapeError::UnicodeEscapeInByte => {
                EscapeError::UnicodeEscapeInByte
            }
            ra_rustc_lexer::unescape::EscapeError::NonAsciiCharInByte
            | ra_rustc_lexer::unescape::EscapeError::NonAsciiCharInByteString => {
                EscapeError::NonAsciiCharInByte
            }
        }
    }
}

impl From<ra_rustc_lexer::unescape::EscapeError> for SyntaxErrorKind {
    fn from(err: ra_rustc_lexer::unescape::EscapeError) -> Self {
        SyntaxErrorKind::EscapeError(err.into())
    }
}

pub(crate) fn validate(root: &SyntaxNode) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in root.descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Literal, _>(validate_literal)
            .visit::<ast::Block, _>(block::validate_block_node)
            .visit::<ast::FieldExpr, _>(field_expr::validate_field_expr_node)
            .accept(&node);
    }
    errors
}

// FIXME: kill duplication
fn validate_literal(literal: ast::Literal, acc: &mut Vec<SyntaxError>) {
    let token = literal.token();
    let text = token.text().as_str();
    match token.kind() {
        BYTE => {
            if let Some(end) = text.rfind('\'') {
                if let Some(without_quotes) = text.get(2..end) {
                    if let Err((off, err)) = unescape::unescape_byte(without_quotes) {
                        let off = token.text_range().start() + TextUnit::from_usize(off + 2);
                        acc.push(SyntaxError::new(err.into(), off))
                    }
                }
            }
        }
        CHAR => {
            if let Some(end) = text.rfind('\'') {
                if let Some(without_quotes) = text.get(1..end) {
                    if let Err((off, err)) = unescape::unescape_char(without_quotes) {
                        let off = token.text_range().start() + TextUnit::from_usize(off + 1);
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
                            let off = token.text_range().start() + TextUnit::from_usize(off + 2);
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
                            let off = token.text_range().start() + TextUnit::from_usize(off + 1);
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
