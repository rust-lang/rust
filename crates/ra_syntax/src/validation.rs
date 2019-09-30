//! FIXME: write short doc here

mod block;

use rustc_lexer::unescape;

use crate::{
    algo::visit::{visitor_ctx, VisitorCtx},
    ast, AstNode, SyntaxError, SyntaxErrorKind,
    SyntaxKind::{BYTE, BYTE_STRING, CHAR, INT_NUMBER, STRING},
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
    let mut errors = Vec::new();
    for node in root.descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Literal, _>(validate_literal)
            .visit::<ast::BlockExpr, _>(block::validate_block_expr)
            .visit::<ast::FieldExpr, _>(|it, errors| validate_numeric_name(it.name_ref(), errors))
            .visit::<ast::RecordField, _>(|it, errors| validate_numeric_name(it.name_ref(), errors))
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
