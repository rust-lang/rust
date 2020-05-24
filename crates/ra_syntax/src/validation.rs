//! FIXME: write short doc here

mod block;

use crate::{
    ast, match_ast, AstNode, SyntaxError,
    SyntaxKind::{BYTE, BYTE_STRING, CHAR, CONST_DEF, FN_DEF, INT_NUMBER, STRING, TYPE_ALIAS_DEF},
    SyntaxNode, SyntaxToken, TextSize, T,
};
use rustc_lexer::unescape::{
    self, unescape_byte, unescape_byte_literal, unescape_char, unescape_literal, Mode,
};
use std::convert::TryFrom;

fn rustc_unescape_error_to_string(err: unescape::EscapeError) -> &'static str {
    use unescape::EscapeError as EE;

    #[rustfmt::skip]
    let err_message = match err {
        EE::ZeroChars => {
            "Literal must not be empty"
        }
        EE::MoreThanOneChar => {
            "Literal must be one character long"
        }
        EE::LoneSlash => {
            "Character must be escaped: `\\`"
        }
        EE::InvalidEscape => {
            "Invalid escape"
        }
        EE::BareCarriageReturn | EE::BareCarriageReturnInRawString => {
            "Character must be escaped: `\r`"
        }
        EE::EscapeOnlyChar => {
            "Escape character `\\` must be escaped itself"
        }
        EE::TooShortHexEscape => {
            "ASCII hex escape code must have exactly two digits"
        }
        EE::InvalidCharInHexEscape => {
            "ASCII hex escape code must contain only hex characters"
        }
        EE::OutOfRangeHexEscape => {
            "ASCII hex escape code must be at most 0x7F"
        }
        EE::NoBraceInUnicodeEscape => {
            "Missing `{` to begin the unicode escape"
        }
        EE::InvalidCharInUnicodeEscape => {
            "Unicode escape must contain only hex characters and underscores"
        }
        EE::EmptyUnicodeEscape => {
            "Unicode escape must not be empty"
        }
        EE::UnclosedUnicodeEscape => {
            "Missing `}` to terminate the unicode escape"
        }
        EE::LeadingUnderscoreUnicodeEscape => {
            "Unicode escape code must not begin with an underscore"
        }
        EE::OverlongUnicodeEscape => {
            "Unicode escape code must have at most 6 digits"
        }
        EE::LoneSurrogateUnicodeEscape => {
            "Unicode escape code must not be a surrogate"
        }
        EE::OutOfRangeUnicodeEscape => {
            "Unicode escape code must be at most 0x10FFFF"
        }
        EE::UnicodeEscapeInByte => {
            "Byte literals must not contain unicode escapes"
        }
        EE::NonAsciiCharInByte | EE::NonAsciiCharInByteString => {
            "Byte literals must not contain non-ASCII characters"
        }
    };

    err_message
}

pub(crate) fn validate(root: &SyntaxNode) -> Vec<SyntaxError> {
    // FIXME:
    // * Add unescape validation of raw string literals and raw byte string literals
    // * Add validation of doc comments are being attached to nodes

    let mut errors = Vec::new();
    for node in root.descendants() {
        match_ast! {
            match node {
                ast::Literal(it) => validate_literal(it, &mut errors),
                ast::BlockExpr(it) => block::validate_block_expr(it, &mut errors),
                ast::FieldExpr(it) => validate_numeric_name(it.name_ref(), &mut errors),
                ast::RecordField(it) => validate_numeric_name(it.name_ref(), &mut errors),
                ast::Visibility(it) => validate_visibility(it, &mut errors),
                ast::RangeExpr(it) => validate_range_expr(it, &mut errors),
                ast::PathSegment(it) => validate_path_keywords(it, &mut errors),
                _ => (),
            }
        }
    }
    errors
}

fn validate_literal(literal: ast::Literal, acc: &mut Vec<SyntaxError>) {
    // FIXME: move this function to outer scope (https://github.com/rust-analyzer/rust-analyzer/pull/2834#discussion_r366196658)
    fn unquote(text: &str, prefix_len: usize, end_delimiter: char) -> Option<&str> {
        text.rfind(end_delimiter).and_then(|end| text.get(prefix_len..end))
    }

    let token = literal.token();
    let text = token.text().as_str();

    // FIXME: lift this lambda refactor to `fn` (https://github.com/rust-analyzer/rust-analyzer/pull/2834#discussion_r366199205)
    let mut push_err = |prefix_len, (off, err): (usize, unescape::EscapeError)| {
        let off = token.text_range().start() + TextSize::try_from(off + prefix_len).unwrap();
        acc.push(SyntaxError::new_at_offset(rustc_unescape_error_to_string(err), off));
    };

    match token.kind() {
        BYTE => {
            if let Some(Err(e)) = unquote(text, 2, '\'').map(unescape_byte) {
                push_err(2, e);
            }
        }
        CHAR => {
            if let Some(Err(e)) = unquote(text, 1, '\'').map(unescape_char) {
                push_err(1, e);
            }
        }
        BYTE_STRING => {
            if let Some(without_quotes) = unquote(text, 2, '"') {
                unescape_byte_literal(without_quotes, Mode::ByteStr, &mut |range, char| {
                    if let Err(err) = char {
                        push_err(2, (range.start, err));
                    }
                })
            }
        }
        STRING => {
            if let Some(without_quotes) = unquote(text, 1, '"') {
                unescape_literal(without_quotes, Mode::Str, &mut |range, char| {
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
                "Tuple (struct) field access is only allowed through \
                decimal integers with no underscores or suffix",
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

    let impl_def = match parent.parent().and_then(|it| it.parent()).and_then(ast::ImplDef::cast) {
        Some(it) => it,
        None => return,
    };
    if impl_def.target_trait().is_some() {
        errors.push(SyntaxError::new("Unnecessary visibility qualifier", vis.syntax.text_range()));
    }
}

fn validate_range_expr(expr: ast::RangeExpr, errors: &mut Vec<SyntaxError>) {
    if expr.op_kind() == Some(ast::RangeOp::Inclusive) && expr.end().is_none() {
        errors.push(SyntaxError::new(
            "An inclusive range must have an end expression",
            expr.syntax().text_range(),
        ));
    }
}

fn validate_path_keywords(segment: ast::PathSegment, errors: &mut Vec<SyntaxError>) {
    use ast::PathSegmentKind;

    let path = segment.parent_path();
    let is_path_start = segment.coloncolon_token().is_none() && path.qualifier().is_none();

    if let Some(token) = segment.self_token() {
        if !is_path_start {
            errors.push(SyntaxError::new(
                "The `self` keyword is only allowed as the first segment of a path",
                token.text_range(),
            ));
        }
    } else if let Some(token) = segment.crate_token() {
        if !is_path_start || use_prefix(path).is_some() {
            errors.push(SyntaxError::new(
                "The `crate` keyword is only allowed as the first segment of a path",
                token.text_range(),
            ));
        }
    } else if let Some(token) = segment.super_token() {
        if !all_supers(&path) {
            errors.push(SyntaxError::new(
                "The `super` keyword may only be preceded by other `super`s",
                token.text_range(),
            ));
            return;
        }

        let mut curr_path = path;
        while let Some(prefix) = use_prefix(curr_path) {
            if !all_supers(&prefix) {
                errors.push(SyntaxError::new(
                    "The `super` keyword may only be preceded by other `super`s",
                    token.text_range(),
                ));
                return;
            }
            curr_path = prefix;
        }
    }

    fn use_prefix(mut path: ast::Path) -> Option<ast::Path> {
        for node in path.syntax().ancestors().skip(1) {
            match_ast! {
                match node {
                    ast::UseTree(it) => if let Some(tree_path) = it.path() {
                        // Even a top-level path exists within a `UseTree` so we must explicitly
                        // allow our path but disallow anything else
                        if tree_path != path {
                            return Some(tree_path);
                        }
                    },
                    ast::UseTreeList(_it) => continue,
                    ast::Path(parent) => path = parent,
                    _ => return None,
                }
            };
        }
        return None;
    }

    fn all_supers(path: &ast::Path) -> bool {
        let segment = match path.segment() {
            Some(it) => it,
            None => return false,
        };

        if segment.kind() != Some(PathSegmentKind::SuperKw) {
            return false;
        }

        if let Some(ref subpath) = path.qualifier() {
            return all_supers(subpath);
        }

        return true;
    }
}
