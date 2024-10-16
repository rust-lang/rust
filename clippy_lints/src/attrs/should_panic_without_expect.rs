use super::{Attribute, SHOULD_PANIC_WITHOUT_EXPECT};
use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::TokenTree;
use rustc_ast::{AttrArgs, AttrKind};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;
use rustc_span::sym;

pub(super) fn check(cx: &EarlyContext<'_>, attr: &Attribute) {
    if let AttrKind::Normal(normal_attr) = &attr.kind {
        if let AttrArgs::Eq { .. } = &normal_attr.item.args {
            // `#[should_panic = ".."]` found, good
            return;
        }

        if let AttrArgs::Delimited(args) = &normal_attr.item.args
            && let mut tt_iter = args.tokens.trees()
            && let Some(TokenTree::Token(
                Token {
                    kind: TokenKind::Ident(sym::expected, _),
                    ..
                },
                _,
            )) = tt_iter.next()
            && let Some(TokenTree::Token(
                Token {
                    kind: TokenKind::Eq, ..
                },
                _,
            )) = tt_iter.next()
            && let Some(TokenTree::Token(
                Token {
                    kind: TokenKind::Literal(_),
                    ..
                },
                _,
            )) = tt_iter.next()
        {
            // `#[should_panic(expected = "..")]` found, good
            return;
        }

        span_lint_and_sugg(
            cx,
            SHOULD_PANIC_WITHOUT_EXPECT,
            attr.span,
            "#[should_panic] attribute without a reason",
            "consider specifying the expected panic",
            "#[should_panic(expected = /* panic message */)]".into(),
            Applicability::HasPlaceholders,
        );
    }
}
