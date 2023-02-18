use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::ast;
use rustc_ast::{
    token::{Token, TokenKind},
    tokenstream::TokenTree,
};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks assertions without a custom panic message.
    ///
    /// ### Why is this bad?
    /// Without a good custom message, it'd be hard to understand what went wrong when the assertion fails.
    /// A good custom message should be more about why the failure of the assertion is problematic
    /// and not what is failed because the assertion already conveys that.
    ///
    /// ### Example
    /// ```rust
    /// # struct Service { ready: bool }
    /// fn call(service: Service) {
    ///     assert!(service.ready);
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # struct Service { ready: bool }
    /// fn call(service: Service) {
    ///     assert!(service.ready, "`service.poll_ready()` must be called first to ensure that service is ready to receive requests");
    /// }
    /// ```
    #[clippy::version = "1.69.0"]
    pub MISSING_ASSERT_MESSAGE,
    pedantic,
    "checks assertions without a custom panic message"
}

#[derive(Default, Clone, Debug)]
pub struct MissingAssertMessage {
    // This field will be greater than zero if we are inside a `#[test]` or `#[cfg(test)]`
    test_deepnes: usize,
}

impl_lint_pass!(MissingAssertMessage => [MISSING_ASSERT_MESSAGE]);

impl EarlyLintPass for MissingAssertMessage {
    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac_call: &ast::MacCall) {
        if self.test_deepnes != 0 {
            return;
        }

        let Some(last_segment) = mac_call.path.segments.last() else { return; };
        let num_separators_needed = match last_segment.ident.as_str() {
            "assert" | "debug_assert" => 1,
            "assert_eq" | "assert_ne" | "debug_assert_eq" | "debug_assert_ne" => 2,
            _ => return,
        };
        let num_separators = num_commas_on_arguments(mac_call);

        if num_separators < num_separators_needed {
            span_lint_and_help(
                cx,
                MISSING_ASSERT_MESSAGE,
                mac_call.span(),
                "assert without any message",
                None,
                "consider describing why the failing assert is problematic",
            );
        }
    }

    fn check_item(&mut self, _: &EarlyContext<'_>, item: &ast::Item) {
        if item.attrs.iter().any(is_a_test_attribute) {
            self.test_deepnes += 1;
        }
    }

    fn check_item_post(&mut self, _: &EarlyContext<'_>, item: &ast::Item) {
        if item.attrs.iter().any(is_a_test_attribute) {
            self.test_deepnes -= 1;
        }
    }
}

// Returns number of commas (excluding trailing comma) from `MacCall`'s arguments.
fn num_commas_on_arguments(mac_call: &ast::MacCall) -> usize {
    let mut num_separators = 0;
    let mut is_trailing = false;
    for tt in mac_call.args.tokens.trees() {
        match tt {
            TokenTree::Token(
                Token {
                    kind: TokenKind::Comma,
                    span: _,
                },
                _,
            ) => {
                num_separators += 1;
                is_trailing = true;
            },
            _ => {
                is_trailing = false;
            },
        }
    }
    if is_trailing {
        num_separators -= 1;
    }
    num_separators
}

// Returns true if the attribute is either a `#[test]` or a `#[cfg(test)]`.
fn is_a_test_attribute(attr: &ast::Attribute) -> bool {
    if attr.has_name(sym::test) {
        return true;
    }

    if attr.has_name(sym::cfg)
        && let Some(items) = attr.meta_item_list()
        && let [item] = &*items
        && item.has_name(sym::test)
    {
        true
    } else {
        false
    }
}
