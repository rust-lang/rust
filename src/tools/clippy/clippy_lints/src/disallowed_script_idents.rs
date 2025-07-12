use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint;
use rustc_ast::ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_lint::{EarlyContext, EarlyLintPass, Level, LintContext};
use rustc_session::impl_lint_pass;
use unicode_script::{Script, UnicodeScript};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of unicode scripts other than those explicitly allowed
    /// by the lint config.
    ///
    /// This lint doesn't take into account non-text scripts such as `Unknown` and `Linear_A`.
    /// It also ignores the `Common` script type.
    /// While configuring, be sure to use official script name [aliases] from
    /// [the list of supported scripts][supported_scripts].
    ///
    /// See also: [`non_ascii_idents`].
    ///
    /// [aliases]: http://www.unicode.org/reports/tr24/tr24-31.html#Script_Value_Aliases
    /// [supported_scripts]: https://www.unicode.org/iso15924/iso15924-codes.html
    ///
    /// ### Why restrict this?
    /// It may be not desired to have many different scripts for
    /// identifiers in the codebase.
    ///
    /// Note that if you only want to allow typical English, you might want to use
    /// built-in [`non_ascii_idents`] lint instead.
    ///
    /// [`non_ascii_idents`]: https://doc.rust-lang.org/rustc/lints/listing/allowed-by-default.html#non-ascii-idents
    ///
    /// ### Example
    /// ```no_run
    /// // Assuming that `clippy.toml` contains the following line:
    /// // allowed-scripts = ["Latin", "Cyrillic"]
    /// let counter = 10; // OK, latin is allowed.
    /// let счётчик = 10; // OK, cyrillic is allowed.
    /// let zähler = 10; // OK, it's still latin.
    /// let カウンタ = 10; // Will spawn the lint.
    /// ```
    #[clippy::version = "1.55.0"]
    pub DISALLOWED_SCRIPT_IDENTS,
    restriction,
    "usage of non-allowed Unicode scripts"
}

pub struct DisallowedScriptIdents {
    whitelist: FxHashSet<Script>,
}

impl DisallowedScriptIdents {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            whitelist: conf
                .allowed_scripts
                .iter()
                .map(String::as_str)
                .filter_map(Script::from_full_name)
                .collect(),
        }
    }
}

impl_lint_pass!(DisallowedScriptIdents => [DISALLOWED_SCRIPT_IDENTS]);

impl EarlyLintPass for DisallowedScriptIdents {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &ast::Crate) {
        // Implementation is heavily inspired by the implementation of [`non_ascii_idents`] lint:
        // https://github.com/rust-lang/rust/blob/master/compiler/rustc_lint/src/non_ascii_idents.rs

        let check_disallowed_script_idents = cx.builder.lint_level(DISALLOWED_SCRIPT_IDENTS).level != Level::Allow;
        if !check_disallowed_script_idents {
            return;
        }

        let symbols = cx.sess().psess.symbol_gallery.symbols.lock();
        // Sort by `Span` so that error messages make sense with respect to the
        // order of identifier locations in the code.
        let mut symbols: Vec<_> = symbols.iter().collect();
        symbols.sort_unstable_by_key(|k| k.1);

        for &(symbol, &span) in &symbols {
            // Note: `symbol.as_str()` is an expensive operation, thus should not be called
            // more than once for a single symbol.
            let symbol_str = symbol.as_str();

            // Check if any character in the symbol is not part of any allowed script.
            // Fast path for ascii-only idents.
            if !symbol_str.is_ascii()
                && let Some(script) = symbol_str.chars().find_map(|c| {
                    if c.is_ascii() {
                        return None;
                    }

                    c.script_extension()
                        .iter()
                        .find(|script| !self.whitelist.contains(script))
                })
            {
                span_lint(
                    cx,
                    DISALLOWED_SCRIPT_IDENTS,
                    span,
                    format!(
                        "identifier `{symbol_str}` has a Unicode script that is not allowed by configuration: {}",
                        script.full_name()
                    ),
                );
            }
        }
    }
}
