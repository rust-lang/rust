use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::macros::{FormatArgsStorage, root_macro_call_first_node};
use clippy_utils::{is_in_test, sym};
use rustc_hir::{Expr, Impl, Item, ItemKind, OwnerId};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;

mod empty_string;
mod literal;
mod use_debug;
mod with_newline;

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns when you use `println!("")` to
    /// print a newline.
    ///
    /// ### Why is this bad?
    /// You should use `println!()`, which is simpler.
    ///
    /// ### Example
    /// ```no_run
    /// println!("");
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// println!();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub PRINTLN_EMPTY_STRING,
    style,
    "using `println!(\"\")` with an empty string"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns when you use `print!()` with a format
    /// string that ends in a newline.
    ///
    /// ### Why is this bad?
    /// You should use `println!()` instead, which appends the
    /// newline.
    ///
    /// ### Example
    /// ```no_run
    /// # let name = "World";
    /// print!("Hello {}!\n", name);
    /// ```
    /// use println!() instead
    /// ```no_run
    /// # let name = "World";
    /// println!("Hello {}!", name);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub PRINT_WITH_NEWLINE,
    style,
    "using `print!()` with a format string that ends in a single newline"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for printing on *stdout*. The purpose of this lint
    /// is to catch debugging remnants.
    ///
    /// ### Why restrict this?
    /// People often print on *stdout* while debugging an
    /// application and might forget to remove those prints afterward.
    ///
    /// ### Known problems
    /// Only catches `print!` and `println!` calls.
    ///
    /// ### Example
    /// ```no_run
    /// println!("Hello world!");
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub PRINT_STDOUT,
    restriction,
    "printing on stdout"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for printing on *stderr*. The purpose of this lint
    /// is to catch debugging remnants.
    ///
    /// ### Why restrict this?
    /// People often print on *stderr* while debugging an
    /// application and might forget to remove those prints afterward.
    ///
    /// ### Known problems
    /// Only catches `eprint!` and `eprintln!` calls.
    ///
    /// ### Example
    /// ```no_run
    /// eprintln!("Hello world!");
    /// ```
    #[clippy::version = "1.50.0"]
    pub PRINT_STDERR,
    restriction,
    "printing on stderr"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Debug` formatting. The purpose of this
    /// lint is to catch debugging remnants.
    ///
    /// ### Why restrict this?
    /// The purpose of the `Debug` trait is to facilitate debugging Rust code,
    /// and [no guarantees are made about its output][stability].
    /// It should not be used in user-facing output.
    ///
    /// ### Example
    /// ```no_run
    /// # let foo = "bar";
    /// println!("{:?}", foo);
    /// ```
    ///
    /// [stability]: https://doc.rust-lang.org/stable/std/fmt/trait.Debug.html#stability
    #[clippy::version = "pre 1.29.0"]
    pub USE_DEBUG,
    restriction,
    "use of `Debug`-based formatting"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns about the use of literals as `print!`/`println!` args.
    ///
    /// ### Why is this bad?
    /// Using literals as `println!` args is inefficient
    /// (c.f., https://github.com/matthiaskrgr/rust-str-bench) and unnecessary
    /// (i.e., just put the literal in the format string)
    ///
    /// ### Example
    /// ```no_run
    /// println!("{}", "foo");
    /// ```
    /// use the literal without formatting:
    /// ```no_run
    /// println!("foo");
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub PRINT_LITERAL,
    style,
    "printing a literal with a format string"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns when you use `writeln!(buf, "")` to
    /// print a newline.
    ///
    /// ### Why is this bad?
    /// You should use `writeln!(buf)`, which is simpler.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// writeln!(buf, "");
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// writeln!(buf);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub WRITELN_EMPTY_STRING,
    style,
    "using `writeln!(buf, \"\")` with an empty string"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns when you use `write!()` with a format
    /// string that
    /// ends in a newline.
    ///
    /// ### Why is this bad?
    /// You should use `writeln!()` instead, which appends the
    /// newline.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// # let name = "World";
    /// write!(buf, "Hello {}!\n", name);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// # let name = "World";
    /// writeln!(buf, "Hello {}!", name);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub WRITE_WITH_NEWLINE,
    style,
    "using `write!()` with a format string that ends in a single newline"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns about the use of literals as `write!`/`writeln!` args.
    ///
    /// ### Why is this bad?
    /// Using literals as `writeln!` args is inefficient
    /// (c.f., https://github.com/matthiaskrgr/rust-str-bench) and unnecessary
    /// (i.e., just put the literal in the format string)
    ///
    /// ### Example
    /// ```no_run
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// writeln!(buf, "{}", "foo");
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// writeln!(buf, "foo");
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub WRITE_LITERAL,
    style,
    "writing a literal with a format string"
}

pub struct Write {
    format_args: FormatArgsStorage,
    // The outermost `impl Debug` we're currently in. While we're in one, `USE_DEBUG` is deactivated
    outermost_debug_impl: Option<OwnerId>,
    allow_print_in_tests: bool,
}

impl Write {
    pub fn new(conf: &'static Conf, format_args: FormatArgsStorage) -> Self {
        Self {
            format_args,
            outermost_debug_impl: None,
            allow_print_in_tests: conf.allow_print_in_tests,
        }
    }

    fn in_debug_impl(&self) -> bool {
        self.outermost_debug_impl.is_some()
    }
}

impl_lint_pass!(Write => [
    PRINT_WITH_NEWLINE,
    PRINTLN_EMPTY_STRING,
    PRINT_STDOUT,
    PRINT_STDERR,
    USE_DEBUG,
    PRINT_LITERAL,
    WRITE_WITH_NEWLINE,
    WRITELN_EMPTY_STRING,
    WRITE_LITERAL,
]);

impl<'tcx> LateLintPass<'tcx> for Write {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        // Only check for `impl Debug`s if we're not already in one
        if self.outermost_debug_impl.is_none() && is_debug_impl(cx, item) {
            self.outermost_debug_impl = Some(item.owner_id);
        }
    }

    fn check_item_post(&mut self, _cx: &LateContext<'_>, item: &Item<'_>) {
        // Only clear `self.outermost_debug_impl` if we're escaping the _outermost_ debug impl
        if self.outermost_debug_impl == Some(item.owner_id) {
            self.outermost_debug_impl = None;
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else {
            return;
        };
        let Some(diag_name) = cx.tcx.get_diagnostic_name(macro_call.def_id) else {
            return;
        };
        let Some(name) = diag_name.as_str().strip_suffix("_macro") else {
            return;
        };

        let is_build_script = cx
            .sess()
            .opts
            .crate_name
            .as_ref()
            .is_some_and(|crate_name| crate_name == "build_script_build");

        let allowed_in_tests = self.allow_print_in_tests && is_in_test(cx.tcx, expr.hir_id);
        match diag_name {
            sym::print_macro | sym::println_macro if !allowed_in_tests => {
                if !is_build_script {
                    span_lint(cx, PRINT_STDOUT, macro_call.span, format!("use of `{name}!`"));
                }
            },
            sym::eprint_macro | sym::eprintln_macro if !allowed_in_tests => {
                span_lint(cx, PRINT_STDERR, macro_call.span, format!("use of `{name}!`"));
            },
            sym::write_macro | sym::writeln_macro => {},
            _ => return,
        }

        if let Some(format_args) = self.format_args.get(cx, expr, macro_call.expn) {
            // ignore `writeln!(w)` and `write!(v, some_macro!())`
            if format_args.span.from_expansion() {
                return;
            }

            match diag_name {
                sym::print_macro | sym::eprint_macro | sym::write_macro => {
                    with_newline::check(cx, format_args, &macro_call, name);
                },
                sym::println_macro | sym::eprintln_macro | sym::writeln_macro => {
                    empty_string::check(cx, format_args, &macro_call, name);
                },
                _ => {},
            }

            literal::check(cx, format_args, name);

            if !self.in_debug_impl() {
                use_debug::check(cx, format_args);
            }
        }
    }
}

fn is_debug_impl(cx: &LateContext<'_>, item: &Item<'_>) -> bool {
    if let ItemKind::Impl(Impl {
        of_trait: Some(of_trait),
        ..
    }) = &item.kind
        && let Some(trait_id) = of_trait.trait_ref.trait_def_id()
    {
        cx.tcx.is_diagnostic_item(sym::Debug, trait_id)
    } else {
        false
    }
}
