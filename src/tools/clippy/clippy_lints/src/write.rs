use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::macros::{find_format_args, format_arg_removal_span, root_macro_call_first_node, MacroCall};
use clippy_utils::source::{expand_past_previous_comma, snippet_opt};
use clippy_utils::{is_in_cfg_test, is_in_test_function};
use rustc_ast::token::LitKind;
use rustc_ast::{FormatArgPosition, FormatArgs, FormatArgsPiece, FormatOptions, FormatPlaceholder, FormatTrait};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, BytePos};

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns when you use `println!("")` to
    /// print a newline.
    ///
    /// ### Why is this bad?
    /// You should use `println!()`, which is simpler.
    ///
    /// ### Example
    /// ```rust
    /// println!("");
    /// ```
    ///
    /// Use instead:
    /// ```rust
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
    /// ```rust
    /// # let name = "World";
    /// print!("Hello {}!\n", name);
    /// ```
    /// use println!() instead
    /// ```rust
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
    /// ### Why is this bad?
    /// People often print on *stdout* while debugging an
    /// application and might forget to remove those prints afterward.
    ///
    /// ### Known problems
    /// Only catches `print!` and `println!` calls.
    ///
    /// ### Example
    /// ```rust
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
    /// ### Why is this bad?
    /// People often print on *stderr* while debugging an
    /// application and might forget to remove those prints afterward.
    ///
    /// ### Known problems
    /// Only catches `eprint!` and `eprintln!` calls.
    ///
    /// ### Example
    /// ```rust
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
    /// ### Why is this bad?
    /// The purpose of the `Debug` trait is to facilitate
    /// debugging Rust code. It should not be used in user-facing output.
    ///
    /// ### Example
    /// ```rust
    /// # let foo = "bar";
    /// println!("{:?}", foo);
    /// ```
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
    /// ```rust
    /// println!("{}", "foo");
    /// ```
    /// use the literal without formatting:
    /// ```rust
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
    /// ```rust
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// writeln!(buf, "");
    /// ```
    ///
    /// Use instead:
    /// ```rust
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
    /// ```rust
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// # let name = "World";
    /// write!(buf, "Hello {}!\n", name);
    /// ```
    ///
    /// Use instead:
    /// ```rust
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
    /// ```rust
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// writeln!(buf, "{}", "foo");
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// writeln!(buf, "foo");
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub WRITE_LITERAL,
    style,
    "writing a literal with a format string"
}

#[derive(Default)]
pub struct Write {
    in_debug_impl: bool,
    allow_print_in_tests: bool,
}

impl Write {
    pub fn new(allow_print_in_tests: bool) -> Self {
        Self {
            allow_print_in_tests,
            ..Default::default()
        }
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
        if is_debug_impl(cx, item) {
            self.in_debug_impl = true;
        }
    }

    fn check_item_post(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if is_debug_impl(cx, item) {
            self.in_debug_impl = false;
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
            .map_or(false, |crate_name| crate_name == "build_script_build");

        let allowed_in_tests = self.allow_print_in_tests
            && (is_in_test_function(cx.tcx, expr.hir_id) || is_in_cfg_test(cx.tcx, expr.hir_id));
        match diag_name {
            sym::print_macro | sym::println_macro if !allowed_in_tests => {
                if !is_build_script {
                    span_lint(cx, PRINT_STDOUT, macro_call.span, &format!("use of `{name}!`"));
                }
            },
            sym::eprint_macro | sym::eprintln_macro if !allowed_in_tests => {
                span_lint(cx, PRINT_STDERR, macro_call.span, &format!("use of `{name}!`"));
            },
            sym::write_macro | sym::writeln_macro => {},
            _ => return,
        }

        find_format_args(cx, expr, macro_call.expn, |format_args| {
            // ignore `writeln!(w)` and `write!(v, some_macro!())`
            if format_args.span.from_expansion() {
                return;
            }

            match diag_name {
                sym::print_macro | sym::eprint_macro | sym::write_macro => {
                    check_newline(cx, format_args, &macro_call, name);
                },
                sym::println_macro | sym::eprintln_macro | sym::writeln_macro => {
                    check_empty_string(cx, format_args, &macro_call, name);
                },
                _ => {},
            }

            check_literal(cx, format_args, name);

            if !self.in_debug_impl {
                for piece in &format_args.template {
                    if let &FormatArgsPiece::Placeholder(FormatPlaceholder {
                        span: Some(span),
                        format_trait: FormatTrait::Debug,
                        ..
                    }) = piece
                    {
                        span_lint(cx, USE_DEBUG, span, "use of `Debug`-based formatting");
                    }
                }
            }
        });
    }
}

fn is_debug_impl(cx: &LateContext<'_>, item: &Item<'_>) -> bool {
    if let ItemKind::Impl(Impl { of_trait: Some(trait_ref), .. }) = &item.kind
        && let Some(trait_id) = trait_ref.trait_def_id()
    {
        cx.tcx.is_diagnostic_item(sym::Debug, trait_id)
    } else {
        false
    }
}

fn check_newline(cx: &LateContext<'_>, format_args: &FormatArgs, macro_call: &MacroCall, name: &str) {
    let Some(FormatArgsPiece::Literal(last)) = format_args.template.last() else {
        return;
    };

    let count_vertical_whitespace = || {
        format_args
            .template
            .iter()
            .filter_map(|piece| match piece {
                FormatArgsPiece::Literal(literal) => Some(literal),
                FormatArgsPiece::Placeholder(_) => None,
            })
            .flat_map(|literal| literal.as_str().chars())
            .filter(|ch| matches!(ch, '\r' | '\n'))
            .count()
    };

    if last.as_str().ends_with('\n')
        // ignore format strings with other internal vertical whitespace
        && count_vertical_whitespace() == 1
    {
        let mut format_string_span = format_args.span;

        let lint = if name == "write" {
            format_string_span = expand_past_previous_comma(cx, format_string_span);

            WRITE_WITH_NEWLINE
        } else {
            PRINT_WITH_NEWLINE
        };

        span_lint_and_then(
            cx,
            lint,
            macro_call.span,
            &format!("using `{name}!()` with a format string that ends in a single newline"),
            |diag| {
                let name_span = cx.sess().source_map().span_until_char(macro_call.span, '!');
                let Some(format_snippet) = snippet_opt(cx, format_string_span) else {
                    return;
                };

                if format_args.template.len() == 1 && last.as_str() == "\n" {
                    // print!("\n"), write!(f, "\n")

                    diag.multipart_suggestion(
                        format!("use `{name}ln!` instead"),
                        vec![(name_span, format!("{name}ln")), (format_string_span, String::new())],
                        Applicability::MachineApplicable,
                    );
                } else if format_snippet.ends_with("\\n\"") {
                    // print!("...\n"), write!(f, "...\n")

                    let hi = format_string_span.hi();
                    let newline_span = format_string_span.with_lo(hi - BytePos(3)).with_hi(hi - BytePos(1));

                    diag.multipart_suggestion(
                        format!("use `{name}ln!` instead"),
                        vec![(name_span, format!("{name}ln")), (newline_span, String::new())],
                        Applicability::MachineApplicable,
                    );
                }
            },
        );
    }
}

fn check_empty_string(cx: &LateContext<'_>, format_args: &FormatArgs, macro_call: &MacroCall, name: &str) {
    if let [FormatArgsPiece::Literal(literal)] = &format_args.template[..]
        && literal.as_str() == "\n"
    {
        let mut span = format_args.span;

        let lint = if name == "writeln" {
            span = expand_past_previous_comma(cx, span);

            WRITELN_EMPTY_STRING
        } else {
            PRINTLN_EMPTY_STRING
        };

        span_lint_and_then(
            cx,
            lint,
            macro_call.span,
            &format!("empty string literal in `{name}!`"),
            |diag| {
                diag.span_suggestion(
                    span,
                    "remove the empty string",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

fn check_literal(cx: &LateContext<'_>, format_args: &FormatArgs, name: &str) {
    let arg_index = |argument: &FormatArgPosition| argument.index.unwrap_or_else(|pos| pos);

    let mut counts = vec![0u32; format_args.arguments.all_args().len()];
    for piece in &format_args.template {
        if let FormatArgsPiece::Placeholder(placeholder) = piece {
            counts[arg_index(&placeholder.argument)] += 1;
        }
    }

    for piece in &format_args.template {
        if let FormatArgsPiece::Placeholder(FormatPlaceholder {
            argument,
            span: Some(placeholder_span),
            format_trait: FormatTrait::Display,
            format_options,
        }) = piece
            && *format_options == FormatOptions::default()
            && let index = arg_index(argument)
            && counts[index] == 1
            && let Some(arg) = format_args.arguments.by_index(index)
            && let rustc_ast::ExprKind::Lit(lit) = &arg.expr.kind
            && !arg.expr.span.from_expansion()
            && let Some(value_string) = snippet_opt(cx, arg.expr.span)
    {
            let (replacement, replace_raw) = match lit.kind {
                LitKind::Str | LitKind::StrRaw(_)  => match extract_str_literal(&value_string) {
                    Some(extracted) => extracted,
                    None => return,
                },
                LitKind::Char => (
                    match lit.symbol.as_str() {
                        "\"" => "\\\"",
                        "\\'" => "'",
                        _ => match value_string.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
                            Some(stripped) => stripped,
                            None => return,
                        },
                    }
                    .to_string(),
                    false,
                ),
                LitKind::Bool => (lit.symbol.to_string(), false),
                _ => continue,
            };

            let lint = if name.starts_with("write") {
                WRITE_LITERAL
            } else {
                PRINT_LITERAL
            };

            let Some(format_string_snippet) = snippet_opt(cx, format_args.span) else { continue };
            let format_string_is_raw = format_string_snippet.starts_with('r');

            let replacement = match (format_string_is_raw, replace_raw) {
                (false, false) => Some(replacement),
                (false, true) => Some(replacement.replace('"', "\\\"").replace('\\', "\\\\")),
                (true, false) => match conservative_unescape(&replacement) {
                    Ok(unescaped) => Some(unescaped),
                    Err(UnescapeErr::Lint) => None,
                    Err(UnescapeErr::Ignore) => continue,
                },
                (true, true) => {
                    if replacement.contains(['#', '"']) {
                        None
                    } else {
                        Some(replacement)
                    }
                },
            };

            span_lint_and_then(
                cx,
                lint,
                arg.expr.span,
                "literal with an empty format string",
                |diag| {
                    if let Some(replacement) = replacement
                        // `format!("{}", "a")`, `format!("{named}", named = "b")
                        //              ~~~~~                      ~~~~~~~~~~~~~
                        && let Some(removal_span) = format_arg_removal_span(format_args, index)
                    {
                        let replacement = replacement.replace('{', "{{").replace('}', "}}");
                        diag.multipart_suggestion(
                            "try",
                            vec![(*placeholder_span, replacement), (removal_span, String::new())],
                            Applicability::MachineApplicable,
                        );
                    }
                },
            );

        }
    }
}

/// Removes the raw marker, `#`s and quotes from a str, and returns if the literal is raw
///
/// `r#"a"#` -> (`a`, true)
///
/// `"b"` -> (`b`, false)
fn extract_str_literal(literal: &str) -> Option<(String, bool)> {
    let (literal, raw) = match literal.strip_prefix('r') {
        Some(stripped) => (stripped.trim_matches('#'), true),
        None => (literal, false),
    };

    Some((literal.strip_prefix('"')?.strip_suffix('"')?.to_string(), raw))
}

enum UnescapeErr {
    /// Should still be linted, can be manually resolved by author, e.g.
    ///
    /// ```ignore
    /// print!(r"{}", '"');
    /// ```
    Lint,
    /// Should not be linted, e.g.
    ///
    /// ```ignore
    /// print!(r"{}", '\r');
    /// ```
    Ignore,
}

/// Unescape a normal string into a raw string
fn conservative_unescape(literal: &str) -> Result<String, UnescapeErr> {
    let mut unescaped = String::with_capacity(literal.len());
    let mut chars = literal.chars();
    let mut err = false;

    while let Some(ch) = chars.next() {
        match ch {
            '#' => err = true,
            '\\' => match chars.next() {
                Some('\\') => unescaped.push('\\'),
                Some('"') => err = true,
                _ => return Err(UnescapeErr::Ignore),
            },
            _ => unescaped.push(ch),
        }
    }

    if err { Err(UnescapeErr::Lint) } else { Ok(unescaped) }
}
