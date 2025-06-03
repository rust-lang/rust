use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::macros::{FormatArgsStorage, MacroCall, format_arg_removal_span, root_macro_call_first_node};
use clippy_utils::source::{SpanRangeExt, expand_past_previous_comma};
use clippy_utils::{is_in_test, sym};
use rustc_ast::token::LitKind;
use rustc_ast::{
    FormatArgPosition, FormatArgPositionKind, FormatArgs, FormatArgsPiece, FormatCount, FormatOptions,
    FormatPlaceholder, FormatTrait,
};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::{BytePos, Span};

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
    in_debug_impl: bool,
    allow_print_in_tests: bool,
}

impl Write {
    pub fn new(conf: &'static Conf, format_args: FormatArgsStorage) -> Self {
        Self {
            format_args,
            in_debug_impl: false,
            allow_print_in_tests: conf.allow_print_in_tests,
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
        }
    }
}

fn is_debug_impl(cx: &LateContext<'_>, item: &Item<'_>) -> bool {
    if let ItemKind::Impl(Impl {
        of_trait: Some(trait_ref),
        ..
    }) = &item.kind
        && let Some(trait_id) = trait_ref.trait_def_id()
    {
        cx.tcx.is_diagnostic_item(sym::Debug, trait_id)
    } else {
        false
    }
}

fn check_newline(cx: &LateContext<'_>, format_args: &FormatArgs, macro_call: &MacroCall, name: &str) {
    let Some(&FormatArgsPiece::Literal(last)) = format_args.template.last() else {
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
            format!("using `{name}!()` with a format string that ends in a single newline"),
            |diag| {
                let name_span = cx.sess().source_map().span_until_char(macro_call.span, '!');
                let Some(format_snippet) = format_string_span.get_source_text(cx) else {
                    return;
                };

                if format_args.template.len() == 1 && last == sym::LF {
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
    if let [FormatArgsPiece::Literal(sym::LF)] = &format_args.template[..] {
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
            format!("empty string literal in `{name}!`"),
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

    let lint_name = if name.starts_with("write") {
        WRITE_LITERAL
    } else {
        PRINT_LITERAL
    };

    let mut counts = vec![0u32; format_args.arguments.all_args().len()];
    for piece in &format_args.template {
        if let FormatArgsPiece::Placeholder(placeholder) = piece {
            counts[arg_index(&placeholder.argument)] += 1;
        }
    }

    let mut suggestion: Vec<(Span, String)> = vec![];
    // holds index of replaced positional arguments; used to decrement the index of the remaining
    // positional arguments.
    let mut replaced_position: Vec<usize> = vec![];
    let mut sug_span: Option<Span> = None;

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
            && let Some(value_string) = arg.expr.span.get_source_text(cx)
        {
            let (replacement, replace_raw) = match lit.kind {
                LitKind::Str | LitKind::StrRaw(_) => match extract_str_literal(&value_string) {
                    Some(extracted) => extracted,
                    None => return,
                },
                LitKind::Char => (
                    match lit.symbol {
                        sym::DOUBLE_QUOTE => "\\\"",
                        sym::BACKSLASH_SINGLE_QUOTE => "'",
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

            let Some(format_string_snippet) = format_args.span.get_source_text(cx) else {
                continue;
            };
            let format_string_is_raw = format_string_snippet.starts_with('r');

            let replacement = match (format_string_is_raw, replace_raw) {
                (false, false) => Some(replacement),
                (false, true) => Some(replacement.replace('\\', "\\\\").replace('"', "\\\"")),
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

            sug_span = Some(sug_span.unwrap_or(arg.expr.span).to(arg.expr.span));

            if let Some((_, index)) = positional_arg_piece_span(piece) {
                replaced_position.push(index);
            }

            if let Some(replacement) = replacement
                // `format!("{}", "a")`, `format!("{named}", named = "b")
                //              ~~~~~                      ~~~~~~~~~~~~~
                && let Some(removal_span) = format_arg_removal_span(format_args, index)
            {
                let replacement = escape_braces(&replacement, !format_string_is_raw && !replace_raw);
                suggestion.push((*placeholder_span, replacement));
                suggestion.push((removal_span, String::new()));
            }
        }
    }

    // Decrement the index of the remaining by the number of replaced positional arguments
    if !suggestion.is_empty() {
        for piece in &format_args.template {
            relocalize_format_args_indexes(piece, &mut suggestion, &replaced_position);
        }
    }

    if let Some(span) = sug_span {
        span_lint_and_then(cx, lint_name, span, "literal with an empty format string", |diag| {
            if !suggestion.is_empty() {
                diag.multipart_suggestion("try", suggestion, Applicability::MachineApplicable);
            }
        });
    }
}

/// Extract Span and its index from the given `piece`, if it's positional argument.
fn positional_arg_piece_span(piece: &FormatArgsPiece) -> Option<(Span, usize)> {
    match piece {
        FormatArgsPiece::Placeholder(FormatPlaceholder {
            argument:
                FormatArgPosition {
                    index: Ok(index),
                    kind: FormatArgPositionKind::Number,
                    ..
                },
            span: Some(span),
            ..
        }) => Some((*span, *index)),
        _ => None,
    }
}

/// Relocalizes the indexes of positional arguments in the format string
fn relocalize_format_args_indexes(
    piece: &FormatArgsPiece,
    suggestion: &mut Vec<(Span, String)>,
    replaced_position: &[usize],
) {
    if let FormatArgsPiece::Placeholder(FormatPlaceholder {
        argument:
            FormatArgPosition {
                index: Ok(index),
                // Only consider positional arguments
                kind: FormatArgPositionKind::Number,
                span: Some(span),
            },
        format_options,
        ..
    }) = piece
    {
        if suggestion.iter().any(|(s, _)| s.overlaps(*span)) {
            // If the span is already in the suggestion, we don't need to process it again
            return;
        }

        // lambda to get the decremented index based on the replaced positions
        let decremented_index = |index: usize| -> usize {
            let decrement = replaced_position.iter().filter(|&&i| i < index).count();
            index - decrement
        };

        suggestion.push((*span, decremented_index(*index).to_string()));

        // If there are format options, we need to handle them as well
        if *format_options != FormatOptions::default() {
            // lambda to process width and precision format counts and add them to the suggestion
            let mut process_format_count = |count: &Option<FormatCount>, formatter: &dyn Fn(usize) -> String| {
                if let Some(FormatCount::Argument(FormatArgPosition {
                    index: Ok(format_arg_index),
                    kind: FormatArgPositionKind::Number,
                    span: Some(format_arg_span),
                })) = count
                {
                    suggestion.push((*format_arg_span, formatter(decremented_index(*format_arg_index))));
                }
            };

            process_format_count(&format_options.width, &|index: usize| format!("{index}$"));
            process_format_count(&format_options.precision, &|index: usize| format!(".{index}$"));
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

/// Replaces `{` with `{{` and `}` with `}}`. If `preserve_unicode_escapes` is `true` the braces in
/// `\u{xxxx}` are left unmodified
#[expect(clippy::match_same_arms)]
fn escape_braces(literal: &str, preserve_unicode_escapes: bool) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Normal,
        Backslash,
        UnicodeEscape,
    }

    let mut escaped = String::with_capacity(literal.len());
    let mut state = State::Normal;

    for ch in literal.chars() {
        state = match (ch, state) {
            // Escape braces outside of unicode escapes by doubling them up
            ('{' | '}', State::Normal) => {
                escaped.push(ch);
                State::Normal
            },
            // If `preserve_unicode_escapes` isn't enabled stay in `State::Normal`, otherwise:
            //
            // \u{aaaa} \\ \x01
            // ^        ^  ^
            ('\\', State::Normal) if preserve_unicode_escapes => State::Backslash,
            // \u{aaaa}
            //  ^
            ('u', State::Backslash) => State::UnicodeEscape,
            // \xAA \\
            //  ^    ^
            (_, State::Backslash) => State::Normal,
            // \u{aaaa}
            //        ^
            ('}', State::UnicodeEscape) => State::Normal,
            _ => state,
        };

        escaped.push(ch);
    }

    escaped
}
