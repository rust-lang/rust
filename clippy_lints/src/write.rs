use std::borrow::Cow;
use std::ops::Range;

use crate::utils::{snippet_with_applicability, span_lint, span_lint_and_sugg, span_lint_and_then};
use rustc_ast::ast::{Expr, ExprKind, Item, ItemKind, MacCall, StrLit, StrStyle};
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_errors::Applicability;
use rustc_lexer::unescape::{self, EscapeError};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_parse::parser;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::kw;
use rustc_span::{sym, BytePos, Span};

declare_clippy_lint! {
    /// **What it does:** This lint warns when you use `println!("")` to
    /// print a newline.
    ///
    /// **Why is this bad?** You should use `println!()`, which is simpler.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// // Bad
    /// println!("");
    ///
    /// // Good
    /// println!();
    /// ```
    pub PRINTLN_EMPTY_STRING,
    style,
    "using `println!(\"\")` with an empty string"
}

declare_clippy_lint! {
    /// **What it does:** This lint warns when you use `print!()` with a format
    /// string that ends in a newline.
    ///
    /// **Why is this bad?** You should use `println!()` instead, which appends the
    /// newline.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let name = "World";
    /// print!("Hello {}!\n", name);
    /// ```
    /// use println!() instead
    /// ```rust
    /// # let name = "World";
    /// println!("Hello {}!", name);
    /// ```
    pub PRINT_WITH_NEWLINE,
    style,
    "using `print!()` with a format string that ends in a single newline"
}

declare_clippy_lint! {
    /// **What it does:** Checks for printing on *stdout*. The purpose of this lint
    /// is to catch debugging remnants.
    ///
    /// **Why is this bad?** People often print on *stdout* while debugging an
    /// application and might forget to remove those prints afterward.
    ///
    /// **Known problems:** Only catches `print!` and `println!` calls.
    ///
    /// **Example:**
    /// ```rust
    /// println!("Hello world!");
    /// ```
    pub PRINT_STDOUT,
    restriction,
    "printing on stdout"
}

declare_clippy_lint! {
    /// **What it does:** Checks for printing on *stderr*. The purpose of this lint
    /// is to catch debugging remnants.
    ///
    /// **Why is this bad?** People often print on *stderr* while debugging an
    /// application and might forget to remove those prints afterward.
    ///
    /// **Known problems:** Only catches `eprint!` and `eprintln!` calls.
    ///
    /// **Example:**
    /// ```rust
    /// eprintln!("Hello world!");
    /// ```
    pub PRINT_STDERR,
    restriction,
    "printing on stderr"
}

declare_clippy_lint! {
    /// **What it does:** Checks for use of `Debug` formatting. The purpose of this
    /// lint is to catch debugging remnants.
    ///
    /// **Why is this bad?** The purpose of the `Debug` trait is to facilitate
    /// debugging Rust code. It should not be used in user-facing output.
    ///
    /// **Example:**
    /// ```rust
    /// # let foo = "bar";
    /// println!("{:?}", foo);
    /// ```
    pub USE_DEBUG,
    restriction,
    "use of `Debug`-based formatting"
}

declare_clippy_lint! {
    /// **What it does:** This lint warns about the use of literals as `print!`/`println!` args.
    ///
    /// **Why is this bad?** Using literals as `println!` args is inefficient
    /// (c.f., https://github.com/matthiaskrgr/rust-str-bench) and unnecessary
    /// (i.e., just put the literal in the format string)
    ///
    /// **Known problems:** Will also warn with macro calls as arguments that expand to literals
    /// -- e.g., `println!("{}", env!("FOO"))`.
    ///
    /// **Example:**
    /// ```rust
    /// println!("{}", "foo");
    /// ```
    /// use the literal without formatting:
    /// ```rust
    /// println!("foo");
    /// ```
    pub PRINT_LITERAL,
    style,
    "printing a literal with a format string"
}

declare_clippy_lint! {
    /// **What it does:** This lint warns when you use `writeln!(buf, "")` to
    /// print a newline.
    ///
    /// **Why is this bad?** You should use `writeln!(buf)`, which is simpler.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    ///
    /// // Bad
    /// writeln!(buf, "");
    ///
    /// // Good
    /// writeln!(buf);
    /// ```
    pub WRITELN_EMPTY_STRING,
    style,
    "using `writeln!(buf, \"\")` with an empty string"
}

declare_clippy_lint! {
    /// **What it does:** This lint warns when you use `write!()` with a format
    /// string that
    /// ends in a newline.
    ///
    /// **Why is this bad?** You should use `writeln!()` instead, which appends the
    /// newline.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// # let name = "World";
    ///
    /// // Bad
    /// write!(buf, "Hello {}!\n", name);
    ///
    /// // Good
    /// writeln!(buf, "Hello {}!", name);
    /// ```
    pub WRITE_WITH_NEWLINE,
    style,
    "using `write!()` with a format string that ends in a single newline"
}

declare_clippy_lint! {
    /// **What it does:** This lint warns about the use of literals as `write!`/`writeln!` args.
    ///
    /// **Why is this bad?** Using literals as `writeln!` args is inefficient
    /// (c.f., https://github.com/matthiaskrgr/rust-str-bench) and unnecessary
    /// (i.e., just put the literal in the format string)
    ///
    /// **Known problems:** Will also warn with macro calls as arguments that expand to literals
    /// -- e.g., `writeln!(buf, "{}", env!("FOO"))`.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    ///
    /// // Bad
    /// writeln!(buf, "{}", "foo");
    ///
    /// // Good
    /// writeln!(buf, "foo");
    /// ```
    pub WRITE_LITERAL,
    style,
    "writing a literal with a format string"
}

#[derive(Default)]
pub struct Write {
    in_debug_impl: bool,
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
    WRITE_LITERAL
]);

impl EarlyLintPass for Write {
    fn check_item(&mut self, _: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::Impl {
            of_trait: Some(trait_ref),
            ..
        } = &item.kind
        {
            let trait_name = trait_ref
                .path
                .segments
                .iter()
                .last()
                .expect("path has at least one segment")
                .ident
                .name;
            if trait_name == sym::Debug {
                self.in_debug_impl = true;
            }
        }
    }

    fn check_item_post(&mut self, _: &EarlyContext<'_>, _: &Item) {
        self.in_debug_impl = false;
    }

    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac: &MacCall) {
        fn is_build_script(cx: &EarlyContext<'_>) -> bool {
            // Cargo sets the crate name for build scripts to `build_script_build`
            cx.sess
                .opts
                .crate_name
                .as_ref()
                .map_or(false, |crate_name| crate_name == "build_script_build")
        }

        if mac.path == sym!(print) {
            if !is_build_script(cx) {
                span_lint(cx, PRINT_STDOUT, mac.span(), "use of `print!`");
            }
            self.lint_print_with_newline(cx, mac);
        } else if mac.path == sym!(println) {
            if !is_build_script(cx) {
                span_lint(cx, PRINT_STDOUT, mac.span(), "use of `println!`");
            }
            self.lint_println_empty_string(cx, mac);
        } else if mac.path == sym!(eprint) {
            span_lint(cx, PRINT_STDERR, mac.span(), "use of `eprint!`");
            self.lint_print_with_newline(cx, mac);
        } else if mac.path == sym!(eprintln) {
            span_lint(cx, PRINT_STDERR, mac.span(), "use of `eprintln!`");
            self.lint_println_empty_string(cx, mac);
        } else if mac.path == sym!(write) {
            if let (Some(fmt_str), _) = self.check_tts(cx, mac.args.inner_tokens(), true) {
                if check_newlines(&fmt_str) {
                    span_lint_and_then(
                        cx,
                        WRITE_WITH_NEWLINE,
                        mac.span(),
                        "using `write!()` with a format string that ends in a single newline",
                        |err| {
                            err.multipart_suggestion(
                                "use `writeln!()` instead",
                                vec![
                                    (mac.path.span, String::from("writeln")),
                                    (newline_span(&fmt_str), String::new()),
                                ],
                                Applicability::MachineApplicable,
                            );
                        },
                    )
                }
            }
        } else if mac.path == sym!(writeln) {
            if let (Some(fmt_str), expr) = self.check_tts(cx, mac.args.inner_tokens(), true) {
                if fmt_str.symbol == kw::Empty {
                    let mut applicability = Applicability::MachineApplicable;
                    // FIXME: remove this `#[allow(...)]` once the issue #5822 gets fixed
                    #[allow(clippy::option_if_let_else)]
                    let suggestion = if let Some(e) = expr {
                        snippet_with_applicability(cx, e.span, "v", &mut applicability)
                    } else {
                        applicability = Applicability::HasPlaceholders;
                        Cow::Borrowed("v")
                    };

                    span_lint_and_sugg(
                        cx,
                        WRITELN_EMPTY_STRING,
                        mac.span(),
                        format!("using `writeln!({}, \"\")`", suggestion).as_str(),
                        "replace it with",
                        format!("writeln!({})", suggestion),
                        applicability,
                    );
                }
            }
        }
    }
}

/// Given a format string that ends in a newline and its span, calculates the span of the
/// newline, or the format string itself if the format string consists solely of a newline.
fn newline_span(fmtstr: &StrLit) -> Span {
    let sp = fmtstr.span;
    let contents = &fmtstr.symbol.as_str();

    if *contents == r"\n" {
        return sp;
    }

    let newline_sp_hi = sp.hi()
        - match fmtstr.style {
            StrStyle::Cooked => BytePos(1),
            StrStyle::Raw(hashes) => BytePos((1 + hashes).into()),
        };

    let newline_sp_len = if contents.ends_with('\n') {
        BytePos(1)
    } else if contents.ends_with(r"\n") {
        BytePos(2)
    } else {
        panic!("expected format string to contain a newline");
    };

    sp.with_lo(newline_sp_hi - newline_sp_len).with_hi(newline_sp_hi)
}

impl Write {
    /// Checks the arguments of `print[ln]!` and `write[ln]!` calls. It will return a tuple of two
    /// `Option`s. The first `Option` of the tuple is the macro's format string. It includes
    /// the contents of the string, whether it's a raw string, and the span of the literal in the
    /// source. The second `Option` in the tuple is, in the `write[ln]!` case, the expression the
    /// `format_str` should be written to.
    ///
    /// Example:
    ///
    /// Calling this function on
    /// ```rust
    /// # use std::fmt::Write;
    /// # let mut buf = String::new();
    /// # let something = "something";
    /// writeln!(buf, "string to write: {}", something);
    /// ```
    /// will return
    /// ```rust,ignore
    /// (Some("string to write: {}"), Some(buf))
    /// ```
    #[allow(clippy::too_many_lines)]
    fn check_tts<'a>(&self, cx: &EarlyContext<'a>, tts: TokenStream, is_write: bool) -> (Option<StrLit>, Option<Expr>) {
        use rustc_parse_format::{
            AlignUnknown, ArgumentImplicitlyIs, ArgumentIs, ArgumentNamed, CountImplied, FormatSpec, ParseMode, Parser,
            Piece,
        };

        let mut parser = parser::Parser::new(&cx.sess.parse_sess, tts, false, None);
        let mut expr: Option<Expr> = None;
        if is_write {
            expr = match parser.parse_expr().map_err(|mut err| err.cancel()) {
                Ok(p) => Some(p.into_inner()),
                Err(_) => return (None, None),
            };
            // might be `writeln!(foo)`
            if parser.expect(&token::Comma).map_err(|mut err| err.cancel()).is_err() {
                return (None, expr);
            }
        }

        let fmtstr = match parser.parse_str_lit() {
            Ok(fmtstr) => fmtstr,
            Err(_) => return (None, expr),
        };
        let tmp = fmtstr.symbol.as_str();
        let mut args = vec![];
        let mut fmt_parser = Parser::new(&tmp, None, None, false, ParseMode::Format);
        while let Some(piece) = fmt_parser.next() {
            if !fmt_parser.errors.is_empty() {
                return (None, expr);
            }
            if let Piece::NextArgument(arg) = piece {
                if !self.in_debug_impl && arg.format.ty == "?" {
                    // FIXME: modify rustc's fmt string parser to give us the current span
                    span_lint(cx, USE_DEBUG, parser.prev_token.span, "use of `Debug`-based formatting");
                }
                args.push(arg);
            }
        }
        let lint = if is_write { WRITE_LITERAL } else { PRINT_LITERAL };
        let mut idx = 0;
        loop {
            const SIMPLE: FormatSpec<'_> = FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                precision_span: None,
                width: CountImplied,
                width_span: None,
                ty: "",
                ty_span: None,
            };
            if !parser.eat(&token::Comma) {
                return (Some(fmtstr), expr);
            }
            let token_expr = if let Ok(expr) = parser.parse_expr().map_err(|mut err| err.cancel()) {
                expr
            } else {
                return (Some(fmtstr), None);
            };
            match &token_expr.kind {
                ExprKind::Lit(_) => {
                    let mut all_simple = true;
                    let mut seen = false;
                    for arg in &args {
                        match arg.position {
                            ArgumentImplicitlyIs(n) | ArgumentIs(n) => {
                                if n == idx {
                                    all_simple &= arg.format == SIMPLE;
                                    seen = true;
                                }
                            },
                            ArgumentNamed(_) => {},
                        }
                    }
                    if all_simple && seen {
                        span_lint(cx, lint, token_expr.span, "literal with an empty format string");
                    }
                    idx += 1;
                },
                ExprKind::Assign(lhs, rhs, _) => {
                    if let ExprKind::Lit(_) = rhs.kind {
                        if let ExprKind::Path(_, p) = &lhs.kind {
                            let mut all_simple = true;
                            let mut seen = false;
                            for arg in &args {
                                match arg.position {
                                    ArgumentImplicitlyIs(_) | ArgumentIs(_) => {},
                                    ArgumentNamed(name) => {
                                        if *p == name {
                                            seen = true;
                                            all_simple &= arg.format == SIMPLE;
                                        }
                                    },
                                }
                            }
                            if all_simple && seen {
                                span_lint(cx, lint, rhs.span, "literal with an empty format string");
                            }
                        }
                    }
                },
                _ => idx += 1,
            }
        }
    }

    fn lint_println_empty_string(&self, cx: &EarlyContext<'_>, mac: &MacCall) {
        if let (Some(fmt_str), _) = self.check_tts(cx, mac.args.inner_tokens(), false) {
            if fmt_str.symbol == kw::Empty {
                let name = mac.path.segments[0].ident.name;
                span_lint_and_sugg(
                    cx,
                    PRINTLN_EMPTY_STRING,
                    mac.span(),
                    &format!("using `{}!(\"\")`", name),
                    "replace it with",
                    format!("{}!()", name),
                    Applicability::MachineApplicable,
                );
            }
        }
    }

    fn lint_print_with_newline(&self, cx: &EarlyContext<'_>, mac: &MacCall) {
        if let (Some(fmt_str), _) = self.check_tts(cx, mac.args.inner_tokens(), false) {
            if check_newlines(&fmt_str) {
                let name = mac.path.segments[0].ident.name;
                let suggested = format!("{}ln", name);
                span_lint_and_then(
                    cx,
                    PRINT_WITH_NEWLINE,
                    mac.span(),
                    &format!("using `{}!()` with a format string that ends in a single newline", name),
                    |err| {
                        err.multipart_suggestion(
                            &format!("use `{}!` instead", suggested),
                            vec![(mac.path.span, suggested), (newline_span(&fmt_str), String::new())],
                            Applicability::MachineApplicable,
                        );
                    },
                );
            }
        }
    }
}

/// Checks if the format string contains a single newline that terminates it.
///
/// Literal and escaped newlines are both checked (only literal for raw strings).
fn check_newlines(fmtstr: &StrLit) -> bool {
    let mut has_internal_newline = false;
    let mut last_was_cr = false;
    let mut should_lint = false;

    let contents = &fmtstr.symbol.as_str();

    let mut cb = |r: Range<usize>, c: Result<char, EscapeError>| {
        let c = c.unwrap();

        if r.end == contents.len() && c == '\n' && !last_was_cr && !has_internal_newline {
            should_lint = true;
        } else {
            last_was_cr = c == '\r';
            if c == '\n' {
                has_internal_newline = true;
            }
        }
    };

    match fmtstr.style {
        StrStyle::Cooked => unescape::unescape_literal(contents, unescape::Mode::Str, &mut cb),
        StrStyle::Raw(_) => unescape::unescape_literal(contents, unescape::Mode::RawStr, &mut cb),
    }

    should_lint
}
