use crate::utils::{snippet_with_applicability, span_lint, span_lint_and_sugg, span_lint_and_then};
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use std::borrow::Cow;
use syntax::ast::*;
use syntax::parse::{parser, token};
use syntax::tokenstream::TokenStream;
use syntax_pos::{BytePos, Span};

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
    /// println!("");
    /// ```
    pub PRINTLN_EMPTY_STRING,
    style,
    "using `println!(\"\")` with an empty string"
}

declare_clippy_lint! {
    /// **What it does:** This lint warns when you use `print!()` with a format
    /// string that
    /// ends in a newline.
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
    /// writeln!(buf, "");
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
    /// write!(buf, "Hello {}!\n", name);
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
    /// writeln!(buf, "{}", "foo");
    /// ```
    pub WRITE_LITERAL,
    style,
    "writing a literal with a format string"
}

declare_lint_pass!(Write => [
    PRINT_WITH_NEWLINE,
    PRINTLN_EMPTY_STRING,
    PRINT_STDOUT,
    USE_DEBUG,
    PRINT_LITERAL,
    WRITE_WITH_NEWLINE,
    WRITELN_EMPTY_STRING,
    WRITE_LITERAL
]);

impl EarlyLintPass for Write {
    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac: &Mac) {
        if mac.node.path == sym!(println) {
            span_lint(cx, PRINT_STDOUT, mac.span, "use of `println!`");
            if let (Some(fmt_str), _) = check_tts(cx, &mac.node.tts, false) {
                if fmt_str.contents.is_empty() {
                    span_lint_and_sugg(
                        cx,
                        PRINTLN_EMPTY_STRING,
                        mac.span,
                        "using `println!(\"\")`",
                        "replace it with",
                        "println!()".to_string(),
                        Applicability::MachineApplicable,
                    );
                }
            }
        } else if mac.node.path == sym!(print) {
            span_lint(cx, PRINT_STDOUT, mac.span, "use of `print!`");
            if let (Some(fmt_str), _) = check_tts(cx, &mac.node.tts, false) {
                if check_newlines(&fmt_str) {
                    span_lint_and_then(
                        cx,
                        PRINT_WITH_NEWLINE,
                        mac.span,
                        "using `print!()` with a format string that ends in a single newline",
                        |err| {
                            err.multipart_suggestion(
                                "use `println!` instead",
                                vec![
                                    (mac.node.path.span, String::from("println")),
                                    (fmt_str.newline_span(), String::new()),
                                ],
                                Applicability::MachineApplicable,
                            );
                        },
                    );
                }
            }
        } else if mac.node.path == sym!(write) {
            if let (Some(fmt_str), _) = check_tts(cx, &mac.node.tts, true) {
                if check_newlines(&fmt_str) {
                    span_lint_and_then(
                        cx,
                        WRITE_WITH_NEWLINE,
                        mac.span,
                        "using `write!()` with a format string that ends in a single newline",
                        |err| {
                            err.multipart_suggestion(
                                "use `writeln!()` instead",
                                vec![
                                    (mac.node.path.span, String::from("writeln")),
                                    (fmt_str.newline_span(), String::new()),
                                ],
                                Applicability::MachineApplicable,
                            );
                        },
                    )
                }
            }
        } else if mac.node.path == sym!(writeln) {
            if let (Some(fmt_str), expr) = check_tts(cx, &mac.node.tts, true) {
                if fmt_str.contents.is_empty() {
                    let mut applicability = Applicability::MachineApplicable;
                    let suggestion = expr.map_or_else(
                        move || {
                            applicability = Applicability::HasPlaceholders;
                            Cow::Borrowed("v")
                        },
                        move |expr| snippet_with_applicability(cx, expr.span, "v", &mut applicability),
                    );

                    span_lint_and_sugg(
                        cx,
                        WRITELN_EMPTY_STRING,
                        mac.span,
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

/// The arguments of a `print[ln]!` or `write[ln]!` invocation.
struct FmtStr {
    /// The contents of the format string (inside the quotes).
    contents: String,
    style: StrStyle,
    /// The span of the format string, including quotes, the raw marker, and any raw hashes.
    span: Span,
}

impl FmtStr {
    /// Given a format string that ends in a newline and its span, calculates the span of the
    /// newline.
    fn newline_span(&self) -> Span {
        let sp = self.span;

        let newline_sp_hi = sp.hi()
            - match self.style {
                StrStyle::Cooked => BytePos(1),
                StrStyle::Raw(hashes) => BytePos((1 + hashes).into()),
            };

        let newline_sp_len = if self.contents.ends_with('\n') {
            BytePos(1)
        } else if self.contents.ends_with(r"\n") {
            BytePos(2)
        } else {
            panic!("expected format string to contain a newline");
        };

        sp.with_lo(newline_sp_hi - newline_sp_len).with_hi(newline_sp_hi)
    }
}

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
fn check_tts<'a>(cx: &EarlyContext<'a>, tts: &TokenStream, is_write: bool) -> (Option<FmtStr>, Option<Expr>) {
    use fmt_macros::*;
    let tts = tts.clone();

    let mut parser = parser::Parser::new(&cx.sess.parse_sess, tts, None, false, false, None);
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

    let (fmtstr, fmtstyle) = match parser.parse_str().map_err(|mut err| err.cancel()) {
        Ok((fmtstr, fmtstyle)) => (fmtstr.to_string(), fmtstyle),
        Err(_) => return (None, expr),
    };
    let fmtspan = parser.prev_span;
    let tmp = fmtstr.clone();
    let mut args = vec![];
    let mut fmt_parser = Parser::new(&tmp, None, Vec::new(), false);
    while let Some(piece) = fmt_parser.next() {
        if !fmt_parser.errors.is_empty() {
            return (None, expr);
        }
        if let Piece::NextArgument(arg) = piece {
            if arg.format.ty == "?" {
                // FIXME: modify rustc's fmt string parser to give us the current span
                span_lint(cx, USE_DEBUG, parser.prev_span, "use of `Debug`-based formatting");
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
        };
        if !parser.eat(&token::Comma) {
            return (
                Some(FmtStr {
                    contents: fmtstr,
                    style: fmtstyle,
                    span: fmtspan,
                }),
                expr,
            );
        }
        let token_expr = if let Ok(expr) = parser.parse_expr().map_err(|mut err| err.cancel()) {
            expr
        } else {
            return (
                Some(FmtStr {
                    contents: fmtstr,
                    style: fmtstyle,
                    span: fmtspan,
                }),
                None,
            );
        };
        match &token_expr.node {
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
            ExprKind::Assign(lhs, rhs) => {
                if let ExprKind::Lit(_) = rhs.node {
                    if let ExprKind::Path(_, p) = &lhs.node {
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

/// Checks if the format string constains a single newline that terminates it.
///
/// Literal and escaped newlines are both checked (only literal for raw strings).
fn check_newlines(fmt_str: &FmtStr) -> bool {
    let s = &fmt_str.contents;

    if s.ends_with('\n') {
        return true;
    } else if let StrStyle::Raw(_) = fmt_str.style {
        return false;
    }

    if s.len() < 2 {
        return false;
    }

    let bytes = s.as_bytes();
    if bytes[bytes.len() - 2] != b'\\' || bytes[bytes.len() - 1] != b'n' {
        return false;
    }

    let mut escaping = false;
    for (index, &byte) in bytes.iter().enumerate() {
        if escaping {
            if byte == b'n' {
                return index == bytes.len() - 1;
            }
            escaping = false;
        } else if byte == b'\\' {
            escaping = true;
        }
    }

    false
}
