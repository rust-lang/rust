use rustc::lint::*;
use rustc::{declare_lint, lint_array};
use syntax::ast::*;
use syntax::tokenstream::{ThinTokenStream, TokenStream};
use syntax::parse::{token, parser};
use crate::utils::{span_lint, span_lint_and_sugg};

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
declare_clippy_lint! {
    pub PRINTLN_EMPTY_STRING,
    style,
    "using `println!(\"\")` with an empty string"
}

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
/// print!("Hello {}!\n", name);
/// ```
declare_clippy_lint! {
    pub PRINT_WITH_NEWLINE,
    style,
    "using `print!()` with a format string that ends in a newline"
}

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
declare_clippy_lint! {
    pub PRINT_STDOUT,
    restriction,
    "printing on stdout"
}

/// **What it does:** Checks for use of `Debug` formatting. The purpose of this
/// lint is to catch debugging remnants.
///
/// **Why is this bad?** The purpose of the `Debug` trait is to facilitate
/// debugging Rust code. It should not be used in in user-facing output.
///
/// **Example:**
/// ```rust
/// println!("{:?}", foo);
/// ```
declare_clippy_lint! {
    pub USE_DEBUG,
    restriction,
    "use of `Debug`-based formatting"
}

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
declare_clippy_lint! {
    pub PRINT_LITERAL,
    style,
    "printing a literal with a format string"
}

/// **What it does:** This lint warns when you use `writeln!(buf, "")` to
/// print a newline.
///
/// **Why is this bad?** You should use `writeln!(buf)`, which is simpler.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// writeln!("");
/// ```
declare_clippy_lint! {
    pub WRITELN_EMPTY_STRING,
    style,
    "using `writeln!(\"\")` with an empty string"
}

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
/// write!(buf, "Hello {}!\n", name);
/// ```
declare_clippy_lint! {
    pub WRITE_WITH_NEWLINE,
    style,
    "using `write!()` with a format string that ends in a newline"
}

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
/// writeln!(buf, "{}", "foo");
/// ```
declare_clippy_lint! {
    pub WRITE_LITERAL,
    style,
    "writing a literal with a format string"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            PRINT_WITH_NEWLINE,
            PRINTLN_EMPTY_STRING,
            PRINT_STDOUT,
            USE_DEBUG,
            PRINT_LITERAL,
            WRITE_WITH_NEWLINE,
            WRITELN_EMPTY_STRING,
            WRITE_LITERAL
        )
    }
}

impl EarlyLintPass for Pass {
    fn check_mac(&mut self, cx: &EarlyContext, mac: &Mac) {
        if mac.node.path == "println" {
            span_lint(cx, PRINT_STDOUT, mac.span, "use of `println!`");
            if let Some(fmtstr) = check_tts(cx, &mac.node.tts, false) {
                if fmtstr == "" {
                    span_lint_and_sugg(
                        cx,
                        PRINTLN_EMPTY_STRING,
                        mac.span,
                        "using `println!(\"\")`",
                        "replace it with",
                        "println!()".to_string(),
                    );
                }
            }
        } else if mac.node.path == "print" {
            span_lint(cx, PRINT_STDOUT, mac.span, "use of `print!`");
            if let Some(fmtstr) = check_tts(cx, &mac.node.tts, false) {
                if fmtstr.ends_with("\\n") {
                    span_lint(cx, PRINT_WITH_NEWLINE, mac.span,
                            "using `print!()` with a format string that ends in a \
                            newline, consider using `println!()` instead");
                }
            }
        } else if mac.node.path == "write" {
            if let Some(fmtstr) = check_tts(cx, &mac.node.tts, true) {
                if fmtstr.ends_with("\\n") {
                    span_lint(cx, WRITE_WITH_NEWLINE, mac.span,
                            "using `write!()` with a format string that ends in a \
                            newline, consider using `writeln!()` instead");
                }
            }
        } else if mac.node.path == "writeln" {
            if let Some(fmtstr) = check_tts(cx, &mac.node.tts, true) {
                if fmtstr == "" {
                    span_lint_and_sugg(
                        cx,
                        WRITELN_EMPTY_STRING,
                        mac.span,
                        "using `writeln!(v, \"\")`",
                        "replace it with",
                        "writeln!(v)".to_string(),
                    );
                }
            }
        }
    }
}

fn check_tts(cx: &EarlyContext<'a>, tts: &ThinTokenStream, is_write: bool) -> Option<String> {
    let tts = TokenStream::from(tts.clone());
    let mut parser = parser::Parser::new(
        &cx.sess.parse_sess,
        tts,
        None,
        false,
        false,
    );
    if is_write {
        // skip the initial write target
        parser.parse_expr().map_err(|mut err| err.cancel()).ok()?;
        // might be `writeln!(foo)`
        parser.expect(&token::Comma).map_err(|mut err| err.cancel()).ok()?;
    }
    let fmtstr = parser.parse_str().map_err(|mut err| err.cancel()).ok()?.0.to_string();
    use fmt_macros::*;
    let tmp = fmtstr.clone();
    let mut args = vec![];
    let mut fmt_parser = Parser::new(&tmp, None);
    while let Some(piece) = fmt_parser.next() {
        if !fmt_parser.errors.is_empty() {
            return None;
        }
        if let Piece::NextArgument(arg) = piece {
            if arg.format.ty == "?" {
                // FIXME: modify rustc's fmt string parser to give us the current span
                span_lint(cx, USE_DEBUG, parser.prev_span, "use of `Debug`-based formatting");
            }
            args.push(arg);
        }
    }
    let lint = if is_write {
        WRITE_LITERAL
    } else {
        PRINT_LITERAL
    };
    let mut idx = 0;
    loop {
        if !parser.eat(&token::Comma) {
            assert!(parser.eat(&token::Eof));
            return Some(fmtstr);
        }
        let expr = parser.parse_expr().map_err(|mut err| err.cancel()).ok()?;
        const SIMPLE: FormatSpec = FormatSpec {
            fill: None,
            align: AlignUnknown,
            flags: 0,
            precision: CountImplied,
            width: CountImplied,
            ty: "",
        };
        match &expr.node {
            ExprKind::Lit(_) => {
                let mut all_simple = true;
                let mut seen = false;
                for arg in &args {
                    match arg.position {
                        | ArgumentImplicitlyIs(n)
                        | ArgumentIs(n)
                        => if n == idx {
                            all_simple &= arg.format == SIMPLE;
                            seen = true;
                        },
                        ArgumentNamed(_) => {},
                    }
                }
                if all_simple && seen {
                    span_lint(cx, lint, expr.span, "literal with an empty format string");
                }
                idx += 1;
            },
            ExprKind::Assign(lhs, rhs) => {
                if let ExprKind::Path(_, p) = &lhs.node {
                    let mut all_simple = true;
                    let mut seen = false;
                    for arg in &args {
                        match arg.position {
                            | ArgumentImplicitlyIs(_)
                            | ArgumentIs(_)
                            => {},
                            ArgumentNamed(name) => if *p == name {
                                seen = true;
                                all_simple &= arg.format == SIMPLE;
                            },
                        }
                    }
                    if all_simple && seen {
                        span_lint(cx, lint, rhs.span, "literal with an empty format string");
                    }
                }
            },
            _ => idx += 1,
        }
    }
}
