// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use crate::syntax::ast::*;
use crate::syntax::parse::{parser, token};
use crate::syntax::tokenstream::{ThinTokenStream, TokenStream};
use crate::utils::{snippet_with_applicability, span_lint, span_lint_and_sugg};
use std::borrow::Cow;

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
/// use println!() instead
/// ```rust
/// println!("Hello {}!", name);
/// ```
declare_clippy_lint! {
    pub PRINT_WITH_NEWLINE,
    style,
    "using `print!()` with a format string that ends in a single newline"
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
/// use the literal without formatting:
/// ```rust
/// println!("foo");
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
    "using `write!()` with a format string that ends in a single newline"
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
    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac: &Mac) {
        if mac.node.path == "println" {
            span_lint(cx, PRINT_STDOUT, mac.span, "use of `println!`");
            if let Some(fmtstr) = check_tts(cx, &mac.node.tts, false).0 {
                if fmtstr == "" {
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
        } else if mac.node.path == "print" {
            span_lint(cx, PRINT_STDOUT, mac.span, "use of `print!`");
            if let Some(fmtstr) = check_tts(cx, &mac.node.tts, false).0 {
                if fmtstr.ends_with("\\n") &&
                   // don't warn about strings with several `\n`s (#3126)
                   fmtstr.matches("\\n").count() == 1
                {
                    span_lint(
                        cx,
                        PRINT_WITH_NEWLINE,
                        mac.span,
                        "using `print!()` with a format string that ends in a \
                         single newline, consider using `println!()` instead",
                    );
                }
            }
        } else if mac.node.path == "write" {
            if let Some(fmtstr) = check_tts(cx, &mac.node.tts, true).0 {
                if fmtstr.ends_with("\\n") &&
                   // don't warn about strings with several `\n`s (#3126)
                   fmtstr.matches("\\n").count() == 1
                {
                    span_lint(
                        cx,
                        WRITE_WITH_NEWLINE,
                        mac.span,
                        "using `write!()` with a format string that ends in a \
                         single newline, consider using `writeln!()` instead",
                    );
                }
            }
        } else if mac.node.path == "writeln" {
            let check_tts = check_tts(cx, &mac.node.tts, true);
            if let Some(fmtstr) = check_tts.0 {
                if fmtstr == "" {
                    let mut applicability = Applicability::MachineApplicable;
                    let suggestion = check_tts.1.map_or_else(
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

/// Checks the arguments of `print[ln]!` and `write[ln]!` calls. It will return a tuple of two
/// options. The first part of the tuple is `format_str` of the macros. The secund part of the tuple
/// is in the `write[ln]!` case the expression the `format_str` should be written to.
///
/// Example:
///
/// Calling this function on
/// ```rust,ignore
/// writeln!(buf, "string to write: {}", something)
/// ```
/// will return
/// ```rust,ignore
/// (Some("string to write: {}"), Some(buf))
/// ```
fn check_tts<'a>(cx: &EarlyContext<'a>, tts: &ThinTokenStream, is_write: bool) -> (Option<String>, Option<Expr>) {
    use crate::fmt_macros::*;
    let tts = TokenStream::from(tts.clone());
    let mut parser = parser::Parser::new(&cx.sess.parse_sess, tts, None, false, false);
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

    let fmtstr = match parser.parse_str().map_err(|mut err| err.cancel()) {
        Ok(token) => token.0.to_string(),
        Err(_) => return (None, expr),
    };
    let tmp = fmtstr.clone();
    let mut args = vec![];
    let mut fmt_parser = Parser::new(&tmp, None);
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
            width: CountImplied,
            ty: "",
        };
        if !parser.eat(&token::Comma) {
            return (Some(fmtstr), expr);
        }
        let token_expr = match parser.parse_expr().map_err(|mut err| err.cancel()) {
            Ok(expr) => expr,
            Err(_) => return (Some(fmtstr), None),
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
