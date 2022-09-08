use std::borrow::Cow;
use std::iter;
use std::ops::{Deref, Range};

use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, snippet_opt, snippet_with_applicability};
use rustc_ast::ast::{Expr, ExprKind, Impl, Item, ItemKind, MacCall, Path, StrLit, StrStyle};
use rustc_ast::ptr::P;
use rustc_ast::token::{self, LitKind};
use rustc_ast::tokenstream::TokenStream;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_lexer::unescape::{self, EscapeError};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_parse::parser;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::{kw, Symbol};
use rustc_span::{sym, BytePos, InnerSpan, Span, DUMMY_SP};

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
    /// * Only catches `print!` and `println!` calls.
    /// * The lint level is unaffected by crate attributes. The level can still
    ///   be set for functions, modules and other items. To change the level for
    ///   the entire crate, please use command line flags. More information and a
    ///   configuration example can be found in [clippy#6610].
    ///
    /// [clippy#6610]: https://github.com/rust-lang/rust-clippy/issues/6610#issuecomment-977120558
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
    /// * Only catches `eprint!` and `eprintln!` calls.
    /// * The lint level is unaffected by crate attributes. The level can still
    ///   be set for functions, modules and other items. To change the level for
    ///   the entire crate, please use command line flags. More information and a
    ///   configuration example can be found in [clippy#6610].
    ///
    /// [clippy#6610]: https://github.com/rust-lang/rust-clippy/issues/6610#issuecomment-977120558
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
    /// Checks for use of `Debug` formatting. The purpose of this
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
    /// ### Known problems
    /// Will also warn with macro calls as arguments that expand to literals
    /// -- e.g., `println!("{}", env!("FOO"))`.
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
    /// ### Known problems
    /// Will also warn with macro calls as arguments that expand to literals
    /// -- e.g., `writeln!(buf, "{}", env!("FOO"))`.
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

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns when a named parameter in a format string is used as a positional one.
    ///
    /// ### Why is this bad?
    /// It may be confused for an assignment and obfuscates which parameter is being used.
    ///
    /// ### Example
    /// ```rust
    /// println!("{}", x = 10);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// println!("{x}", x = 10);
    /// ```
    #[clippy::version = "1.63.0"]
    pub POSITIONAL_NAMED_FORMAT_PARAMETERS,
    suspicious,
    "named parameter in a format string is used positionally"
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
    WRITE_LITERAL,
    POSITIONAL_NAMED_FORMAT_PARAMETERS,
]);

impl EarlyLintPass for Write {
    fn check_item(&mut self, _: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::Impl(box Impl {
            of_trait: Some(trait_ref),
            ..
        }) = &item.kind
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
            cx.sess()
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
            if let (Some(fmt_str), dest) = self.check_tts(cx, mac.args.inner_tokens(), true) {
                if check_newlines(&fmt_str) {
                    let (nl_span, only_nl) = newline_span(&fmt_str);
                    let nl_span = match (dest, only_nl) {
                        // Special case of `write!(buf, "\n")`: Mark everything from the end of
                        // `buf` for removal so no trailing comma [`writeln!(buf, )`] remains.
                        (Some(dest_expr), true) => nl_span.with_lo(dest_expr.span.hi()),
                        _ => nl_span,
                    };
                    span_lint_and_then(
                        cx,
                        WRITE_WITH_NEWLINE,
                        mac.span(),
                        "using `write!()` with a format string that ends in a single newline",
                        |err| {
                            err.multipart_suggestion(
                                "use `writeln!()` instead",
                                vec![(mac.path.span, String::from("writeln")), (nl_span, String::new())],
                                Applicability::MachineApplicable,
                            );
                        },
                    );
                }
            }
        } else if mac.path == sym!(writeln) {
            if let (Some(fmt_str), expr) = self.check_tts(cx, mac.args.inner_tokens(), true) {
                if fmt_str.symbol == kw::Empty {
                    let mut applicability = Applicability::MachineApplicable;
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
/// Return this and a boolean indicating whether it only consisted of a newline.
fn newline_span(fmtstr: &StrLit) -> (Span, bool) {
    let sp = fmtstr.span;
    let contents = fmtstr.symbol.as_str();

    if contents == r"\n" {
        return (sp, true);
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

    (sp.with_lo(newline_sp_hi - newline_sp_len).with_hi(newline_sp_hi), false)
}

/// Stores a list of replacement spans for each argument, but only if all the replacements used an
/// empty format string.
#[derive(Default)]
struct SimpleFormatArgs {
    unnamed: Vec<Vec<Span>>,
    complex_unnamed: Vec<Vec<Span>>,
    named: Vec<(Symbol, Vec<Span>)>,
}
impl SimpleFormatArgs {
    fn get_unnamed(&self) -> impl Iterator<Item = &[Span]> {
        self.unnamed.iter().map(|x| match x.as_slice() {
            // Ignore the dummy span added from out of order format arguments.
            [DUMMY_SP] => &[],
            x => x,
        })
    }

    fn get_complex_unnamed(&self) -> impl Iterator<Item = &[Span]> {
        self.complex_unnamed.iter().map(Vec::as_slice)
    }

    fn get_named(&self, n: &Path) -> &[Span] {
        self.named.iter().find(|x| *n == x.0).map_or(&[], |x| x.1.as_slice())
    }

    fn push(&mut self, arg: rustc_parse_format::Argument<'_>, span: Span) {
        use rustc_parse_format::{
            AlignUnknown, ArgumentImplicitlyIs, ArgumentIs, ArgumentNamed, CountImplied, FormatSpec,
        };

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

        match arg.position {
            ArgumentIs(n) | ArgumentImplicitlyIs(n) => {
                if self.unnamed.len() <= n {
                    // Use a dummy span to mark all unseen arguments.
                    self.unnamed.resize_with(n, || vec![DUMMY_SP]);
                    if arg.format == SIMPLE {
                        self.unnamed.push(vec![span]);
                    } else {
                        self.unnamed.push(Vec::new());
                    }
                } else {
                    let args = &mut self.unnamed[n];
                    match (args.as_mut_slice(), arg.format == SIMPLE) {
                        // A non-empty format string has been seen already.
                        ([], _) => (),
                        // Replace the dummy span, if it exists.
                        ([dummy @ DUMMY_SP], true) => *dummy = span,
                        ([_, ..], true) => args.push(span),
                        ([_, ..], false) => *args = Vec::new(),
                    }
                }
            },
            ArgumentNamed(n) => {
                let n = Symbol::intern(n);
                if let Some(x) = self.named.iter_mut().find(|x| x.0 == n) {
                    match x.1.as_slice() {
                        // A non-empty format string has been seen already.
                        [] => (),
                        [_, ..] if arg.format == SIMPLE => x.1.push(span),
                        [_, ..] => x.1 = Vec::new(),
                    }
                } else if arg.format == SIMPLE {
                    self.named.push((n, vec![span]));
                } else {
                    self.named.push((n, Vec::new()));
                }
            },
        };
    }

    fn push_to_complex(&mut self, span: Span, position: usize) {
        if self.complex_unnamed.len() <= position {
            self.complex_unnamed.resize_with(position, Vec::new);
            self.complex_unnamed.push(vec![span]);
        } else {
            let args: &mut Vec<Span> = &mut self.complex_unnamed[position];
            args.push(span);
        }
    }

    fn push_complex(
        &mut self,
        cx: &EarlyContext<'_>,
        arg: rustc_parse_format::Argument<'_>,
        str_lit_span: Span,
        fmt_span: Span,
    ) {
        use rustc_parse_format::{ArgumentImplicitlyIs, ArgumentIs, CountIsParam, CountIsStar};

        let snippet = snippet_opt(cx, fmt_span);

        let end = snippet
            .as_ref()
            .and_then(|s| s.find(':'))
            .or_else(|| fmt_span.hi().0.checked_sub(fmt_span.lo().0 + 1).map(|u| u as usize));

        if let (ArgumentIs(n) | ArgumentImplicitlyIs(n), Some(end)) = (arg.position, end) {
            let span = fmt_span.from_inner(InnerSpan::new(1, end));
            self.push_to_complex(span, n);
        };

        if let (CountIsParam(n) | CountIsStar(n), Some(span)) = (arg.format.precision, arg.format.precision_span) {
            // We need to do this hack as precision spans should be converted from .* to .foo$
            let hack = if snippet.as_ref().and_then(|s| s.find('*')).is_some() {
                0
            } else {
                1
            };

            let span = str_lit_span.from_inner(InnerSpan {
                start: span.start + 1,
                end: span.end - hack,
            });
            self.push_to_complex(span, n);
        };

        if let (CountIsParam(n), Some(span)) = (arg.format.width, arg.format.width_span) {
            let span = str_lit_span.from_inner(InnerSpan {
                start: span.start,
                end: span.end - 1,
            });
            self.push_to_complex(span, n);
        };
    }
}

impl Write {
    /// Parses a format string into a collection of spans for each argument. This only keeps track
    /// of empty format arguments. Will also lint usages of debug format strings outside of debug
    /// impls.
    fn parse_fmt_string(&self, cx: &EarlyContext<'_>, str_lit: &StrLit) -> Option<SimpleFormatArgs> {
        use rustc_parse_format::{ParseMode, Parser, Piece};

        let str_sym = str_lit.symbol_unescaped.as_str();
        let style = match str_lit.style {
            StrStyle::Cooked => None,
            StrStyle::Raw(n) => Some(n as usize),
        };

        let mut parser = Parser::new(str_sym, style, snippet_opt(cx, str_lit.span), false, ParseMode::Format);
        let mut args = SimpleFormatArgs::default();

        while let Some(arg) = parser.next() {
            let arg = match arg {
                Piece::String(_) => continue,
                Piece::NextArgument(arg) => arg,
            };
            let span = parser
                .arg_places
                .last()
                .map_or(DUMMY_SP, |&x| str_lit.span.from_inner(InnerSpan::new(x.start, x.end)));

            if !self.in_debug_impl && arg.format.ty == "?" {
                // FIXME: modify rustc's fmt string parser to give us the current span
                span_lint(cx, USE_DEBUG, span, "use of `Debug`-based formatting");
            }
            args.push(arg, span);
            args.push_complex(cx, arg, str_lit.span, span);
        }

        parser.errors.is_empty().then_some(args)
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
    fn check_tts<'a>(&self, cx: &EarlyContext<'a>, tts: TokenStream, is_write: bool) -> (Option<StrLit>, Option<Expr>) {
        let mut parser = parser::Parser::new(&cx.sess().parse_sess, tts, false, None);
        let expr = if is_write {
            match parser
                .parse_expr()
                .map(rustc_ast::ptr::P::into_inner)
                .map_err(DiagnosticBuilder::cancel)
            {
                // write!(e, ...)
                Ok(p) if parser.eat(&token::Comma) => Some(p),
                // write!(e) or error
                e => return (None, e.ok()),
            }
        } else {
            None
        };

        let fmtstr = match parser.parse_str_lit() {
            Ok(fmtstr) => fmtstr,
            Err(_) => return (None, expr),
        };

        let args = match self.parse_fmt_string(cx, &fmtstr) {
            Some(args) => args,
            None => return (Some(fmtstr), expr),
        };

        let lint = if is_write { WRITE_LITERAL } else { PRINT_LITERAL };
        let mut unnamed_args = args.get_unnamed();
        let mut complex_unnamed_args = args.get_complex_unnamed();
        loop {
            if !parser.eat(&token::Comma) {
                return (Some(fmtstr), expr);
            }

            let comma_span = parser.prev_token.span;
            let token_expr = if let Ok(expr) = parser.parse_expr().map_err(DiagnosticBuilder::cancel) {
                expr
            } else {
                return (Some(fmtstr), None);
            };
            let complex_unnamed_arg = complex_unnamed_args.next();

            let (fmt_spans, lit) = match &token_expr.kind {
                ExprKind::Lit(lit) => (unnamed_args.next().unwrap_or(&[]), lit),
                ExprKind::Assign(lhs, rhs, _) => {
                    if let Some(span) = complex_unnamed_arg {
                        for x in span {
                            Self::report_positional_named_param(cx, *x, lhs, rhs);
                        }
                    }
                    match (&lhs.kind, &rhs.kind) {
                        (ExprKind::Path(_, p), ExprKind::Lit(lit)) => (args.get_named(p), lit),
                        _ => continue,
                    }
                },
                _ => {
                    unnamed_args.next();
                    continue;
                },
            };

            let replacement: String = match lit.token_lit.kind {
                LitKind::StrRaw(_) | LitKind::ByteStrRaw(_) if matches!(fmtstr.style, StrStyle::Raw(_)) => {
                    lit.token_lit.symbol.as_str().replace('{', "{{").replace('}', "}}")
                },
                LitKind::Str | LitKind::ByteStr if matches!(fmtstr.style, StrStyle::Cooked) => {
                    lit.token_lit.symbol.as_str().replace('{', "{{").replace('}', "}}")
                },
                LitKind::StrRaw(_)
                | LitKind::Str
                | LitKind::ByteStrRaw(_)
                | LitKind::ByteStr
                | LitKind::Integer
                | LitKind::Float
                | LitKind::Err => continue,
                LitKind::Byte | LitKind::Char => match lit.token_lit.symbol.as_str() {
                    "\"" if matches!(fmtstr.style, StrStyle::Cooked) => "\\\"",
                    "\"" if matches!(fmtstr.style, StrStyle::Raw(0)) => continue,
                    "\\\\" if matches!(fmtstr.style, StrStyle::Raw(_)) => "\\",
                    "\\'" => "'",
                    "{" => "{{",
                    "}" => "}}",
                    x if matches!(fmtstr.style, StrStyle::Raw(_)) && x.starts_with('\\') => continue,
                    x => x,
                }
                .into(),
                LitKind::Bool => lit.token_lit.symbol.as_str().deref().into(),
            };

            if !fmt_spans.is_empty() {
                span_lint_and_then(
                    cx,
                    lint,
                    token_expr.span,
                    "literal with an empty format string",
                    |diag| {
                        diag.multipart_suggestion(
                            "try this",
                            iter::once((comma_span.to(token_expr.span), String::new()))
                                .chain(fmt_spans.iter().copied().zip(iter::repeat(replacement)))
                                .collect(),
                            Applicability::MachineApplicable,
                        );
                    },
                );
            }
        }
    }

    fn report_positional_named_param(cx: &EarlyContext<'_>, span: Span, lhs: &P<Expr>, _rhs: &P<Expr>) {
        if let ExprKind::Path(_, _p) = &lhs.kind {
            let mut applicability = Applicability::MachineApplicable;
            let name = snippet_with_applicability(cx, lhs.span, "name", &mut applicability);
            // We need to do this hack as precision spans should be converted from .* to .foo$
            let hack = snippet(cx, span, "").contains('*');

            span_lint_and_sugg(
                cx,
                POSITIONAL_NAMED_FORMAT_PARAMETERS,
                span,
                &format!("named parameter {} is used as a positional parameter", name),
                "replace it with",
                if hack {
                    format!("{}$", name)
                } else {
                    format!("{}", name)
                },
                applicability,
            );
        };
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
                            vec![(mac.path.span, suggested), (newline_span(&fmt_str).0, String::new())],
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

    let contents = fmtstr.symbol.as_str();

    let mut cb = |r: Range<usize>, c: Result<char, EscapeError>| {
        let c = match c {
            Ok(c) => c,
            Err(e) if !e.is_fatal() => return,
            Err(e) => panic!("{:?}", e),
        };

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
