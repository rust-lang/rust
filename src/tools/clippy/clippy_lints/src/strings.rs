use clippy_utils::diagnostics::{span_lint, span_lint_and_help, span_lint_and_sugg};
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::ty::is_type_lang_item;
use clippy_utils::{get_expr_use_or_unification_node, peel_blocks, SpanlessEq};
use clippy_utils::{get_parent_expr, is_lint_allowed, is_path_diagnostic_item, method_calls};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{BinOpKind, BorrowKind, Expr, ExprKind, LangItem, Node, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for string appends of the form `x = x + y` (without
    /// `let`!).
    ///
    /// ### Why is this bad?
    /// It's not really bad, but some people think that the
    /// `.push_str(_)` method is more readable.
    ///
    /// ### Example
    /// ```rust
    /// let mut x = "Hello".to_owned();
    /// x = x + ", World";
    ///
    /// // More readable
    /// x += ", World";
    /// x.push_str(", World");
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub STRING_ADD_ASSIGN,
    pedantic,
    "using `x = x + ..` where x is a `String` instead of `push_str()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for all instances of `x + _` where `x` is of type
    /// `String`, but only if [`string_add_assign`](#string_add_assign) does *not*
    /// match.
    ///
    /// ### Why is this bad?
    /// It's not bad in and of itself. However, this particular
    /// `Add` implementation is asymmetric (the other operand need not be `String`,
    /// but `x` does), while addition as mathematically defined is symmetric, also
    /// the `String::push_str(_)` function is a perfectly good replacement.
    /// Therefore, some dislike it and wish not to have it in their code.
    ///
    /// That said, other people think that string addition, having a long tradition
    /// in other languages is actually fine, which is why we decided to make this
    /// particular lint `allow` by default.
    ///
    /// ### Example
    /// ```rust
    /// let x = "Hello".to_owned();
    /// x + ", World";
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let mut x = "Hello".to_owned();
    /// x.push_str(", World");
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub STRING_ADD,
    restriction,
    "using `x + ..` where x is a `String` instead of `push_str()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the `as_bytes` method called on string literals
    /// that contain only ASCII characters.
    ///
    /// ### Why is this bad?
    /// Byte string literals (e.g., `b"foo"`) can be used
    /// instead. They are shorter but less discoverable than `as_bytes()`.
    ///
    /// ### Known problems
    /// `"str".as_bytes()` and the suggested replacement of `b"str"` are not
    /// equivalent because they have different types. The former is `&[u8]`
    /// while the latter is `&[u8; 3]`. That means in general they will have a
    /// different set of methods and different trait implementations.
    ///
    /// ```compile_fail
    /// fn f(v: Vec<u8>) {}
    ///
    /// f("...".as_bytes().to_owned()); // works
    /// f(b"...".to_owned()); // does not work, because arg is [u8; 3] not Vec<u8>
    ///
    /// fn g(r: impl std::io::Read) {}
    ///
    /// g("...".as_bytes()); // works
    /// g(b"..."); // does not work
    /// ```
    ///
    /// The actual equivalent of `"str".as_bytes()` with the same type is not
    /// `b"str"` but `&b"str"[..]`, which is a great deal of punctuation and not
    /// more readable than a function call.
    ///
    /// ### Example
    /// ```rust
    /// let bstr = "a byte string".as_bytes();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let bstr = b"a byte string";
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub STRING_LIT_AS_BYTES,
    nursery,
    "calling `as_bytes` on a string literal instead of using a byte string literal"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for slice operations on strings
    ///
    /// ### Why is this bad?
    /// UTF-8 characters span multiple bytes, and it is easy to inadvertently confuse character
    /// counts and string indices. This may lead to panics, and should warrant some test cases
    /// containing wide UTF-8 characters. This lint is most useful in code that should avoid
    /// panics at all costs.
    ///
    /// ### Known problems
    /// Probably lots of false positives. If an index comes from a known valid position (e.g.
    /// obtained via `char_indices` over the same string), it is totally OK.
    ///
    /// ### Example
    /// ```rust,should_panic
    /// &"Ölkanne"[1..];
    /// ```
    #[clippy::version = "1.58.0"]
    pub STRING_SLICE,
    restriction,
    "slicing a string"
}

declare_lint_pass!(StringAdd => [STRING_ADD, STRING_ADD_ASSIGN, STRING_SLICE]);

impl<'tcx> LateLintPass<'tcx> for StringAdd {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if in_external_macro(cx.sess(), e.span) {
            return;
        }
        match e.kind {
            ExprKind::Binary(
                Spanned {
                    node: BinOpKind::Add, ..
                },
                left,
                _,
            ) => {
                if is_string(cx, left) {
                    if !is_lint_allowed(cx, STRING_ADD_ASSIGN, e.hir_id) {
                        let parent = get_parent_expr(cx, e);
                        if let Some(p) = parent {
                            if let ExprKind::Assign(target, _, _) = p.kind {
                                // avoid duplicate matches
                                if SpanlessEq::new(cx).eq_expr(target, left) {
                                    return;
                                }
                            }
                        }
                    }
                    span_lint(
                        cx,
                        STRING_ADD,
                        e.span,
                        "you added something to a string. Consider using `String::push_str()` instead",
                    );
                }
            },
            ExprKind::Assign(target, src, _) => {
                if is_string(cx, target) && is_add(cx, src, target) {
                    span_lint(
                        cx,
                        STRING_ADD_ASSIGN,
                        e.span,
                        "you assigned the result of adding something to this string. Consider using \
                         `String::push_str()` instead",
                    );
                }
            },
            ExprKind::Index(target, _idx) => {
                let e_ty = cx.typeck_results().expr_ty(target).peel_refs();
                if e_ty.is_str() || is_type_lang_item(cx, e_ty, LangItem::String) {
                    span_lint(
                        cx,
                        STRING_SLICE,
                        e.span,
                        "indexing into a string may panic if the index is within a UTF-8 character",
                    );
                }
            },
            _ => {},
        }
    }
}

fn is_string(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    is_type_lang_item(cx, cx.typeck_results().expr_ty(e).peel_refs(), LangItem::String)
}

fn is_add(cx: &LateContext<'_>, src: &Expr<'_>, target: &Expr<'_>) -> bool {
    match peel_blocks(src).kind {
        ExprKind::Binary(
            Spanned {
                node: BinOpKind::Add, ..
            },
            left,
            _,
        ) => SpanlessEq::new(cx).eq_expr(target, left),
        _ => false,
    }
}

declare_clippy_lint! {
    /// ### What it does
    /// Check if the string is transformed to byte array and casted back to string.
    ///
    /// ### Why is this bad?
    /// It's unnecessary, the string can be used directly.
    ///
    /// ### Example
    /// ```rust
    /// std::str::from_utf8(&"Hello World!".as_bytes()[6..11]).unwrap();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// &"Hello World!"[6..11];
    /// ```
    #[clippy::version = "1.50.0"]
    pub STRING_FROM_UTF8_AS_BYTES,
    complexity,
    "casting string slices to byte slices and back"
}

// Max length a b"foo" string can take
const MAX_LENGTH_BYTE_STRING_LIT: usize = 32;

declare_lint_pass!(StringLitAsBytes => [STRING_LIT_AS_BYTES, STRING_FROM_UTF8_AS_BYTES]);

impl<'tcx> LateLintPass<'tcx> for StringLitAsBytes {
    #[expect(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        use rustc_ast::LitKind;

        if_chain! {
            // Find std::str::converts::from_utf8
            if let ExprKind::Call(fun, args) = e.kind;
            if is_path_diagnostic_item(cx, fun, sym::str_from_utf8);

            // Find string::as_bytes
            if let ExprKind::AddrOf(BorrowKind::Ref, _, args) = args[0].kind;
            if let ExprKind::Index(left, right) = args.kind;
            let (method_names, expressions, _) = method_calls(left, 1);
            if method_names.len() == 1;
            if expressions.len() == 1;
            if expressions[0].1.is_empty();
            if method_names[0] == sym!(as_bytes);

            // Check for slicer
            if let ExprKind::Struct(QPath::LangItem(LangItem::Range, ..), _, _) = right.kind;

            then {
                let mut applicability = Applicability::MachineApplicable;
                let string_expression = &expressions[0].0;

                let snippet_app = snippet_with_applicability(
                    cx,
                    string_expression.span, "..",
                    &mut applicability,
                );

                span_lint_and_sugg(
                    cx,
                    STRING_FROM_UTF8_AS_BYTES,
                    e.span,
                    "calling a slice of `as_bytes()` with `from_utf8` should be not necessary",
                    "try",
                    format!("Some(&{snippet_app}[{}])", snippet(cx, right.span, "..")),
                    applicability
                )
            }
        }

        if_chain! {
            if !in_external_macro(cx.sess(), e.span);
            if let ExprKind::MethodCall(path, receiver, ..) = &e.kind;
            if path.ident.name == sym!(as_bytes);
            if let ExprKind::Lit(lit) = &receiver.kind;
            if let LitKind::Str(lit_content, _) = &lit.node;
            then {
                let callsite = snippet(cx, receiver.span.source_callsite(), r#""foo""#);
                let mut applicability = Applicability::MachineApplicable;
                if callsite.starts_with("include_str!") {
                    span_lint_and_sugg(
                        cx,
                        STRING_LIT_AS_BYTES,
                        e.span,
                        "calling `as_bytes()` on `include_str!(..)`",
                        "consider using `include_bytes!(..)` instead",
                        snippet_with_applicability(cx, receiver.span, r#""foo""#, &mut applicability).replacen(
                            "include_str",
                            "include_bytes",
                            1,
                        ),
                        applicability,
                    );
                } else if lit_content.as_str().is_ascii()
                    && lit_content.as_str().len() <= MAX_LENGTH_BYTE_STRING_LIT
                    && !receiver.span.from_expansion()
                {
                    if let Some((parent, id)) = get_expr_use_or_unification_node(cx.tcx, e)
                        && let Node::Expr(parent) = parent
                        && let ExprKind::Match(scrutinee, ..) = parent.kind
                        && scrutinee.hir_id == id
                    {
                        // Don't lint. Byte strings produce `&[u8; N]` whereas `as_bytes()` produces
                        // `&[u8]`. This change would prevent matching with different sized slices.
                    } else {
                        span_lint_and_sugg(
                            cx,
                            STRING_LIT_AS_BYTES,
                            e.span,
                            "calling `as_bytes()` on a string literal",
                            "consider using a byte string literal instead",
                            format!(
                                "b{}",
                                snippet_with_applicability(cx, receiver.span, r#""foo""#, &mut applicability)
                            ),
                            applicability,
                        );
                    }
                }
            }
        }

        if_chain! {
            if let ExprKind::MethodCall(path, recv, [], _) = &e.kind;
            if path.ident.name == sym!(into_bytes);
            if let ExprKind::MethodCall(path, recv, [], _) = &recv.kind;
            if matches!(path.ident.name.as_str(), "to_owned" | "to_string");
            if let ExprKind::Lit(lit) = &recv.kind;
            if let LitKind::Str(lit_content, _) = &lit.node;

            if lit_content.as_str().is_ascii();
            if lit_content.as_str().len() <= MAX_LENGTH_BYTE_STRING_LIT;
            if !recv.span.from_expansion();
            then {
                let mut applicability = Applicability::MachineApplicable;

                span_lint_and_sugg(
                    cx,
                    STRING_LIT_AS_BYTES,
                    e.span,
                    "calling `into_bytes()` on a string literal",
                    "consider using a byte string literal instead",
                    format!(
                        "b{}.to_vec()",
                        snippet_with_applicability(cx, recv.span, r#""..""#, &mut applicability)
                    ),
                    applicability,
                );
            }
        }
    }
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for `.to_string()` method calls on values of type `&str`.
    ///
    /// ### Why is this bad?
    /// The `to_string` method is also used on other types to convert them to a string.
    /// When called on a `&str` it turns the `&str` into the owned variant `String`, which can be better
    /// expressed with `.to_owned()`.
    ///
    /// ### Example
    /// ```rust
    /// // example code where clippy issues a warning
    /// let _ = "str".to_string();
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise clippy warning
    /// let _ = "str".to_owned();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub STR_TO_STRING,
    restriction,
    "using `to_string()` on a `&str`, which should be `to_owned()`"
}

declare_lint_pass!(StrToString => [STR_TO_STRING]);

impl<'tcx> LateLintPass<'tcx> for StrToString {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(path, self_arg, ..) = &expr.kind;
            if path.ident.name == sym::to_string;
            let ty = cx.typeck_results().expr_ty(self_arg);
            if let ty::Ref(_, ty, ..) = ty.kind();
            if ty.is_str();
            then {
                span_lint_and_help(
                    cx,
                    STR_TO_STRING,
                    expr.span,
                    "`to_string()` called on a `&str`",
                    None,
                    "consider using `.to_owned()`",
                );
            }
        }
    }
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for `.to_string()` method calls on values of type `String`.
    ///
    /// ### Why is this bad?
    /// The `to_string` method is also used on other types to convert them to a string.
    /// When called on a `String` it only clones the `String`, which can be better expressed with `.clone()`.
    ///
    /// ### Example
    /// ```rust
    /// // example code where clippy issues a warning
    /// let msg = String::from("Hello World");
    /// let _ = msg.to_string();
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise clippy warning
    /// let msg = String::from("Hello World");
    /// let _ = msg.clone();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub STRING_TO_STRING,
    restriction,
    "using `to_string()` on a `String`, which should be `clone()`"
}

declare_lint_pass!(StringToString => [STRING_TO_STRING]);

impl<'tcx> LateLintPass<'tcx> for StringToString {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(path, self_arg, ..) = &expr.kind;
            if path.ident.name == sym::to_string;
            let ty = cx.typeck_results().expr_ty(self_arg);
            if is_type_lang_item(cx, ty, LangItem::String);
            then {
                span_lint_and_help(
                    cx,
                    STRING_TO_STRING,
                    expr.span,
                    "`to_string()` called on a `String`",
                    None,
                    "consider using `.clone()`",
                );
            }
        }
    }
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns about calling `str::trim` (or variants) before `str::split_whitespace`.
    ///
    /// ### Why is this bad?
    /// `split_whitespace` already ignores leading and trailing whitespace.
    ///
    /// ### Example
    /// ```rust
    /// " A B C ".trim().split_whitespace();
    /// ```
    /// Use instead:
    /// ```rust
    /// " A B C ".split_whitespace();
    /// ```
    #[clippy::version = "1.62.0"]
    pub TRIM_SPLIT_WHITESPACE,
    style,
    "using `str::trim()` or alike before `str::split_whitespace`"
}
declare_lint_pass!(TrimSplitWhitespace => [TRIM_SPLIT_WHITESPACE]);

impl<'tcx> LateLintPass<'tcx> for TrimSplitWhitespace {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        let tyckres = cx.typeck_results();
        if_chain! {
            if let ExprKind::MethodCall(path, split_recv, [], split_ws_span) = expr.kind;
            if path.ident.name == sym!(split_whitespace);
            if let Some(split_ws_def_id) = tyckres.type_dependent_def_id(expr.hir_id);
            if cx.tcx.is_diagnostic_item(sym::str_split_whitespace, split_ws_def_id);
            if let ExprKind::MethodCall(path, _trim_recv, [], trim_span) = split_recv.kind;
            if let trim_fn_name @ ("trim" | "trim_start" | "trim_end") = path.ident.name.as_str();
            if let Some(trim_def_id) = tyckres.type_dependent_def_id(split_recv.hir_id);
            if is_one_of_trim_diagnostic_items(cx, trim_def_id);
            then {
                span_lint_and_sugg(
                    cx,
                    TRIM_SPLIT_WHITESPACE,
                    trim_span.with_hi(split_ws_span.lo()),
                    &format!("found call to `str::{trim_fn_name}` before `str::split_whitespace`"),
                    &format!("remove `{trim_fn_name}()`"),
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

fn is_one_of_trim_diagnostic_items(cx: &LateContext<'_>, trim_def_id: DefId) -> bool {
    cx.tcx.is_diagnostic_item(sym::str_trim, trim_def_id)
        || cx.tcx.is_diagnostic_item(sym::str_trim_start, trim_def_id)
        || cx.tcx.is_diagnostic_item(sym::str_trim_end, trim_def_id)
}
