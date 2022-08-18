use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::macros::{is_format_macro, root_macro_call_first_node, FormatArg, FormatArgsExpn};
use clippy_utils::{get_parent_as_impl, is_diag_trait_item, path_to_local, peel_ref_operators};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Impl, ImplItem, ImplItemKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, symbol::kw, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for format trait implementations (e.g. `Display`) with a recursive call to itself
    /// which uses `self` as a parameter.
    /// This is typically done indirectly with the `write!` macro or with `to_string()`.
    ///
    /// ### Why is this bad?
    /// This will lead to infinite recursion and a stack overflow.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Structure(i32);
    /// impl fmt::Display for Structure {
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ///         write!(f, "{}", self.to_string())
    ///     }
    /// }
    ///
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::fmt;
    ///
    /// struct Structure(i32);
    /// impl fmt::Display for Structure {
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ///         write!(f, "{}", self.0)
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub RECURSIVE_FORMAT_IMPL,
    correctness,
    "Format trait method called while implementing the same Format trait"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of `println`, `print`, `eprintln` or `eprint` in an
    /// implementation of a formatting trait.
    ///
    /// ### Why is this bad?
    /// Using a print macro is likely unintentional since formatting traits
    /// should write to the `Formatter`, not stdout/stderr.
    ///
    /// ### Example
    /// ```rust
    /// use std::fmt::{Display, Error, Formatter};
    ///
    /// struct S;
    /// impl Display for S {
    ///     fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    ///         println!("S");
    ///
    ///         Ok(())
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::fmt::{Display, Error, Formatter};
    ///
    /// struct S;
    /// impl Display for S {
    ///     fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    ///         writeln!(f, "S");
    ///
    ///         Ok(())
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.61.0"]
    pub PRINT_IN_FORMAT_IMPL,
    suspicious,
    "use of a print macro in a formatting trait impl"
}

#[derive(Clone, Copy)]
struct FormatTrait {
    /// e.g. `sym::Display`
    name: Symbol,
    /// `f` in `fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {}`
    formatter_name: Option<Symbol>,
}

#[derive(Default)]
pub struct FormatImpl {
    // Whether we are inside Display or Debug trait impl - None for neither
    format_trait_impl: Option<FormatTrait>,
}

impl FormatImpl {
    pub fn new() -> Self {
        Self {
            format_trait_impl: None,
        }
    }
}

impl_lint_pass!(FormatImpl => [RECURSIVE_FORMAT_IMPL, PRINT_IN_FORMAT_IMPL]);

impl<'tcx> LateLintPass<'tcx> for FormatImpl {
    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &ImplItem<'_>) {
        self.format_trait_impl = is_format_trait_impl(cx, impl_item);
    }

    fn check_impl_item_post(&mut self, cx: &LateContext<'_>, impl_item: &ImplItem<'_>) {
        // Assume no nested Impl of Debug and Display within eachother
        if is_format_trait_impl(cx, impl_item).is_some() {
            self.format_trait_impl = None;
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some(format_trait_impl) = self.format_trait_impl else { return };

        if format_trait_impl.name == sym::Display {
            check_to_string_in_display(cx, expr);
        }

        check_self_in_format_args(cx, expr, format_trait_impl);
        check_print_in_format_impl(cx, expr, format_trait_impl);
    }
}

fn check_to_string_in_display(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if_chain! {
        // Get the hir_id of the object we are calling the method on
        if let ExprKind::MethodCall(path, [ref self_arg, ..], _) = expr.kind;
        // Is the method to_string() ?
        if path.ident.name == sym::to_string;
        // Is the method a part of the ToString trait? (i.e. not to_string() implemented
        // separately)
        if let Some(expr_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if is_diag_trait_item(cx, expr_def_id, sym::ToString);
        // Is the method is called on self
        if let ExprKind::Path(QPath::Resolved(_, path)) = self_arg.kind;
        if let [segment] = path.segments;
        if segment.ident.name == kw::SelfLower;
        then {
            span_lint(
                cx,
                RECURSIVE_FORMAT_IMPL,
                expr.span,
                "using `self.to_string` in `fmt::Display` implementation will cause infinite recursion",
            );
        }
    }
}

fn check_self_in_format_args<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, impl_trait: FormatTrait) {
    // Check each arg in format calls - do we ever use Display on self (directly or via deref)?
    if_chain! {
        if let Some(outer_macro) = root_macro_call_first_node(cx, expr);
        if let macro_def_id = outer_macro.def_id;
        if let Some(format_args) = FormatArgsExpn::find_nested(cx, expr, outer_macro.expn);
        if is_format_macro(cx, macro_def_id);
        then {
            for arg in format_args.args {
                if arg.format.r#trait != impl_trait.name {
                    continue;
                }
                check_format_arg_self(cx, expr, &arg, impl_trait);
            }
        }
    }
}

fn check_format_arg_self(cx: &LateContext<'_>, expr: &Expr<'_>, arg: &FormatArg<'_>, impl_trait: FormatTrait) {
    // Handle multiple dereferencing of references e.g. &&self
    // Handle dereference of &self -> self that is equivalent (i.e. via *self in fmt() impl)
    // Since the argument to fmt is itself a reference: &self
    let reference = peel_ref_operators(cx, arg.param.value);
    let map = cx.tcx.hir();
    // Is the reference self?
    if path_to_local(reference).map(|x| map.name(x)) == Some(kw::SelfLower) {
        let FormatTrait { name, .. } = impl_trait;
        span_lint(
            cx,
            RECURSIVE_FORMAT_IMPL,
            expr.span,
            &format!("using `self` as `{name}` in `impl {name}` will cause infinite recursion"),
        );
    }
}

fn check_print_in_format_impl(cx: &LateContext<'_>, expr: &Expr<'_>, impl_trait: FormatTrait) {
    if_chain! {
        if let Some(macro_call) = root_macro_call_first_node(cx, expr);
        if let Some(name) = cx.tcx.get_diagnostic_name(macro_call.def_id);
        then {
            let replacement = match name {
                sym::print_macro | sym::eprint_macro => "write",
                sym::println_macro | sym::eprintln_macro => "writeln",
                _ => return,
            };

            let name = name.as_str().strip_suffix("_macro").unwrap();

            span_lint_and_sugg(
                cx,
                PRINT_IN_FORMAT_IMPL,
                macro_call.span,
                &format!("use of `{}!` in `{}` impl", name, impl_trait.name),
                "replace with",
                if let Some(formatter_name) = impl_trait.formatter_name {
                    format!("{}!({}, ..)", replacement, formatter_name)
                } else {
                    format!("{}!(..)", replacement)
                },
                Applicability::HasPlaceholders,
            );
        }
    }
}

fn is_format_trait_impl(cx: &LateContext<'_>, impl_item: &ImplItem<'_>) -> Option<FormatTrait> {
    if_chain! {
        if impl_item.ident.name == sym::fmt;
        if let ImplItemKind::Fn(_, body_id) = impl_item.kind;
        if let Some(Impl { of_trait: Some(trait_ref),..}) = get_parent_as_impl(cx.tcx, impl_item.hir_id());
        if let Some(did) = trait_ref.trait_def_id();
        if let Some(name) = cx.tcx.get_diagnostic_name(did);
        if matches!(name, sym::Debug | sym::Display);
        then {
            let body = cx.tcx.hir().body(body_id);
            let formatter_name = body.params.get(1)
                .and_then(|param| param.pat.simple_ident())
                .map(|ident| ident.name);

            Some(FormatTrait {
                name,
                formatter_name,
            })
        } else {
            None
        }
    }
}
