use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::macros::{FormatArgsStorage, find_format_arg_expr, is_format_macro, root_macro_call_first_node};
use clippy_utils::{get_parent_as_impl, is_diag_trait_item, path_to_local, peel_ref_operators, sym};
use rustc_ast::{FormatArgsPiece, FormatTrait};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Impl, ImplItem, ImplItemKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Symbol;
use rustc_span::symbol::kw;

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
    /// ```no_run
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
    /// ```no_run
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
    /// Checks for usage of `println`, `print`, `eprintln` or `eprint` in an
    /// implementation of a formatting trait.
    ///
    /// ### Why is this bad?
    /// Using a print macro is likely unintentional since formatting traits
    /// should write to the `Formatter`, not stdout/stderr.
    ///
    /// ### Example
    /// ```no_run
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
    /// ```no_run
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
struct FormatTraitNames {
    /// e.g. `sym::Display`
    name: Symbol,
    /// `f` in `fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {}`
    formatter_name: Option<Symbol>,
}

pub struct FormatImpl {
    format_args: FormatArgsStorage,
    // Whether we are inside Display or Debug trait impl - None for neither
    format_trait_impl: Option<FormatTraitNames>,
}

impl FormatImpl {
    pub fn new(format_args: FormatArgsStorage) -> Self {
        Self {
            format_args,
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
        // Assume no nested Impl of Debug and Display within each other
        if is_format_trait_impl(cx, impl_item).is_some() {
            self.format_trait_impl = None;
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(format_trait_impl) = self.format_trait_impl {
            let linter = FormatImplExpr {
                cx,
                format_args: &self.format_args,
                expr,
                format_trait_impl,
            };
            linter.check_to_string_in_display();
            linter.check_self_in_format_args();
            linter.check_print_in_format_impl();
        }
    }
}

struct FormatImplExpr<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    format_args: &'a FormatArgsStorage,
    expr: &'tcx Expr<'tcx>,
    format_trait_impl: FormatTraitNames,
}

impl FormatImplExpr<'_, '_> {
    fn check_to_string_in_display(&self) {
        if self.format_trait_impl.name == sym::Display
            && let ExprKind::MethodCall(path, self_arg, [], _) = self.expr.kind
            // Get the hir_id of the object we are calling the method on
            // Is the method to_string() ?
            && path.ident.name == sym::to_string
            // Is the method a part of the ToString trait? (i.e. not to_string() implemented
            // separately)
            && let Some(expr_def_id) = self.cx.typeck_results().type_dependent_def_id(self.expr.hir_id)
            && is_diag_trait_item(self.cx, expr_def_id, sym::ToString)
            // Is the method is called on self
            && let ExprKind::Path(QPath::Resolved(_, path)) = self_arg.kind
            && let [segment] = path.segments
            && segment.ident.name == kw::SelfLower
        {
            span_lint(
                self.cx,
                RECURSIVE_FORMAT_IMPL,
                self.expr.span,
                "using `self.to_string` in `fmt::Display` implementation will cause infinite recursion",
            );
        }
    }

    fn check_self_in_format_args(&self) {
        // Check each arg in format calls - do we ever use Display on self (directly or via deref)?
        if let Some(outer_macro) = root_macro_call_first_node(self.cx, self.expr)
            && let macro_def_id = outer_macro.def_id
            && is_format_macro(self.cx, macro_def_id)
            && let Some(format_args) = self.format_args.get(self.cx, self.expr, outer_macro.expn)
        {
            for piece in &format_args.template {
                if let FormatArgsPiece::Placeholder(placeholder) = piece
                    && let trait_name = match placeholder.format_trait {
                        FormatTrait::Display => sym::Display,
                        FormatTrait::Debug => sym::Debug,
                        FormatTrait::LowerExp => sym::LowerExp,
                        FormatTrait::UpperExp => sym::UpperExp,
                        FormatTrait::Octal => sym::Octal,
                        FormatTrait::Pointer => sym::Pointer,
                        FormatTrait::Binary => sym::Binary,
                        FormatTrait::LowerHex => sym::LowerHex,
                        FormatTrait::UpperHex => sym::UpperHex,
                    }
                    && trait_name == self.format_trait_impl.name
                    && let Ok(index) = placeholder.argument.index
                    && let Some(arg) = format_args.arguments.all_args().get(index)
                    && let Some(arg_expr) = find_format_arg_expr(self.expr, arg)
                {
                    self.check_format_arg_self(arg_expr);
                }
            }
        }
    }

    fn check_format_arg_self(&self, arg: &Expr<'_>) {
        // Handle multiple dereferencing of references e.g. &&self
        // Handle dereference of &self -> self that is equivalent (i.e. via *self in fmt() impl)
        // Since the argument to fmt is itself a reference: &self
        let reference = peel_ref_operators(self.cx, arg);
        // Is the reference self?
        if path_to_local(reference).map(|x| self.cx.tcx.hir_name(x)) == Some(kw::SelfLower) {
            let FormatTraitNames { name, .. } = self.format_trait_impl;
            span_lint(
                self.cx,
                RECURSIVE_FORMAT_IMPL,
                self.expr.span,
                format!("using `self` as `{name}` in `impl {name}` will cause infinite recursion"),
            );
        }
    }

    fn check_print_in_format_impl(&self) {
        if let Some(macro_call) = root_macro_call_first_node(self.cx, self.expr)
            && let Some(name) = self.cx.tcx.get_diagnostic_name(macro_call.def_id)
        {
            let replacement = match name {
                sym::print_macro | sym::eprint_macro => "write",
                sym::println_macro | sym::eprintln_macro => "writeln",
                _ => return,
            };

            let name = name.as_str().strip_suffix("_macro").unwrap();

            span_lint_and_sugg(
                self.cx,
                PRINT_IN_FORMAT_IMPL,
                macro_call.span,
                format!("use of `{name}!` in `{}` impl", self.format_trait_impl.name),
                "replace with",
                if let Some(formatter_name) = self.format_trait_impl.formatter_name {
                    format!("{replacement}!({formatter_name}, ..)")
                } else {
                    format!("{replacement}!(..)")
                },
                Applicability::HasPlaceholders,
            );
        }
    }
}

fn is_format_trait_impl(cx: &LateContext<'_>, impl_item: &ImplItem<'_>) -> Option<FormatTraitNames> {
    if impl_item.ident.name == sym::fmt
        && let ImplItemKind::Fn(_, body_id) = impl_item.kind
        && let Some(Impl {
            of_trait: Some(of_trait),
            ..
        }) = get_parent_as_impl(cx.tcx, impl_item.hir_id())
        && let Some(did) = of_trait.trait_ref.trait_def_id()
        && let Some(name) = cx.tcx.get_diagnostic_name(did)
        && matches!(name, sym::Debug | sym::Display)
    {
        let body = cx.tcx.hir_body(body_id);
        let formatter_name = body
            .params
            .get(1)
            .and_then(|param| param.pat.simple_ident())
            .map(|ident| ident.name);

        Some(FormatTraitNames { name, formatter_name })
    } else {
        None
    }
}
