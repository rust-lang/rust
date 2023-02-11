use clippy_utils::diagnostics::span_lint_and_sugg;
use if_chain::if_chain;
use rustc_ast::ast::{BindingAnnotation, ByRef, Lifetime, Mutability, Param, PatKind, Path, TyKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::kw;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// The lint checks for `self` in fn parameters that
    /// specify the `Self`-type explicitly
    /// ### Why is this bad?
    /// Increases the amount and decreases the readability of code
    ///
    /// ### Example
    /// ```rust
    /// enum ValType {
    ///     I32,
    ///     I64,
    ///     F32,
    ///     F64,
    /// }
    ///
    /// impl ValType {
    ///     pub fn bytes(self: Self) -> usize {
    ///         match self {
    ///             Self::I32 | Self::F32 => 4,
    ///             Self::I64 | Self::F64 => 8,
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// Could be rewritten as
    ///
    /// ```rust
    /// enum ValType {
    ///     I32,
    ///     I64,
    ///     F32,
    ///     F64,
    /// }
    ///
    /// impl ValType {
    ///     pub fn bytes(self) -> usize {
    ///         match self {
    ///             Self::I32 | Self::F32 => 4,
    ///             Self::I64 | Self::F64 => 8,
    ///         }
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.47.0"]
    pub NEEDLESS_ARBITRARY_SELF_TYPE,
    complexity,
    "type of `self` parameter is already by default `Self`"
}

declare_lint_pass!(NeedlessArbitrarySelfType => [NEEDLESS_ARBITRARY_SELF_TYPE]);

enum Mode {
    Ref(Option<Lifetime>),
    Value,
}

fn check_param_inner(cx: &EarlyContext<'_>, path: &Path, span: Span, binding_mode: &Mode, mutbl: Mutability) {
    if_chain! {
        if let [segment] = &path.segments[..];
        if segment.ident.name == kw::SelfUpper;
        then {
            // In case we have a named lifetime, we check if the name comes from expansion.
            // If it does, at this point we know the rest of the parameter was written by the user,
            // so let them decide what the name of the lifetime should be.
            // See #6089 for more details.
            let mut applicability = Applicability::MachineApplicable;
            let self_param = match (binding_mode, mutbl) {
                (Mode::Ref(None), Mutability::Mut) => "&mut self".to_string(),
                (Mode::Ref(Some(lifetime)), Mutability::Mut) => {
                    if lifetime.ident.span.from_expansion() {
                        applicability = Applicability::HasPlaceholders;
                        "&'_ mut self".to_string()
                    } else {
                        format!("&{} mut self", &lifetime.ident.name)
                    }
                },
                (Mode::Ref(None), Mutability::Not) => "&self".to_string(),
                (Mode::Ref(Some(lifetime)), Mutability::Not) => {
                    if lifetime.ident.span.from_expansion() {
                        applicability = Applicability::HasPlaceholders;
                        "&'_ self".to_string()
                    } else {
                        format!("&{} self", &lifetime.ident.name)
                    }
                },
                (Mode::Value, Mutability::Mut) => "mut self".to_string(),
                (Mode::Value, Mutability::Not) => "self".to_string(),
            };

            span_lint_and_sugg(
                cx,
                NEEDLESS_ARBITRARY_SELF_TYPE,
                span,
                "the type of the `self` parameter does not need to be arbitrary",
                "consider to change this parameter to",
                self_param,
                applicability,
            )
        }
    }
}

impl EarlyLintPass for NeedlessArbitrarySelfType {
    fn check_param(&mut self, cx: &EarlyContext<'_>, p: &Param) {
        // Bail out if the parameter it's not a receiver or was not written by the user
        if !p.is_self() || p.span.from_expansion() {
            return;
        }

        match &p.ty.kind {
            TyKind::Path(None, path) => {
                if let PatKind::Ident(BindingAnnotation(ByRef::No, mutbl), _, _) = p.pat.kind {
                    check_param_inner(cx, path, p.span.to(p.ty.span), &Mode::Value, mutbl);
                }
            },
            TyKind::Ref(lifetime, mut_ty) => {
                if_chain! {
                if let TyKind::Path(None, path) = &mut_ty.ty.kind;
                if let PatKind::Ident(BindingAnnotation::NONE, _, _) = p.pat.kind;
                    then {
                        check_param_inner(cx, path, p.span.to(p.ty.span), &Mode::Ref(*lifetime), mut_ty.mutbl);
                    }
                }
            },
            _ => {},
        }
    }
}
