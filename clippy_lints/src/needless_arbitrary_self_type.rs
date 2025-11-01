use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use rustc_ast::ast::{BindingMode, ByRef, Lifetime, Param, PatKind, TyKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::kw;

declare_clippy_lint! {
    /// ### What it does
    /// The lint checks for `self` in fn parameters that
    /// specify the `Self`-type explicitly
    /// ### Why is this bad?
    /// Increases the amount and decreases the readability of code
    ///
    /// ### Example
    /// ```no_run
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
    /// ```no_run
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

impl EarlyLintPass for NeedlessArbitrarySelfType {
    fn check_param(&mut self, cx: &EarlyContext<'_>, p: &Param) {
        // Bail out if the parameter it's not a receiver or was not written by the user
        if !p.is_self() || p.span.from_expansion() {
            return;
        }

        let (path, binding_mode, mutbl) = match &p.ty.kind {
            TyKind::Path(None, path) if let PatKind::Ident(BindingMode(ByRef::No, mutbl), _, _) = p.pat.kind => {
                (path, Mode::Value, mutbl)
            },
            TyKind::Ref(lifetime, mut_ty)
                if let TyKind::Path(None, path) = &mut_ty.ty.kind
                    && let PatKind::Ident(BindingMode::NONE, _, _) = p.pat.kind =>
            {
                (path, Mode::Ref(*lifetime), mut_ty.mutbl)
            },
            _ => return,
        };

        let span = p.span.to(p.ty.span);
        if let [segment] = &path.segments[..]
            && segment.ident.name == kw::SelfUpper
        {
            span_lint_and_then(
                cx,
                NEEDLESS_ARBITRARY_SELF_TYPE,
                span,
                "the type of the `self` parameter does not need to be arbitrary",
                |diag| {
                    let mut applicability = Applicability::MachineApplicable;
                    let add = match binding_mode {
                        Mode::Value => String::new(),
                        Mode::Ref(None) => mutbl.ref_prefix_str().to_string(),
                        Mode::Ref(Some(lifetime)) => {
                            // In case we have a named lifetime, we check if the name comes from expansion.
                            // If it does, at this point we know the rest of the parameter was written by the user,
                            // so let them decide what the name of the lifetime should be.
                            // See #6089 for more details.
                            let lt_name = if lifetime.ident.span.from_expansion() {
                                applicability = Applicability::HasPlaceholders;
                                "'_".into()
                            } else {
                                snippet_with_applicability(cx, lifetime.ident.span, "'_", &mut applicability)
                            };
                            format!("&{lt_name} {mut_}", mut_ = mutbl.prefix_str())
                        },
                    };

                    let mut sugg = vec![(p.ty.span.with_lo(p.span.hi()), String::new())];
                    if !add.is_empty() {
                        sugg.push((p.span.shrink_to_lo(), add));
                    }
                    diag.multipart_suggestion_verbose("remove the type", sugg, applicability);
                },
            );
        }
    }
}
