use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::span_is_local;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::path_def_id;
use clippy_utils::source::snippet_opt;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_path, Visitor};
use rustc_hir::{
    GenericArg, GenericArgs, HirId, Impl, ImplItemKind, ImplItemRef, Item, ItemKind, PatKind, Path, PathSegment, Ty,
    TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::{hir::nested_filter::OnlyBodies, ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::{kw, sym};
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Searches for implementations of the `Into<..>` trait and suggests to implement `From<..>` instead.
    ///
    /// ### Why is this bad?
    /// According the std docs implementing `From<..>` is preferred since it gives you `Into<..>` for free where the reverse isn't true.
    ///
    /// ### Example
    /// ```rust
    /// struct StringWrapper(String);
    ///
    /// impl Into<StringWrapper> for String {
    ///     fn into(self) -> StringWrapper {
    ///         StringWrapper(self)
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// struct StringWrapper(String);
    ///
    /// impl From<String> for StringWrapper {
    ///     fn from(s: String) -> StringWrapper {
    ///         StringWrapper(s)
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub FROM_OVER_INTO,
    style,
    "Warns on implementations of `Into<..>` to use `From<..>`"
}

pub struct FromOverInto {
    msrv: Msrv,
}

impl FromOverInto {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        FromOverInto { msrv }
    }
}

impl_lint_pass!(FromOverInto => [FROM_OVER_INTO]);

impl<'tcx> LateLintPass<'tcx> for FromOverInto {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if !self.msrv.meets(msrvs::RE_REBALANCING_COHERENCE) || !span_is_local(item.span) {
            return;
        }

        if let ItemKind::Impl(Impl {
            of_trait: Some(hir_trait_ref),
            self_ty,
            items: [impl_item_ref],
            ..
        }) = item.kind
            && let Some(into_trait_seg) = hir_trait_ref.path.segments.last()
            // `impl Into<target_ty> for self_ty`
            && let Some(GenericArgs { args: [GenericArg::Type(target_ty)], .. }) = into_trait_seg.args
            && let Some(middle_trait_ref) = cx.tcx.impl_trait_ref(item.owner_id).map(ty::EarlyBinder::instantiate_identity)
            && cx.tcx.is_diagnostic_item(sym::Into, middle_trait_ref.def_id)
            && !matches!(middle_trait_ref.args.type_at(1).kind(), ty::Alias(ty::Opaque, _))
        {
            span_lint_and_then(
                cx,
                FROM_OVER_INTO,
                cx.tcx.sess.source_map().guess_head_span(item.span),
                "an implementation of `From` is preferred since it gives you `Into<_>` for free where the reverse isn't true",
                |diag| {
                    // If the target type is likely foreign mention the orphan rules as it's a common source of confusion
                    if path_def_id(cx, target_ty.peel_refs()).map_or(true, |id| !id.is_local()) {
                        diag.help(
                            "`impl From<Local> for Foreign` is allowed by the orphan rules, for more information see\n\
                            https://doc.rust-lang.org/reference/items/implementations.html#trait-implementation-coherence"
                        );
                    }

                    let message = format!("replace the `Into` implementation with `From<{}>`", middle_trait_ref.self_ty());
                    if let Some(suggestions) = convert_to_from(cx, into_trait_seg, target_ty, self_ty, impl_item_ref) {
                        diag.multipart_suggestion(message, suggestions, Applicability::MachineApplicable);
                    } else {
                        diag.help(message);
                    }
                },
            );
        }
    }

    extract_msrv_attr!(LateContext);
}

/// Finds the occurences of `Self` and `self`
struct SelfFinder<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    /// Occurences of `Self`
    upper: Vec<Span>,
    /// Occurences of `self`
    lower: Vec<Span>,
    /// If any of the `self`/`Self` usages were from an expansion, or the body contained a binding
    /// already named `val`
    invalid: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for SelfFinder<'a, 'tcx> {
    type NestedFilter = OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }

    fn visit_path(&mut self, path: &Path<'tcx>, _id: HirId) {
        for segment in path.segments {
            match segment.ident.name {
                kw::SelfLower => self.lower.push(segment.ident.span),
                kw::SelfUpper => self.upper.push(segment.ident.span),
                _ => continue,
            }

            self.invalid |= segment.ident.span.from_expansion();
        }

        if !self.invalid {
            walk_path(self, path);
        }
    }

    fn visit_name(&mut self, name: Symbol) {
        if name == sym::val {
            self.invalid = true;
        }
    }
}

fn convert_to_from(
    cx: &LateContext<'_>,
    into_trait_seg: &PathSegment<'_>,
    target_ty: &Ty<'_>,
    self_ty: &Ty<'_>,
    impl_item_ref: &ImplItemRef,
) -> Option<Vec<(Span, String)>> {
    if !target_ty.find_self_aliases().is_empty() {
        // It's tricky to expand self-aliases correctly, we'll ignore it to not cause a
        // bad suggestion/fix.
        return None;
    }
    let impl_item = cx.tcx.hir().impl_item(impl_item_ref.id);
    let ImplItemKind::Fn(ref sig, body_id) = impl_item.kind else { return None };
    let body = cx.tcx.hir().body(body_id);
    let [input] = body.params else { return None };
    let PatKind::Binding(.., self_ident, None) = input.pat.kind else { return None };

    let from = snippet_opt(cx, self_ty.span)?;
    let into = snippet_opt(cx, target_ty.span)?;

    let mut suggestions = vec![
        // impl Into<T> for U  ->  impl From<T> for U
        //      ~~~~                    ~~~~
        (into_trait_seg.ident.span, String::from("From")),
        // impl Into<T> for U  ->  impl Into<U> for U
        //           ~                       ~
        (target_ty.span, from.clone()),
        // impl Into<T> for U  ->  impl Into<T> for T
        //                  ~                       ~
        (self_ty.span, into),
        // fn into(self) -> T  ->  fn from(self) -> T
        //    ~~~~                    ~~~~
        (impl_item.ident.span, String::from("from")),
        // fn into([mut] self) -> T  ->  fn into([mut] v: T) -> T
        //               ~~~~                          ~~~~
        (self_ident.span, format!("val: {from}")),
        // fn into(self) -> T  ->  fn into(self) -> Self
        //                  ~                       ~~~~
        (sig.decl.output.span(), String::from("Self")),
    ];

    let mut finder = SelfFinder {
        cx,
        upper: Vec::new(),
        lower: Vec::new(),
        invalid: false,
    };
    finder.visit_expr(body.value);

    if finder.invalid {
        return None;
    }

    // don't try to replace e.g. `Self::default()` with `&[T]::default()`
    if !finder.upper.is_empty() && !matches!(self_ty.kind, TyKind::Path(_)) {
        return None;
    }

    for span in finder.upper {
        suggestions.push((span, from.clone()));
    }
    for span in finder.lower {
        suggestions.push((span, String::from("val")));
    }

    Some(suggestions)
}
