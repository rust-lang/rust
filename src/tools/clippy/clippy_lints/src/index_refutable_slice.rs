use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::IfLet;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::ty::is_copy;
use clippy_utils::{is_expn_of, is_lint_allowed, path_to_local, sym};
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::HirId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::symbol::Ident;

declare_clippy_lint! {
    /// ### What it does
    /// The lint checks for slice bindings in patterns that are only used to
    /// access individual slice values.
    ///
    /// ### Why is this bad?
    /// Accessing slice values using indices can lead to panics. Using refutable
    /// patterns can avoid these. Binding to individual values also improves the
    /// readability as they can be named.
    ///
    /// ### Limitations
    /// This lint currently only checks for immutable access inside `if let`
    /// patterns.
    ///
    /// ### Example
    /// ```no_run
    /// let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    ///
    /// if let Some(slice) = slice {
    ///     println!("{}", slice[0]);
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    ///
    /// if let Some(&[first, ..]) = slice {
    ///     println!("{}", first);
    /// }
    /// ```
    #[clippy::version = "1.59.0"]
    pub INDEX_REFUTABLE_SLICE,
    pedantic,
    "avoid indexing on slices which could be destructed"
}

pub struct IndexRefutableSlice {
    max_suggested_slice: u64,
    msrv: Msrv,
}

impl IndexRefutableSlice {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            max_suggested_slice: conf.max_suggested_slice_pattern_length,
            msrv: conf.msrv,
        }
    }
}

impl_lint_pass!(IndexRefutableSlice => [INDEX_REFUTABLE_SLICE]);

impl<'tcx> LateLintPass<'tcx> for IndexRefutableSlice {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if let Some(IfLet { let_pat, if_then, .. }) = IfLet::hir(cx, expr)
            && (!expr.span.from_expansion() || is_expn_of(expr.span, sym::if_chain).is_some())
            && !is_lint_allowed(cx, INDEX_REFUTABLE_SLICE, expr.hir_id)
            && let found_slices = find_slice_values(cx, let_pat)
            && !found_slices.is_empty()
            && let filtered_slices = filter_lintable_slices(cx, found_slices, self.max_suggested_slice, if_then)
            && !filtered_slices.is_empty()
            && self.msrv.meets(cx, msrvs::SLICE_PATTERNS)
        {
            for slice in filtered_slices.values() {
                lint_slice(cx, slice);
            }
        }
    }
}

fn find_slice_values(cx: &LateContext<'_>, pat: &hir::Pat<'_>) -> FxIndexMap<HirId, SliceLintInformation> {
    let mut removed_pat: FxHashSet<HirId> = FxHashSet::default();
    let mut slices: FxIndexMap<HirId, SliceLintInformation> = FxIndexMap::default();
    pat.walk_always(|pat| {
        // We'll just ignore mut and ref mut for simplicity sake right now
        if let hir::PatKind::Binding(hir::BindingMode(by_ref, hir::Mutability::Not), value_hir_id, ident, sub_pat) =
            pat.kind
            && by_ref != hir::ByRef::Yes(hir::Mutability::Mut)
        {
            // This block catches bindings with sub patterns. It would be hard to build a correct suggestion
            // for them and it's likely that the user knows what they are doing in such a case.
            if removed_pat.contains(&value_hir_id) {
                return;
            }
            if sub_pat.is_some() {
                removed_pat.insert(value_hir_id);
                // FIXME(rust/#120456) - is `swap_remove` correct?
                slices.swap_remove(&value_hir_id);
                return;
            }

            let bound_ty = cx.typeck_results().node_type(pat.hir_id);
            if let Some(inner_ty) = bound_ty.peel_refs().builtin_index() {
                // The values need to use the `ref` keyword if they can't be copied.
                // This will need to be adjusted if the lint want to support mutable access in the future
                let src_is_ref = bound_ty.is_ref() && by_ref == hir::ByRef::No;
                let needs_ref = !(src_is_ref || is_copy(cx, inner_ty));

                let slice_info = slices
                    .entry(value_hir_id)
                    .or_insert_with(|| SliceLintInformation::new(ident, needs_ref));
                slice_info.pattern_spans.push(pat.span);
            }
        }
    });

    slices
}

fn lint_slice(cx: &LateContext<'_>, slice: &SliceLintInformation) {
    let used_indices = slice
        .index_use
        .iter()
        .map(|(index, _)| *index)
        .collect::<FxIndexSet<_>>();

    let value_name = |index| format!("{}_{}", slice.ident.name, index);

    if let Some(max_index) = used_indices.iter().max() {
        let opt_ref = if slice.needs_ref { "ref " } else { "" };
        let pat_sugg_idents = (0..=*max_index)
            .map(|index| {
                if used_indices.contains(&index) {
                    format!("{opt_ref}{}", value_name(index))
                } else {
                    "_".to_string()
                }
            })
            .collect::<Vec<_>>();
        let pat_sugg = format!("[{}, ..]", pat_sugg_idents.join(", "));

        let mut suggestions = Vec::new();

        // Add the binding pattern suggestion
        if !slice.pattern_spans.is_empty() {
            suggestions.extend(slice.pattern_spans.iter().map(|span| (*span, pat_sugg.clone())));
        }

        // Add the index replacement suggestions
        if !slice.index_use.is_empty() {
            suggestions.extend(slice.index_use.iter().map(|(index, span)| (*span, value_name(*index))));
        }

        span_lint_and_then(
            cx,
            INDEX_REFUTABLE_SLICE,
            slice.ident.span,
            "this binding can be a slice pattern to avoid indexing",
            |diag| {
                diag.multipart_suggestion(
                    "replace the binding and indexed access with a slice pattern",
                    suggestions,
                    Applicability::MaybeIncorrect,
                );
            },
        );
    }
}

#[derive(Debug)]
struct SliceLintInformation {
    ident: Ident,
    needs_ref: bool,
    pattern_spans: Vec<Span>,
    index_use: Vec<(u64, Span)>,
}

impl SliceLintInformation {
    fn new(ident: Ident, needs_ref: bool) -> Self {
        Self {
            ident,
            needs_ref,
            pattern_spans: Vec::new(),
            index_use: Vec::new(),
        }
    }
}

fn filter_lintable_slices<'tcx>(
    cx: &LateContext<'tcx>,
    slice_lint_info: FxIndexMap<HirId, SliceLintInformation>,
    max_suggested_slice: u64,
    scope: &'tcx hir::Expr<'tcx>,
) -> FxIndexMap<HirId, SliceLintInformation> {
    let mut visitor = SliceIndexLintingVisitor {
        cx,
        slice_lint_info,
        max_suggested_slice,
    };

    intravisit::walk_expr(&mut visitor, scope);

    visitor.slice_lint_info
}

struct SliceIndexLintingVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    slice_lint_info: FxIndexMap<HirId, SliceLintInformation>,
    max_suggested_slice: u64,
}

impl<'tcx> Visitor<'tcx> for SliceIndexLintingVisitor<'_, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let Some(local_id) = path_to_local(expr) {
            let Self {
                cx,
                ref mut slice_lint_info,
                max_suggested_slice,
            } = *self;

            if let Some(use_info) = slice_lint_info.get_mut(&local_id)
                // Checking for slice indexing
                && let parent_id = cx.tcx.parent_hir_id(expr.hir_id)
                && let hir::Node::Expr(parent_expr) = cx.tcx.hir_node(parent_id)
                && let hir::ExprKind::Index(_, index_expr, _) = parent_expr.kind
                && let Some(Constant::Int(index_value)) = ConstEvalCtxt::new(cx).eval(index_expr)
                && let Ok(index_value) = index_value.try_into()
                && index_value < max_suggested_slice

                // Make sure that this slice index is read only
                && let hir::Node::Expr(maybe_addrof_expr) = cx.tcx.parent_hir_node(parent_id)
                && let hir::ExprKind::AddrOf(_kind, hir::Mutability::Not, _inner_expr) = maybe_addrof_expr.kind
            {
                use_info
                    .index_use
                    .push((index_value, cx.tcx.hir_span(parent_expr.hir_id)));
                return;
            }

            // The slice was used for something other than indexing
            // FIXME(rust/#120456) - is `swap_remove` correct?
            self.slice_lint_info.swap_remove(&local_id);
        }
        intravisit::walk_expr(self, expr);
    }
}
