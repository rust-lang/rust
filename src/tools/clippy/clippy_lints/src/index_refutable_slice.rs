use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::IfLet;
use clippy_utils::ty::is_copy;
use clippy_utils::{is_expn_of, is_lint_allowed, meets_msrv, msrvs, path_to_local};
use if_chain::if_chain;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{symbol::Ident, Span};

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
    /// ```rust
    /// let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    ///
    /// if let Some(slice) = slice {
    ///     println!("{}", slice[0]);
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    ///
    /// if let Some(&[first, ..]) = slice {
    ///     println!("{}", first);
    /// }
    /// ```
    #[clippy::version = "1.59.0"]
    pub INDEX_REFUTABLE_SLICE,
    nursery,
    "avoid indexing on slices which could be destructed"
}

#[derive(Copy, Clone)]
pub struct IndexRefutableSlice {
    max_suggested_slice: u64,
    msrv: Option<RustcVersion>,
}

impl IndexRefutableSlice {
    pub fn new(max_suggested_slice_pattern_length: u64, msrv: Option<RustcVersion>) -> Self {
        Self {
            max_suggested_slice: max_suggested_slice_pattern_length,
            msrv,
        }
    }
}

impl_lint_pass!(IndexRefutableSlice => [INDEX_REFUTABLE_SLICE]);

impl<'tcx> LateLintPass<'tcx> for IndexRefutableSlice {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if_chain! {
            if !expr.span.from_expansion() || is_expn_of(expr.span, "if_chain").is_some();
            if let Some(IfLet {let_pat, if_then, ..}) = IfLet::hir(cx, expr);
            if !is_lint_allowed(cx, INDEX_REFUTABLE_SLICE, expr.hir_id);
            if meets_msrv(self.msrv, msrvs::SLICE_PATTERNS);

            let found_slices = find_slice_values(cx, let_pat);
            if !found_slices.is_empty();
            let filtered_slices = filter_lintable_slices(cx, found_slices, self.max_suggested_slice, if_then);
            if !filtered_slices.is_empty();
            then {
                for slice in filtered_slices.values() {
                    lint_slice(cx, slice);
                }
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

fn find_slice_values(cx: &LateContext<'_>, pat: &hir::Pat<'_>) -> FxIndexMap<hir::HirId, SliceLintInformation> {
    let mut removed_pat: FxHashSet<hir::HirId> = FxHashSet::default();
    let mut slices: FxIndexMap<hir::HirId, SliceLintInformation> = FxIndexMap::default();
    pat.walk_always(|pat| {
        // We'll just ignore mut and ref mut for simplicity sake right now
        if let hir::PatKind::Binding(
            hir::BindingAnnotation(by_ref, hir::Mutability::Not),
            value_hir_id,
            ident,
            sub_pat,
        ) = pat.kind
        {
            // This block catches bindings with sub patterns. It would be hard to build a correct suggestion
            // for them and it's likely that the user knows what they are doing in such a case.
            if removed_pat.contains(&value_hir_id) {
                return;
            }
            if sub_pat.is_some() {
                removed_pat.insert(value_hir_id);
                slices.remove(&value_hir_id);
                return;
            }

            let bound_ty = cx.typeck_results().node_type(pat.hir_id);
            if let ty::Slice(inner_ty) | ty::Array(inner_ty, _) = bound_ty.peel_refs().kind() {
                // The values need to use the `ref` keyword if they can't be copied.
                // This will need to be adjusted if the lint want to support mutable access in the future
                let src_is_ref = bound_ty.is_ref() && by_ref != hir::ByRef::Yes;
                let needs_ref = !(src_is_ref || is_copy(cx, *inner_ty));

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
        .collect::<FxHashSet<_>>();

    let value_name = |index| format!("{}_{}", slice.ident.name, index);

    if let Some(max_index) = used_indices.iter().max() {
        let opt_ref = if slice.needs_ref { "ref " } else { "" };
        let pat_sugg_idents = (0..=*max_index)
            .map(|index| {
                if used_indices.contains(&index) {
                    format!("{}{}", opt_ref, value_name(index))
                } else {
                    "_".to_string()
                }
            })
            .collect::<Vec<_>>();
        let pat_sugg = format!("[{}, ..]", pat_sugg_idents.join(", "));

        span_lint_and_then(
            cx,
            INDEX_REFUTABLE_SLICE,
            slice.ident.span,
            "this binding can be a slice pattern to avoid indexing",
            |diag| {
                diag.multipart_suggestion(
                    "try using a slice pattern here",
                    slice
                        .pattern_spans
                        .iter()
                        .map(|span| (*span, pat_sugg.clone()))
                        .collect(),
                    Applicability::MaybeIncorrect,
                );

                diag.multipart_suggestion(
                    "and replace the index expressions here",
                    slice
                        .index_use
                        .iter()
                        .map(|(index, span)| (*span, value_name(*index)))
                        .collect(),
                    Applicability::MaybeIncorrect,
                );

                // The lint message doesn't contain a warning about the removed index expression,
                // since `filter_lintable_slices` will only return slices where all access indices
                // are known at compile time. Therefore, they can be removed without side effects.
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

fn filter_lintable_slices<'a, 'tcx>(
    cx: &'a LateContext<'tcx>,
    slice_lint_info: FxIndexMap<hir::HirId, SliceLintInformation>,
    max_suggested_slice: u64,
    scope: &'tcx hir::Expr<'tcx>,
) -> FxIndexMap<hir::HirId, SliceLintInformation> {
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
    slice_lint_info: FxIndexMap<hir::HirId, SliceLintInformation>,
    max_suggested_slice: u64,
}

impl<'a, 'tcx> Visitor<'tcx> for SliceIndexLintingVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let Some(local_id) = path_to_local(expr) {
            let Self {
                cx,
                ref mut slice_lint_info,
                max_suggested_slice,
            } = *self;

            if_chain! {
                // Check if this is even a local we're interested in
                if let Some(use_info) = slice_lint_info.get_mut(&local_id);

                let map = cx.tcx.hir();

                // Checking for slice indexing
                let parent_id = map.get_parent_node(expr.hir_id);
                if let Some(hir::Node::Expr(parent_expr)) = map.find(parent_id);
                if let hir::ExprKind::Index(_, index_expr) = parent_expr.kind;
                if let Some((Constant::Int(index_value), _)) = constant(cx, cx.typeck_results(), index_expr);
                if let Ok(index_value) = index_value.try_into();
                if index_value < max_suggested_slice;

                // Make sure that this slice index is read only
                let maybe_addrof_id = map.get_parent_node(parent_id);
                if let Some(hir::Node::Expr(maybe_addrof_expr)) = map.find(maybe_addrof_id);
                if let hir::ExprKind::AddrOf(_kind, hir::Mutability::Not, _inner_expr) = maybe_addrof_expr.kind;
                then {
                    use_info.index_use.push((index_value, map.span(parent_expr.hir_id)));
                    return;
                }
            }

            // The slice was used for something other than indexing
            self.slice_lint_info.remove(&local_id);
        }
        intravisit::walk_expr(self, expr);
    }
}
