use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, Stmt};
use rustc_middle::ty::{Mutability, TyKind};
use rustc_session::lint::fcw;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{BytePos, Span};

use crate::lints::{MutRefSugg, RefOfMutStatic, StaticMutRefsInteriorMutabilitySugg};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `static_mut_refs` lint checks for shared or mutable references
    /// of mutable static inside `unsafe` blocks and `unsafe` functions.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021
    /// fn main() {
    ///     static mut X: i32 = 23;
    ///     static mut Y: i32 = 24;
    ///
    ///     unsafe {
    ///         let y = &X;
    ///         let ref x = X;
    ///         let (x, y) = (&X, &Y);
    ///         foo(&X);
    ///     }
    /// }
    ///
    /// unsafe fn _foo() {
    ///     static mut X: i32 = 23;
    ///     static mut Y: i32 = 24;
    ///
    ///     let y = &X;
    ///     let ref x = X;
    ///     let (x, y) = (&X, &Y);
    ///     foo(&X);
    /// }
    ///
    /// fn foo<'a>(_x: &'a i32) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Shared or mutable references of mutable static are almost always a mistake and
    /// can lead to undefined behavior and various other problems in your code.
    ///
    /// This lint is "warn" by default on editions up to 2021, in 2024 is "deny".
    pub STATIC_MUT_REFS,
    Warn,
    "creating a shared reference to mutable static",
    @future_incompatible = FutureIncompatibleInfo {
        reason: fcw!(EditionError 2024 "static-mut-references"),
        explain_reason: false,
    };
    @edition Edition2024 => Deny;
}

declare_lint_pass!(StaticMutRefs => [STATIC_MUT_REFS]);

impl<'tcx> LateLintPass<'tcx> for StaticMutRefs {
    #[allow(rustc::usage_of_ty_tykind)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        let err_span = expr.span;
        match expr.kind {
            hir::ExprKind::AddrOf(borrow_kind, m, ex)
                if matches!(borrow_kind, hir::BorrowKind::Ref)
                    && let Some(static_mut) = path_is_static_mut(ex, err_span) =>
            {
                let source_map = cx.sess().source_map();
                let snippet = source_map.span_to_snippet(err_span);

                let sugg_span = if let Ok(snippet) = snippet {
                    // ( ( &IDENT ) )
                    // ~~~~ exclude these from the suggestion span to avoid unmatching parens
                    let exclude_n_bytes: u32 = snippet
                        .chars()
                        .take_while(|ch| ch.is_whitespace() || *ch == '(')
                        .map(|ch| ch.len_utf8() as u32)
                        .sum();

                    err_span.with_lo(err_span.lo() + BytePos(exclude_n_bytes)).with_hi(ex.span.lo())
                } else {
                    err_span.with_hi(ex.span.lo())
                };

                emit_static_mut_refs(
                    cx,
                    static_mut.err_span,
                    sugg_span,
                    m,
                    !expr.span.from_expansion(),
                    static_mut.def_id,
                );
            }
            hir::ExprKind::MethodCall(_, e, _, _)
                if let Some(static_mut) = path_is_static_mut(e, expr.span)
                    && let typeck = cx.typeck_results()
                    && let Some(method_def_id) = typeck.type_dependent_def_id(expr.hir_id)
                    && let inputs =
                        cx.tcx.fn_sig(method_def_id).skip_binder().inputs().skip_binder()
                    && let Some(receiver) = inputs.get(0)
                    && let TyKind::Ref(_, _, m) = receiver.kind() =>
            {
                emit_static_mut_refs(
                    cx,
                    static_mut.err_span,
                    static_mut.err_span.shrink_to_lo(),
                    *m,
                    false,
                    static_mut.def_id,
                );
            }
            _ => {}
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &Stmt<'_>) {
        if let hir::StmtKind::Let(loc) = stmt.kind
            && let hir::PatKind::Binding(ba, _, _, _) = loc.pat.kind
            && let hir::ByRef::Yes(_, m) = ba.0
            && let Some(init) = loc.init
            && let Some(static_mut) = path_is_static_mut(init, init.span)
        {
            emit_static_mut_refs(
                cx,
                static_mut.err_span,
                static_mut.err_span.shrink_to_lo(),
                m,
                false,
                static_mut.def_id,
            );
        }
    }
}

struct StaticMutInfo {
    err_span: Span,
    def_id: DefId,
}

fn path_is_static_mut(mut expr: &hir::Expr<'_>, mut err_span: Span) -> Option<StaticMutInfo> {
    if err_span.from_expansion() {
        err_span = expr.span;
    }

    while let hir::ExprKind::Field(e, _) = expr.kind {
        expr = e;
    }

    if let hir::ExprKind::Path(qpath) = expr.kind
        && let hir::QPath::Resolved(_, path) = qpath
        && let hir::def::Res::Def(def_kind, def_id) = path.res
        && let hir::def::DefKind::Static { safety: _, mutability: Mutability::Mut, nested: false } =
            def_kind
    {
        return Some(StaticMutInfo { err_span, def_id });
    }
    None
}

fn emit_static_mut_refs(
    cx: &LateContext<'_>,
    span: Span,
    sugg_span: Span,
    mutable: Mutability,
    suggest_addr_of: bool,
    def_id: DefId,
) {
    let (shared_label, shared_note, mut_note, sugg) = match mutable {
        Mutability::Mut => {
            let sugg =
                if suggest_addr_of { Some(MutRefSugg::Mut { span: sugg_span }) } else { None };
            ("mutable ", false, true, sugg)
        }
        Mutability::Not => {
            let sugg =
                if suggest_addr_of { Some(MutRefSugg::Shared { span: sugg_span }) } else { None };
            ("shared ", true, false, sugg)
        }
    };

    let (interior_mutability_help, interior_mutability_sugg) =
        interior_mutability_suggestion(cx, def_id);

    cx.emit_span_lint(
        STATIC_MUT_REFS,
        span,
        RefOfMutStatic {
            span,
            sugg,
            shared_label,
            shared_note,
            mut_note,
            interior_mutability_help,
            interior_mutability_sugg,
        },
    );
}

fn interior_mutability_suggestion(
    cx: &LateContext<'_>,
    def_id: DefId,
) -> (bool, Option<StaticMutRefsInteriorMutabilitySugg>) {
    let static_ty = cx.tcx.type_of(def_id).skip_binder();
    let has_interior_mutability = !static_ty.is_freeze(cx.tcx, cx.typing_env());

    if !has_interior_mutability {
        return (false, None);
    }

    let sugg =
        static_mutability_span(cx, def_id).map(|span| StaticMutRefsInteriorMutabilitySugg { span });
    (true, sugg)
}

fn static_mutability_span(cx: &LateContext<'_>, def_id: DefId) -> Option<Span> {
    let hir_id = cx.tcx.hir_get_if_local(def_id)?;
    let hir::Node::Item(item) = hir_id else { return None };
    let (mutability, ident) = match item.kind {
        hir::ItemKind::Static(mutability, ident, _, _) => (mutability, ident),
        _ => return None,
    };
    if mutability != hir::Mutability::Mut {
        return None;
    }

    let vis_span = item.vis_span.find_ancestor_inside(item.span)?;
    if !item.span.can_be_used_for_suggestions() || !vis_span.can_be_used_for_suggestions() {
        return None;
    }

    let header_span = vis_span.between(ident.span);
    if !header_span.can_be_used_for_suggestions() {
        return None;
    }

    let source_map = cx.sess().source_map();
    let snippet = source_map.span_to_snippet(header_span).ok()?;

    let (_static_start, static_end) = find_word(&snippet, "static", 0)?;
    let (mut_start, mut_end) = find_word(&snippet, "mut", static_end)?;
    let mut_end = extend_trailing_space(&snippet, mut_end);

    Some(
        header_span
            .with_lo(header_span.lo() + BytePos(mut_start as u32))
            .with_hi(header_span.lo() + BytePos(mut_end as u32)),
    )
}

fn find_word(snippet: &str, word: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = snippet.as_bytes();
    let word_bytes = word.as_bytes();
    let mut search = start;
    while search <= snippet.len() {
        let found = snippet[search..].find(word)?;
        let idx = search + found;
        let end = idx + word_bytes.len();
        let before_ok = idx == 0 || !is_ident_char(bytes[idx - 1]);
        let after_ok = end >= bytes.len() || !is_ident_char(bytes[end]);
        if before_ok && after_ok {
            return Some((idx, end));
        }
        search = end;
    }
    None
}

fn is_ident_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn extend_trailing_space(snippet: &str, mut end: usize) -> usize {
    if let Some(ch) = snippet[end..].chars().next()
        && (ch == ' ' || ch == '\t')
    {
        end += ch.len_utf8();
    }
    end
}
