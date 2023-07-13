use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_from_proc_macro;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{walk_item, Visitor};
use rustc_hir::{GenericParamKind, HirId, Item, ItemKind, ItemLocalId, Node, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for idents which comprise of a single letter.
    ///
    /// Note: This lint can be very noisy when enabled; it may be desirable to only enable it
    /// temporarily.
    ///
    /// ### Why is this bad?
    /// In many cases it's not, but at times it can severely hinder readability. Some codebases may
    /// wish to disallow this to improve readability.
    ///
    /// ### Example
    /// ```rust,ignore
    /// for m in movies {
    ///     let title = m.t;
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// for movie in movies {
    ///     let title = movie.title;
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub MIN_IDENT_CHARS,
    restriction,
    "disallows idents that are too short"
}
impl_lint_pass!(MinIdentChars => [MIN_IDENT_CHARS]);

#[derive(Clone)]
pub struct MinIdentChars {
    pub allowed_idents_below_min_chars: FxHashSet<String>,
    pub min_ident_chars_threshold: u64,
}

impl MinIdentChars {
    #[expect(clippy::cast_possible_truncation)]
    fn is_ident_too_short(&self, cx: &LateContext<'_>, str: &str, span: Span) -> bool {
        !in_external_macro(cx.sess(), span)
            && str.len() <= self.min_ident_chars_threshold as usize
            && !str.starts_with('_')
            && !str.is_empty()
            && self.allowed_idents_below_min_chars.get(&str.to_owned()).is_none()
    }
}

impl LateLintPass<'_> for MinIdentChars {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if self.min_ident_chars_threshold == 0 {
            return;
        }

        walk_item(&mut IdentVisitor { conf: self, cx }, item);
    }

    // This is necessary as `Node::Pat`s are not visited in `visit_id`. :/
    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &Pat<'_>) {
        if let PatKind::Binding(_, _, ident, ..) = pat.kind
            && let str = ident.as_str()
            && self.is_ident_too_short(cx, str, ident.span)
        {
            emit_min_ident_chars(self, cx, str, ident.span);
        }
    }
}

struct IdentVisitor<'cx, 'tcx> {
    conf: &'cx MinIdentChars,
    cx: &'cx LateContext<'tcx>,
}

impl Visitor<'_> for IdentVisitor<'_, '_> {
    fn visit_id(&mut self, hir_id: HirId) {
        let Self { conf, cx } = *self;
        // FIXME(#112534) Reimplementation of `find`, as it uses indexing, which can (and will in
        // async functions, or really anything async) panic. This should probably be fixed on the
        // rustc side, this is just a temporary workaround.
        let node = if hir_id.local_id == ItemLocalId::from_u32(0) {
            // In this case, we can just use `find`, `Owner`'s `node` field is private anyway so we can't
            // reimplement it even if we wanted to
            cx.tcx.hir().find(hir_id)
        } else {
            let Some(owner) = cx.tcx.hir_owner_nodes(hir_id.owner).as_owner() else {
                return;
            };
            owner.nodes.get(hir_id.local_id).copied().flatten().map(|p| p.node)
        };
        let Some(node) = node else {
            return;
        };
        let Some(ident) = node.ident() else {
            return;
        };

        let str = ident.as_str();
        if conf.is_ident_too_short(cx, str, ident.span) {
            if let Node::Item(item) = node && let ItemKind::Use(..) = item.kind {
                return;
            }
            // `struct Awa<T>(T)`
            //                ^
            if let Node::PathSegment(path) = node {
                if let Res::Def(def_kind, ..) = path.res && let DefKind::TyParam = def_kind {
                    return;
                }
                if matches!(path.res, Res::PrimTy(..)) || path.res.opt_def_id().is_some_and(|def_id| !def_id.is_local())
                {
                    return;
                }
            }
            // `struct Awa<T>(T)`
            //             ^
            if let Node::GenericParam(generic_param) = node
                && let GenericParamKind::Type { .. } = generic_param.kind
            {
                return;
            }

            if is_from_proc_macro(cx, &ident) {
                return;
            }

            emit_min_ident_chars(conf, cx, str, ident.span);
        }
    }
}

fn emit_min_ident_chars(conf: &MinIdentChars, cx: &impl LintContext, ident: &str, span: Span) {
    let help = if conf.min_ident_chars_threshold == 1 {
        Cow::Borrowed("this ident consists of a single char")
    } else {
        Cow::Owned(format!(
            "this ident is too short ({} <= {})",
            ident.len(),
            conf.min_ident_chars_threshold,
        ))
    };
    span_lint(cx, MIN_IDENT_CHARS, span, &help);
}
