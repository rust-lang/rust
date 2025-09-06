use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_from_proc_macro;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{Visitor, walk_item, walk_trait_item};
use rustc_hir::{
    GenericParamKind, HirId, Impl, ImplItem, ImplItemImplKind, ImplItemKind, Item, ItemKind, ItemLocalId, Node, Pat,
    PatKind, TraitItem, UsePath,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::Ident;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for identifiers which consist of a single character (or fewer than the configured threshold).
    ///
    /// Note: This lint can be very noisy when enabled; it may be desirable to only enable it
    /// temporarily.
    ///
    /// ### Why restrict this?
    /// To improve readability by requiring that every variable has a name more specific than a single letter can be.
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
    ///
    /// ### Limitations
    /// Trait implementations which use the same function or parameter name as the trait declaration will
    /// not be warned about, even if the name is below the configured limit.
    #[clippy::version = "1.72.0"]
    pub MIN_IDENT_CHARS,
    restriction,
    "disallows idents that are too short"
}
impl_lint_pass!(MinIdentChars => [MIN_IDENT_CHARS]);

pub struct MinIdentChars {
    allowed_idents_below_min_chars: FxHashSet<String>,
    min_ident_chars_threshold: u64,
}

impl MinIdentChars {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            allowed_idents_below_min_chars: conf.allowed_idents_below_min_chars.iter().cloned().collect(),
            min_ident_chars_threshold: conf.min_ident_chars_threshold,
        }
    }

    #[expect(clippy::cast_possible_truncation)]
    fn is_ident_too_short(&self, cx: &LateContext<'_>, str: &str, span: Span) -> bool {
        !span.in_external_macro(cx.sess().source_map())
            && str.len() <= self.min_ident_chars_threshold as usize
            && !str.starts_with('_')
            && !str.is_empty()
            && !self.allowed_idents_below_min_chars.contains(str)
    }
}

impl LateLintPass<'_> for MinIdentChars {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if self.min_ident_chars_threshold == 0 {
            return;
        }

        walk_item(&mut IdentVisitor { conf: self, cx }, item);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, item: &TraitItem<'_>) {
        if self.min_ident_chars_threshold == 0 {
            return;
        }

        // If the function is declared but not defined in a trait, check_pat isn't called so we need to
        // check this explicitly
        if matches!(&item.kind, rustc_hir::TraitItemKind::Fn(_, _)) {
            let param_names = cx.tcx.fn_arg_idents(item.owner_id.to_def_id());
            for ident in param_names.iter().flatten() {
                let str = ident.as_str();
                if self.is_ident_too_short(cx, str, ident.span) {
                    emit_min_ident_chars(self, cx, str, ident.span);
                }
            }
        }

        walk_trait_item(&mut IdentVisitor { conf: self, cx }, item);
    }

    // This is necessary as `Node::Pat`s are not visited in `visit_id`. :/
    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &Pat<'_>) {
        if let PatKind::Binding(_, _, ident, ..) = pat.kind
            && let str = ident.as_str()
            && self.is_ident_too_short(cx, str, ident.span)
            && is_not_in_trait_impl(cx, pat, ident)
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
            Some(cx.tcx.hir_node(hir_id))
        } else {
            let owner = cx.tcx.hir_owner_nodes(hir_id.owner);
            owner.nodes.get(hir_id.local_id).copied().map(|p| p.node)
        };
        let Some(node) = node else {
            return;
        };
        let Some(ident) = node.ident() else {
            return;
        };

        let str = ident.as_str();
        if conf.is_ident_too_short(cx, str, ident.span) {
            // Check whether the node is part of a `impl` for a trait.
            if matches!(cx.tcx.parent_hir_node(hir_id), Node::TraitRef(_)) {
                return;
            }

            // Check whether the node is part of a `use` statement. We don't want to emit a warning if the user
            // has no control over the type.
            let usenode = opt_as_use_node(node).or_else(|| {
                cx.tcx
                    .hir_parent_iter(hir_id)
                    .find_map(|(_, node)| opt_as_use_node(node))
            });

            // If the name of the identifier is the same as the one of the imported item, this means that we
            // found a `use foo::bar`. We can early-return to not emit the warning.
            // If however the identifier is different, this means it is an alias (`use foo::bar as baz`). In
            // this case, we need to emit the warning for `baz`.
            if let Some(imported_item_path) = usenode
                // use `present_items` because it could be in any of type_ns, value_ns, macro_ns
                && let Some(Res::Def(_, imported_item_defid)) = imported_item_path.res.value_ns
                && cx.tcx.item_name(imported_item_defid).as_str() == str
            {
                return;
            }

            // `struct Array<T, const N: usize>([T; N])`
            //                                   ^  ^
            if let Node::PathSegment(path) = node {
                if let Res::Def(def_kind, ..) = path.res
                    && let DefKind::TyParam | DefKind::ConstParam = def_kind
                {
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

            // `struct Array<T, const N: usize>([T; N])`
            //                        ^
            if let Node::GenericParam(generic_param) = node
                && let GenericParamKind::Const { .. } = generic_param.kind
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
    span_lint(cx, MIN_IDENT_CHARS, span, help);
}

/// Attempt to convert the node to an [`ItemKind::Use`] node.
///
/// If it is, return the [`UsePath`] contained within.
fn opt_as_use_node(node: Node<'_>) -> Option<&'_ UsePath<'_>> {
    if let Node::Item(item) = node
        && let ItemKind::Use(path, _) = item.kind
    {
        Some(path)
    } else {
        None
    }
}

/// Check if a pattern is a function param in an impl block for a trait and that the param name is
/// the same than in the trait definition.
fn is_not_in_trait_impl(cx: &LateContext<'_>, pat: &Pat<'_>, ident: Ident) -> bool {
    let parent_node = cx.tcx.parent_hir_node(pat.hir_id);
    if !matches!(parent_node, Node::Param(_)) {
        return true;
    }

    for (_, parent_node) in cx.tcx.hir_parent_iter(pat.hir_id) {
        if let Node::ImplItem(impl_item) = parent_node
            && matches!(impl_item.kind, ImplItemKind::Fn(_, _))
        {
            let impl_parent_node = cx.tcx.parent_hir_node(impl_item.hir_id());
            if let Node::Item(parent_item) = impl_parent_node
                && let ItemKind::Impl(Impl { of_trait: Some(_), .. }) = &parent_item.kind
                && let Some(name) = get_param_name(impl_item, cx, ident)
            {
                return name != ident.name;
            }

            return true;
        }
    }

    true
}

fn get_param_name(impl_item: &ImplItem<'_>, cx: &LateContext<'_>, ident: Ident) -> Option<Symbol> {
    if let ImplItemImplKind::Trait { trait_item_def_id: Ok(trait_item_def_id), .. } = impl_item.impl_kind {
        let trait_param_names = cx.tcx.fn_arg_idents(trait_item_def_id);

        let ImplItemKind::Fn(_, body_id) = impl_item.kind else {
            return None;
        };

        if let Some(param_index) = cx
            .tcx
            .hir_body_param_idents(body_id)
            .position(|param_ident| param_ident.is_some_and(|param_ident| param_ident.span == ident.span))
            && let Some(trait_param_name) = trait_param_names.get(param_index)
            && let Some(trait_param_ident) = trait_param_name
        {
            return Some(trait_param_ident.name);
        }
    }

    None
}
