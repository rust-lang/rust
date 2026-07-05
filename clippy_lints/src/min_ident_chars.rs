use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::is_from_proc_macro;
use core::iter;
use core::num::NonZero;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::pluralize;
use rustc_hir::{
    FieldDef, HirId, ImplItem, ImplItemImplKind, ImplItemKind, Item, ItemKind, Node, Pat, PatKind, TraitFn, TraitItem,
    TraitItemKind, UseKind, Variant,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::{Ident, Symbol};
use std::borrow::Cow::{self, Borrowed, Owned};

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
    min_chars_threshold: u64,
    /// The number of parameter bindings which still need to be skipped for the current
    /// function.
    ///
    /// Impl trait functions have special rules are handled in `check_impl_item` since
    /// they have special rules. This is used to signal to `check_pat` the number of parameter
    current_fn_params_remaining: u32,
}

impl MinIdentChars {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            allowed_idents_below_min_chars: conf.allowed_idents_below_min_chars.iter().cloned().collect(),
            min_chars_threshold: conf.min_ident_chars_threshold,
            current_fn_params_remaining: 0,
        }
    }

    #[expect(clippy::cast_possible_truncation)]
    fn check_sym(&self, s: Symbol) -> Option<NonZero<usize>> {
        let s = s.as_str();
        if s.len() / 4 <= self.min_chars_threshold as usize
            && !s.starts_with('_')
            && !self.allowed_idents_below_min_chars.contains(s)
            && let len = s.chars().count()
            && let Some(missing) = (self.min_chars_threshold as usize).checked_sub(len)
        {
            NonZero::new(missing + 1)
        } else {
            None
        }
    }

    fn lint_msg(&self, missing: NonZero<usize>) -> Cow<'static, str> {
        if self.min_chars_threshold == 1 {
            Borrowed("this identifier consists of only a single character")
        } else {
            let n = missing.get();
            Owned(format!("this identifier is {n} character{} too short", pluralize!(n)))
        }
    }

    fn emit(&self, cx: &LateContext<'_>, ident: Ident, missing: NonZero<usize>) {
        if !ident.span.in_external_macro(cx.tcx.sess.source_map()) && !is_from_proc_macro(cx, &ident) {
            span_lint_and_then(cx, MIN_IDENT_CHARS, ident.span, self.lint_msg(missing), |diag| {
                // FIXME(@Jarcho): Capture a span from the config and point to it here.
                if self.min_chars_threshold != 1 {
                    diag.note_once(format!("the configured threshold is {}", self.min_chars_threshold));
                }
            });
        }
    }

    fn emit_hir(&self, cx: &LateContext<'_>, id: HirId, ident: Ident, missing: NonZero<usize>) {
        if !ident.span.in_external_macro(cx.tcx.sess.source_map()) && !is_from_proc_macro(cx, &ident) {
            span_lint_hir_and_then(cx, MIN_IDENT_CHARS, id, ident.span, self.lint_msg(missing), |diag| {
                diag.note_once(format!("the configured threshold is {}", self.min_chars_threshold));
            });
        }
    }
}

impl LateLintPass<'_> for MinIdentChars {
    fn check_item(&mut self, cx: &LateContext<'_>, i: &Item<'_>) {
        if self.min_chars_threshold == 0 {
            return;
        }
        let ident = match i.kind {
            ItemKind::Const(ident, ..)
            | ItemKind::Enum(ident, ..)
            | ItemKind::ExternCrate(None, ident)
            | ItemKind::Fn { ident, .. }
            | ItemKind::Macro(ident, ..)
            | ItemKind::Mod(ident, _)
            | ItemKind::Static(_, ident, ..)
            | ItemKind::Struct(ident, ..)
            | ItemKind::Trait { ident, .. }
            | ItemKind::TraitAlias(_, ident, ..)
            | ItemKind::TyAlias(ident, ..)
            | ItemKind::Union(ident, ..) => ident,
            ItemKind::Use(path, UseKind::Single(ident))
                if path.segments.last().is_some_and(|p| p.ident.span != ident.span) =>
            {
                ident
            },

            ItemKind::ExternCrate(..)
            | ItemKind::ForeignMod { .. }
            | ItemKind::GlobalAsm { .. }
            | ItemKind::Impl(_)
            | ItemKind::Use(..) => return,
        };
        if let Some(missing) = self.check_sym(ident.name)
            && !(matches!(i.kind, ItemKind::Fn { .. })
                && cx.tcx.codegen_fn_attrs(i.owner_id.def_id).contains_extern_indicator())
        {
            self.emit(cx, ident, missing);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, i: &TraitItem<'_>) {
        if self.min_chars_threshold == 0 {
            return;
        }
        if let TraitItemKind::Fn(_, TraitFn::Required(idents)) = i.kind {
            for &ident in idents {
                if let Some(ident) = ident
                    && let Some(missing) = self.check_sym(ident.name)
                {
                    self.emit(cx, ident, missing);
                }
            }
        }
        if let Some(missing) = self.check_sym(i.ident.name)
            && !(matches!(i.kind, TraitItemKind::Fn(_, TraitFn::Provided(_)))
                && cx.tcx.codegen_fn_attrs(i.owner_id.def_id).contains_extern_indicator())
        {
            self.emit(cx, i.ident, missing);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, i: &ImplItem<'_>) {
        if self.min_chars_threshold == 0 {
            return;
        }
        if let ImplItemImplKind::Trait {
            trait_item_def_id: Ok(trait_item),
            ..
        } = i.impl_kind
        {
            // For trait impls only lint if the parameter names are different from the trait's.
            if let ImplItemKind::Fn(_, body) = i.kind {
                let mut named_params_count = 0;
                for (trait_ident, param) in iter::zip(cx.tcx.fn_arg_idents(trait_item), cx.tcx.hir_body(body).params) {
                    if let PatKind::Binding(_, _, ident, _) = param.pat.kind {
                        named_params_count += 1;
                        if trait_ident.is_none_or(|trait_ident| ident.name != trait_ident.name)
                            && let Some(missing) = self.check_sym(ident.name)
                        {
                            self.emit_hir(cx, param.pat.hir_id, ident, missing);
                        }
                    }
                }
                // Signal the number of parameters to skip to `check_pat`.
                // This needs to be added since impl items can technically appear inside
                // both the function signature and generic predicates.
                self.current_fn_params_remaining += named_params_count;
            }
        } else if let Some(missing) = self.check_sym(i.ident.name)
            && !(matches!(i.kind, ImplItemKind::Fn(..))
                && cx.tcx.codegen_fn_attrs(i.owner_id.def_id).contains_extern_indicator())
        {
            self.emit(cx, i.ident, missing);
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'_>, f: &FieldDef<'_>) {
        if let Some(missing) = self.check_sym(f.ident.name) {
            self.emit(cx, f.ident, missing);
        }
    }

    fn check_variant(&mut self, cx: &LateContext<'_>, v: &Variant<'_>) {
        if let Some(missing) = self.check_sym(v.ident.name) {
            self.emit(cx, v.ident, missing);
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &Pat<'_>) {
        if let PatKind::Binding(_, _, ident, ..) = pat.kind
            && self.min_chars_threshold != 0
        {
            if self.current_fn_params_remaining != 0
                && let Node::Param(_) = cx.tcx.parent_hir_node(pat.hir_id)
            {
                self.current_fn_params_remaining -= 1;
            } else if let Some(missing) = self.check_sym(ident.name) {
                self.emit(cx, ident, missing);
            }
        }
    }
}
