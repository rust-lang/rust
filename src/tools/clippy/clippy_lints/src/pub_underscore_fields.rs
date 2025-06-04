use clippy_config::Conf;
use clippy_config::types::PubUnderscoreFieldsBehaviour;
use clippy_utils::attrs::is_doc_hidden;
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::is_path_lang_item;
use rustc_hir::{FieldDef, Item, ItemKind, LangItem};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks whether any field of the struct is prefixed with an `_` (underscore) and also marked
    /// `pub` (public)
    ///
    /// ### Why is this bad?
    /// Fields prefixed with an `_` are inferred as unused, which suggests it should not be marked
    /// as `pub`, because marking it as `pub` infers it will be used.
    ///
    /// ### Example
    /// ```rust
    /// struct FileHandle {
    ///     pub _descriptor: usize,
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// struct FileHandle {
    ///     _descriptor: usize,
    /// }
    /// ```
    ///
    /// OR
    ///
    /// ```rust
    /// struct FileHandle {
    ///     pub descriptor: usize,
    /// }
    /// ```
    #[clippy::version = "1.77.0"]
    pub PUB_UNDERSCORE_FIELDS,
    pedantic,
    "struct field prefixed with underscore and marked public"
}

pub struct PubUnderscoreFields {
    behavior: PubUnderscoreFieldsBehaviour,
}
impl_lint_pass!(PubUnderscoreFields => [PUB_UNDERSCORE_FIELDS]);

impl PubUnderscoreFields {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            behavior: conf.pub_underscore_fields_behavior,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for PubUnderscoreFields {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        // This lint only pertains to structs.
        let ItemKind::Struct(_, _, variant_data) = &item.kind else {
            return;
        };

        let is_visible = |field: &FieldDef<'_>| match self.behavior {
            PubUnderscoreFieldsBehaviour::PubliclyExported => cx.effective_visibilities.is_reachable(field.def_id),
            PubUnderscoreFieldsBehaviour::AllPubFields => {
                // If there is a visibility span then the field is marked pub in some way.
                !field.vis_span.is_empty()
            },
        };

        for field in variant_data.fields() {
            // Only pertains to fields that start with an underscore, and are public.
            if field.ident.as_str().starts_with('_') && is_visible(field)
                // We ignore fields that have `#[doc(hidden)]`.
                && !is_doc_hidden(cx.tcx.hir_attrs(field.hir_id))
                // We ignore fields that are `PhantomData`.
                && !is_path_lang_item(cx, field.ty, LangItem::PhantomData)
            {
                span_lint_hir_and_then(
                    cx,
                    PUB_UNDERSCORE_FIELDS,
                    field.hir_id,
                    field.vis_span.to(field.ident.span),
                    "field marked as public but also inferred as unused because it's prefixed with `_`",
                    |diag| {
                        diag.help("consider removing the underscore, or making the field private");
                    },
                );
            }
        }
    }
}
