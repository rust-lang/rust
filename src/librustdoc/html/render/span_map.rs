use crate::clean::{self, rustc_span, PrimitiveType};
use crate::html::sources;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{ExprKind, HirId, Item, ItemKind, Mod, Node};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_span::hygiene::MacroKind;
use rustc_span::{BytePos, ExpnKind, Span};

use std::path::{Path, PathBuf};

/// This enum allows us to store two different kinds of information:
///
/// In case the `span` definition comes from the same crate, we can simply get the `span` and use
/// it as is.
///
/// Otherwise, we store the definition `DefId` and will generate a link to the documentation page
/// instead of the source code directly.
#[derive(Debug)]
pub(crate) enum LinkFromSrc {
    Local(clean::Span),
    External(DefId),
    Primitive(PrimitiveType),
    Doc(DefId),
}

/// This function will do at most two things:
///
/// 1. Generate a `span` correspondence map which links an item `span` to its definition `span`.
/// 2. Collect the source code files.
///
/// It returns the `krate`, the source code files and the `span` correspondence map.
///
/// Note about the `span` correspondence map: the keys are actually `(lo, hi)` of `span`s. We don't
/// need the `span` context later on, only their position, so instead of keep a whole `Span`, we
/// only keep the `lo` and `hi`.
pub(crate) fn collect_spans_and_sources(
    tcx: TyCtxt<'_>,
    krate: &clean::Crate,
    src_root: &Path,
    include_sources: bool,
    generate_link_to_definition: bool,
) -> (FxHashMap<PathBuf, String>, FxHashMap<Span, LinkFromSrc>) {
    let mut visitor = SpanMapVisitor { tcx, matches: FxHashMap::default() };

    if include_sources {
        if generate_link_to_definition {
            tcx.hir().walk_toplevel_module(&mut visitor);
        }
        let sources = sources::collect_local_sources(tcx, src_root, krate);
        (sources, visitor.matches)
    } else {
        (Default::default(), Default::default())
    }
}

struct SpanMapVisitor<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) matches: FxHashMap<Span, LinkFromSrc>,
}

impl<'tcx> SpanMapVisitor<'tcx> {
    /// This function is where we handle `hir::Path` elements and add them into the "span map".
    fn handle_path(&mut self, path: &rustc_hir::Path<'_>) {
        match path.res {
            // FIXME: For now, we handle `DefKind` if it's not a `DefKind::TyParam`.
            // Would be nice to support them too alongside the other `DefKind`
            // (such as primitive types!).
            Res::Def(kind, def_id) if kind != DefKind::TyParam => {
                let link = if def_id.as_local().is_some() {
                    LinkFromSrc::Local(rustc_span(def_id, self.tcx))
                } else {
                    LinkFromSrc::External(def_id)
                };
                self.matches.insert(path.span, link);
            }
            Res::Local(_) => {
                if let Some(span) = self.tcx.hir().res_span(path.res) {
                    self.matches.insert(path.span, LinkFromSrc::Local(clean::Span::new(span)));
                }
            }
            Res::PrimTy(p) => {
                // FIXME: Doesn't handle "path-like" primitives like arrays or tuples.
                self.matches.insert(path.span, LinkFromSrc::Primitive(PrimitiveType::from(p)));
            }
            Res::Err => {}
            _ => {}
        }
    }

    /// Used to generate links on items' definition to go to their documentation page.
    pub(crate) fn extract_info_from_hir_id(&mut self, hir_id: HirId) {
        if let Some(Node::Item(item)) = self.tcx.hir().find(hir_id) {
            if let Some(span) = self.tcx.def_ident_span(item.owner_id) {
                let cspan = clean::Span::new(span);
                // If the span isn't from the current crate, we ignore it.
                if cspan.inner().is_dummy() || cspan.cnum(self.tcx.sess) != LOCAL_CRATE {
                    return;
                }
                self.matches.insert(span, LinkFromSrc::Doc(item.owner_id.to_def_id()));
            }
        }
    }

    /// Adds the macro call into the span map. Returns `true` if the `span` was inside a macro
    /// expansion, whether or not it was added to the span map.
    ///
    /// The idea for the macro support is to check if the current `Span` comes from expansion. If
    /// so, we loop until we find the macro definition by using `outer_expn_data` in a loop.
    /// Finally, we get the information about the macro itself (`span` if "local", `DefId`
    /// otherwise) and store it inside the span map.
    fn handle_macro(&mut self, span: Span) -> bool {
        if !span.from_expansion() {
            return false;
        }
        // So if the `span` comes from a macro expansion, we need to get the original
        // macro's `DefId`.
        let mut data = span.ctxt().outer_expn_data();
        let mut call_site = data.call_site;
        // Macros can expand to code containing macros, which will in turn be expanded, etc.
        // So the idea here is to "go up" until we're back to code that was generated from
        // macro expansion so that we can get the `DefId` of the original macro that was at the
        // origin of this expansion.
        while call_site.from_expansion() {
            data = call_site.ctxt().outer_expn_data();
            call_site = data.call_site;
        }

        let macro_name = match data.kind {
            ExpnKind::Macro(MacroKind::Bang, macro_name) => macro_name,
            // Even though we don't handle this kind of macro, this `data` still comes from
            // expansion so we return `true` so we don't go any deeper in this code.
            _ => return true,
        };
        let link_from_src = match data.macro_def_id {
            Some(macro_def_id) => {
                if macro_def_id.is_local() {
                    LinkFromSrc::Local(clean::Span::new(data.def_site))
                } else {
                    LinkFromSrc::External(macro_def_id)
                }
            }
            None => return true,
        };
        let new_span = data.call_site;
        let macro_name = macro_name.as_str();
        // The "call_site" includes the whole macro with its "arguments". We only want
        // the macro name.
        let new_span = new_span.with_hi(new_span.lo() + BytePos(macro_name.len() as u32));
        self.matches.insert(new_span, link_from_src);
        true
    }
}

impl<'tcx> Visitor<'tcx> for SpanMapVisitor<'tcx> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_path(&mut self, path: &rustc_hir::Path<'tcx>, _id: HirId) {
        if self.handle_macro(path.span) {
            return;
        }
        self.handle_path(path);
        intravisit::walk_path(self, path);
    }

    fn visit_mod(&mut self, m: &'tcx Mod<'tcx>, span: Span, id: HirId) {
        // To make the difference between "mod foo {}" and "mod foo;". In case we "import" another
        // file, we want to link to it. Otherwise no need to create a link.
        if !span.overlaps(m.spans.inner_span) {
            // Now that we confirmed it's a file import, we want to get the span for the module
            // name only and not all the "mod foo;".
            if let Some(Node::Item(item)) = self.tcx.hir().find(id) {
                self.matches.insert(
                    item.ident.span,
                    LinkFromSrc::Local(clean::Span::new(m.spans.inner_span)),
                );
            }
        } else {
            // If it's a "mod foo {}", we want to look to its documentation page.
            self.extract_info_from_hir_id(id);
        }
        intravisit::walk_mod(self, m, id);
    }

    fn visit_expr(&mut self, expr: &'tcx rustc_hir::Expr<'tcx>) {
        if let ExprKind::MethodCall(segment, ..) = expr.kind {
            let hir = self.tcx.hir();
            let body_id = hir.enclosing_body_owner(segment.hir_id);
            // FIXME: this is showing error messages for parts of the code that are not
            // compiled (because of cfg)!
            //
            // See discussion in https://github.com/rust-lang/rust/issues/69426#issuecomment-1019412352
            let typeck_results = self
                .tcx
                .typeck_body(hir.maybe_body_owned_by(body_id).expect("a body which isn't a body"));
            if let Some(def_id) = typeck_results.type_dependent_def_id(expr.hir_id) {
                let link = if def_id.as_local().is_some() {
                    LinkFromSrc::Local(rustc_span(def_id, self.tcx))
                } else {
                    LinkFromSrc::External(def_id)
                };
                self.matches.insert(segment.ident.span, link);
            }
        } else if self.handle_macro(expr.span) {
            // We don't want to go deeper into the macro.
            return;
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        match item.kind {
            ItemKind::Static(_, _, _)
            | ItemKind::Const(_, _)
            | ItemKind::Fn(_, _, _)
            | ItemKind::Macro(_, _)
            | ItemKind::TyAlias(_, _)
            | ItemKind::Enum(_, _)
            | ItemKind::Struct(_, _)
            | ItemKind::Union(_, _)
            | ItemKind::Trait(_, _, _, _, _)
            | ItemKind::TraitAlias(_, _) => self.extract_info_from_hir_id(item.hir_id()),
            ItemKind::Impl(_)
            | ItemKind::Use(_, _)
            | ItemKind::ExternCrate(_)
            | ItemKind::ForeignMod { .. }
            | ItemKind::GlobalAsm(_)
            | ItemKind::OpaqueTy(_)
            // We already have "visit_mod" above so no need to check it here.
            | ItemKind::Mod(_) => {}
        }
        intravisit::walk_item(self, item);
    }
}
