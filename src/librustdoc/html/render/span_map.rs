use crate::clean::types::rustc_span;
use crate::clean::{self, PrimitiveType};
use crate::html::sources;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{ExprKind, HirId, Item, ItemKind, Mod, Node};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

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
    External(clean::Span),
    Primitive(PrimitiveType),
    Doc(DefId),
}

/// This function will do at most two things:
///
/// 1. Generate a `span` correspondance map which links an item `span` to its definition `span`.
/// 2. Collect the source code files.
///
/// It returns the `krate`, the source code files and the `span` correspondance map.
///
/// Note about the `span` correspondance map: the keys are actually `(lo, hi)` of `span`s. We don't
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
    fn handle_path(&mut self, path: &rustc_hir::Path<'_>, path_span: Option<Span>) {
        match path.res {
            // FIXME: For now, we only handle `DefKind` if it's not `DefKind::TyParam` or
            // `DefKind::Macro`. Would be nice to support them too alongside the other `DefKind`
            // (such as primitive types!).
            Res::Def(kind, def_id) if kind != DefKind::TyParam => {
                if matches!(kind, DefKind::Macro(_)) {
                    return;
                }
                let span = rustc_span(def_id, self.tcx);
                let link = if def_id.as_local().is_some() {
                    LinkFromSrc::Local(span)
                } else {
                    LinkFromSrc::External(span)
                };
                self.matches.insert(path_span.unwrap_or(path.span), link);
            }
            Res::Local(_) => {
                if let Some(span) = self.tcx.hir().res_span(path.res) {
                    self.matches.insert(
                        path_span.unwrap_or(path.span),
                        LinkFromSrc::Local(clean::Span::new(span)),
                    );
                }
            }
            Res::PrimTy(p) => {
                // FIXME: Doesn't handle "path-like" primitives like arrays or tuples.
                let span = path_span.unwrap_or(path.span);
                self.matches.insert(span, LinkFromSrc::Primitive(PrimitiveType::from(p)));
            }
            Res::Err => {}
            _ => {}
        }
    }

    /// Used to generate links on items' definition to go to their documentation page.
    pub(crate) fn extract_info_from_hir_id(&mut self, hir_id: HirId) {
        if let Some(def_id) = self.tcx.hir().opt_local_def_id(hir_id) {
            if let Some(span) = self.tcx.def_ident_span(def_id) {
                let cspan = clean::Span::new(span);
                let def_id = def_id.to_def_id();
                // If the span isn't from the current crate, we ignore it.
                if cspan.is_dummy() || cspan.cnum(self.tcx.sess) != LOCAL_CRATE {
                    return;
                }
                self.matches.insert(span, LinkFromSrc::Doc(def_id));
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for SpanMapVisitor<'tcx> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_path(&mut self, path: &'tcx rustc_hir::Path<'tcx>, _id: HirId) {
        self.handle_path(path, None);
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
            if let Some(hir_id) = segment.hir_id {
                let hir = self.tcx.hir();
                let body_id = hir.enclosing_body_owner(hir_id);
                // FIXME: this is showing error messages for parts of the code that are not
                // compiled (because of cfg)!
                //
                // See discussion in https://github.com/rust-lang/rust/issues/69426#issuecomment-1019412352
                let typeck_results = self.tcx.typeck_body(
                    hir.maybe_body_owned_by(body_id).expect("a body which isn't a body"),
                );
                if let Some(def_id) = typeck_results.type_dependent_def_id(expr.hir_id) {
                    let span = rustc_span(def_id, self.tcx);
                    let link = if def_id.as_local().is_some() {
                        LinkFromSrc::Local(span)
                    } else {
                        LinkFromSrc::External(span)
                    };
                    self.matches.insert(segment.ident.span, link);
                }
            }
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_use(&mut self, path: &'tcx rustc_hir::Path<'tcx>, id: HirId) {
        self.handle_path(path, None);
        intravisit::walk_use(self, path, id);
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
