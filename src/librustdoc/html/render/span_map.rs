use crate::clean::{self, PrimitiveType};
use crate::html::sources;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{ExprKind, GenericParam, GenericParamKind, HirId, Mod, Node};
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
crate enum LinkFromSrc {
    Local(clean::Span),
    External(DefId),
    Primitive(PrimitiveType),
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
crate fn collect_spans_and_sources(
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
        let sources = sources::collect_local_sources(tcx, src_root, &krate);
        (sources, visitor.matches)
    } else {
        (Default::default(), Default::default())
    }
}

struct SpanMapVisitor<'tcx> {
    crate tcx: TyCtxt<'tcx>,
    crate matches: FxHashMap<Span, LinkFromSrc>,
}

impl<'tcx> SpanMapVisitor<'tcx> {
    /// This function is where we handle `hir::Path` elements and add them into the "span map".
    fn handle_path(&mut self, path: &rustc_hir::Path<'_>, path_span: Option<Span>) {
        let info = match path.res {
            // FIXME: For now, we only handle `DefKind` if it's not `DefKind::TyParam` or
            // `DefKind::Macro`. Would be nice to support them too alongside the other `DefKind`
            // (such as primitive types!).
            Res::Def(kind, def_id) if kind != DefKind::TyParam => {
                if matches!(kind, DefKind::Macro(_)) {
                    return;
                }
                Some(def_id)
            }
            Res::Local(_) => None,
            Res::PrimTy(p) => {
                // FIXME: Doesn't handle "path-like" primitives like arrays or tuples.
                let span = path_span.unwrap_or(path.span);
                self.matches.insert(span, LinkFromSrc::Primitive(PrimitiveType::from(p)));
                return;
            }
            Res::Err => return,
            _ => return,
        };
        if let Some(span) = self.tcx.hir().res_span(path.res) {
            self.matches
                .insert(path_span.unwrap_or(path.span), LinkFromSrc::Local(clean::Span::new(span)));
        } else if let Some(def_id) = info {
            self.matches.insert(path_span.unwrap_or(path.span), LinkFromSrc::External(def_id));
        }
    }
}

impl<'tcx> Visitor<'tcx> for SpanMapVisitor<'tcx> {
    type Map = rustc_middle::hir::map::Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.tcx.hir())
    }

    fn visit_generic_param(&mut self, p: &'tcx GenericParam<'tcx>) {
        if !matches!(p.kind, GenericParamKind::Type { .. }) {
            return;
        }
        for bound in p.bounds {
            if let Some(trait_ref) = bound.trait_ref() {
                self.handle_path(trait_ref.path, None);
            }
        }
    }

    fn visit_path(&mut self, path: &'tcx rustc_hir::Path<'tcx>, _id: HirId) {
        self.handle_path(path, None);
        intravisit::walk_path(self, path);
    }

    fn visit_mod(&mut self, m: &'tcx Mod<'tcx>, span: Span, id: HirId) {
        // To make the difference between "mod foo {}" and "mod foo;". In case we "import" another
        // file, we want to link to it. Otherwise no need to create a link.
        if !span.overlaps(m.inner) {
            // Now that we confirmed it's a file import, we want to get the span for the module
            // name only and not all the "mod foo;".
            if let Some(Node::Item(item)) = self.tcx.hir().find(id) {
                self.matches.insert(item.ident.span, LinkFromSrc::Local(clean::Span::new(m.inner)));
            }
        }
        intravisit::walk_mod(self, m, id);
    }

    fn visit_expr(&mut self, expr: &'tcx rustc_hir::Expr<'tcx>) {
        if let ExprKind::MethodCall(segment, method_span, _, _) = expr.kind {
            if let Some(hir_id) = segment.hir_id {
                let hir = self.tcx.hir();
                let body_id = hir.enclosing_body_owner(hir_id);
                let typeck_results = self.tcx.sess.with_disabled_diagnostic(|| {
                    self.tcx.typeck_body(
                        hir.maybe_body_owned_by(body_id).expect("a body which isn't a body"),
                    )
                });
                if let Some(def_id) = typeck_results.type_dependent_def_id(expr.hir_id) {
                    self.matches.insert(
                        method_span,
                        match hir.span_if_local(def_id) {
                            Some(span) => LinkFromSrc::Local(clean::Span::new(span)),
                            None => LinkFromSrc::External(def_id),
                        },
                    );
                }
            }
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_use(&mut self, path: &'tcx rustc_hir::Path<'tcx>, id: HirId) {
        self.handle_path(path, None);
        intravisit::walk_use(self, path, id);
    }
}
