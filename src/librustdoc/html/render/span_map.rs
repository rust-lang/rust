use crate::clean;
use crate::html::sources;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{ExprKind, GenericParam, GenericParamKind, HirId, Mod, Node};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

#[derive(Debug)]
crate enum LinkFromSrc {
    Local(Span),
    External(DefId),
}

crate fn collect_spans_and_sources(
    tcx: TyCtxt<'_>,
    krate: clean::Crate,
    src_root: &std::path::Path,
    include_sources: bool,
    generate_link_to_definition: bool,
) -> (clean::Crate, FxHashMap<std::path::PathBuf, String>, FxHashMap<(u32, u32), LinkFromSrc>) {
    let mut visitor = SpanMapVisitor { tcx, matches: FxHashMap::default() };

    if include_sources {
        if generate_link_to_definition {
            intravisit::walk_crate(&mut visitor, tcx.hir().krate());
        }
        let (krate, sources) = sources::collect_local_sources(tcx, src_root, krate);
        (krate, sources, visitor.matches)
    } else {
        (krate, Default::default(), Default::default())
    }
}

fn span_to_tuple(span: Span) -> (u32, u32) {
    (span.lo().0, span.hi().0)
}

struct SpanMapVisitor<'tcx> {
    crate tcx: TyCtxt<'tcx>,
    crate matches: FxHashMap<(u32, u32), LinkFromSrc>,
}

impl<'tcx> SpanMapVisitor<'tcx> {
    fn handle_path(&mut self, path: &rustc_hir::Path<'_>, path_span: Option<Span>) -> bool {
        let info = match path.res {
            Res::Def(kind, def_id) if kind != DefKind::TyParam => {
                if matches!(kind, DefKind::Macro(_)) {
                    return false;
                }
                Some(def_id)
            }
            Res::Local(_) => None,
            _ => return true,
        };
        if let Some(span) = self.tcx.hir().res_span(path.res) {
            self.matches.insert(
                path_span.map(span_to_tuple).unwrap_or_else(|| span_to_tuple(path.span)),
                LinkFromSrc::Local(span),
            );
        } else if let Some(def_id) = info {
            self.matches.insert(
                path_span.map(span_to_tuple).unwrap_or_else(|| span_to_tuple(path.span)),
                LinkFromSrc::External(def_id),
            );
        }
        true
    }
}

impl Visitor<'tcx> for SpanMapVisitor<'tcx> {
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
                self.handle_path(&trait_ref.path, None);
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
            if let Some(node) = self.tcx.hir().find(id) {
                match node {
                    Node::Item(item) => {
                        self.matches
                            .insert(span_to_tuple(item.ident.span), LinkFromSrc::Local(m.inner));
                    }
                    _ => {}
                }
            }
        }
        intravisit::walk_mod(self, m, id);
    }

    fn visit_expr(&mut self, expr: &'tcx rustc_hir::Expr<'tcx>) {
        match expr.kind {
            ExprKind::MethodCall(segment, method_span, _, _) => {
                if let Some(hir_id) = segment.hir_id {
                    let hir = self.tcx.hir();
                    let body_id = hir.enclosing_body_owner(hir_id);
                    // FIXME: this is showing error messages for parts of the code that are not
                    // compiled (because of cfg)!
                    let typeck_results = self.tcx.typeck_body(
                        hir.maybe_body_owned_by(body_id).expect("a body which isn't a body"),
                    );
                    if let Some(def_id) = typeck_results.type_dependent_def_id(expr.hir_id) {
                        match hir.span_if_local(def_id) {
                            Some(span) => {
                                self.matches
                                    .insert(span_to_tuple(method_span), LinkFromSrc::Local(span));
                            }
                            None => {
                                self.matches.insert(
                                    span_to_tuple(method_span),
                                    LinkFromSrc::External(def_id),
                                );
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_use(&mut self, path: &'tcx rustc_hir::Path<'tcx>, id: HirId) {
        self.handle_path(path, None);
        intravisit::walk_use(self, path, id);
    }
}
