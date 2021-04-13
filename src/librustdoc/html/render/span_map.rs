use crate::clean;
use crate::html::sources;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{/*ExprKind, */ GenericParam, GenericParamKind, HirId};
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

    // fn visit_expr(&mut self, expr: &'tcx rustc_hir::Expr<'tcx>) {
    //     match expr.kind {
    //         ExprKind::MethodCall(segment, method_span, _, _) => {
    //             if let Some(hir_id) = segment.hir_id {
    //                 // call https://doc.rust-lang.org/beta/nightly-rustc/rustc_middle/ty/context/struct.TypeckResults.html#method.type_dependent_def_id
    //             }
    //         }
    //         _ => {}
    //     }
    //     intravisit::walk_expr(self, expr);
    // }

    fn visit_use(&mut self, path: &'tcx rustc_hir::Path<'tcx>, id: HirId) {
        self.handle_path(path, None);
        intravisit::walk_use(self, path, id);
    }
}
