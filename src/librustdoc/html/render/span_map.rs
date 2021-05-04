use crate::clean;
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
    Local(Span),
    External(DefId),
}

/// This struct is used only as index in the `span_map`, not as [`Span`]! `Span`s contain
/// some extra information (the syntax context) we don't need. **Do not convert this type back to
/// `Span`!!!**
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
crate struct LightSpan {
    crate lo: u32,
    crate hi: u32,
}

impl LightSpan {
    /// Before explaining what this method does, some global explanations on rust's `Span`:
    ///
    /// Each source code file is stored in the source map in the compiler and has a
    /// `lo` and a `hi` (lowest and highest bytes in this source map which can be seen as one huge
    /// string to simplify things). So in this case, this represents the starting byte of the
    /// current file. It'll be used later on to retrieve the "definition span" from the
    /// `span_correspondance_map` (which is inside `context`).
    ///
    /// This when we transform the "span" we have from reading the input into a "span" which can be
    /// used as index to the `span_correspondance_map` to get the definition of this item.
    ///
    /// So in here, `file_span_lo` is representing the "lo" byte in the global source map, and to
    /// make our "span" works in there, we simply add `file_span_lo` to our values.
    crate fn new_in_file(file_span_lo: u32, lo: u32, hi: u32) -> Self {
        Self { lo: lo + file_span_lo, hi: hi + file_span_lo }
    }

    crate fn empty() -> Self {
        Self { lo: 0, hi: 0 }
    }

    /// Extra the `lo` and `hi` from the [`Span`] and discard the unused syntax context.
    fn new_from_span(sp: Span) -> Self {
        Self { lo: sp.lo().0, hi: sp.hi().0 }
    }
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
    krate: clean::Crate,
    src_root: &Path,
    include_sources: bool,
    generate_link_to_definition: bool,
) -> (clean::Crate, FxHashMap<PathBuf, String>, FxHashMap<LightSpan, LinkFromSrc>) {
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

struct SpanMapVisitor<'tcx> {
    crate tcx: TyCtxt<'tcx>,
    crate matches: FxHashMap<LightSpan, LinkFromSrc>,
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
                path_span
                    .map(LightSpan::new_from_span)
                    .unwrap_or_else(|| LightSpan::new_from_span(path.span)),
                LinkFromSrc::Local(span),
            );
        } else if let Some(def_id) = info {
            self.matches.insert(
                path_span
                    .map(LightSpan::new_from_span)
                    .unwrap_or_else(|| LightSpan::new_from_span(path.span)),
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
                        self.matches.insert(
                            LightSpan::new_from_span(item.ident.span),
                            LinkFromSrc::Local(m.inner),
                        );
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
                                self.matches.insert(
                                    LightSpan::new_from_span(method_span),
                                    LinkFromSrc::Local(span),
                                );
                            }
                            None => {
                                self.matches.insert(
                                    LightSpan::new_from_span(method_span),
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
