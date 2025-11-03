use std::path::{Path, PathBuf};

use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::intravisit::{self, Visitor, VisitorExt};
use rustc_hir::{ExprKind, HirId, Item, ItemKind, Mod, Node, QPath};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_span::{BytePos, ExpnKind};

use crate::clean::{self, PrimitiveType, rustc_span};
use crate::html::sources;

/// This is a stripped down version of [`rustc_span::Span`] that only contains the start and end byte positions of the span.
///
/// Profiling showed that the `Span` interner was taking up a lot of the run-time when highlighting, and since we
/// never actually use the context and parent that are stored in a normal `Span`, we can replace its usages with this
/// one, which is much cheaper to construct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Span {
    lo: BytePos,
    hi: BytePos,
}

impl From<rustc_span::Span> for Span {
    fn from(value: rustc_span::Span) -> Self {
        Self { lo: value.lo(), hi: value.hi() }
    }
}

impl Span {
    pub(crate) fn lo(self) -> BytePos {
        self.lo
    }

    pub(crate) fn hi(self) -> BytePos {
        self.hi
    }

    pub(crate) fn with_lo(self, lo: BytePos) -> Self {
        Self { lo, hi: self.hi() }
    }

    pub(crate) fn with_hi(self, hi: BytePos) -> Self {
        Self { lo: self.lo(), hi }
    }
}

pub(crate) const DUMMY_SP: Span = Span { lo: BytePos(0), hi: BytePos(0) };

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
/// It returns the source code files and the `span` correspondence map.
///
/// Note about the `span` correspondence map: the keys are actually `(lo, hi)` of `span`s. We don't
/// need the `span` context later on, only their position, so instead of keeping a whole `Span`, we
/// only keep the `lo` and `hi`.
pub(crate) fn collect_spans_and_sources(
    tcx: TyCtxt<'_>,
    krate: &clean::Crate,
    src_root: &Path,
    include_sources: bool,
    generate_link_to_definition: bool,
) -> (FxIndexMap<PathBuf, String>, FxHashMap<Span, LinkFromSrc>) {
    if include_sources {
        let mut visitor = SpanMapVisitor { tcx, matches: FxHashMap::default() };

        if generate_link_to_definition {
            tcx.hir_walk_toplevel_module(&mut visitor);
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

impl SpanMapVisitor<'_> {
    /// This function is where we handle `hir::Path` elements and add them into the "span map".
    fn handle_path(&mut self, path: &rustc_hir::Path<'_>, only_use_last_segment: bool) {
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
                // In case the path ends with generics, we remove them from the span.
                let span = if only_use_last_segment
                    && let Some(path_span) = path.segments.last().map(|segment| segment.ident.span)
                {
                    path_span
                } else {
                    path.segments
                        .last()
                        .map(|last| {
                            // In `use` statements, the included item is not in the path segments.
                            // However, it doesn't matter because you can't have generics on `use`
                            // statements.
                            if path.span.contains(last.ident.span) {
                                path.span.with_hi(last.ident.span.hi())
                            } else {
                                path.span
                            }
                        })
                        .unwrap_or(path.span)
                };
                self.matches.insert(span.into(), link);
            }
            Res::Local(_) if let Some(span) = self.tcx.hir_res_span(path.res) => {
                let path_span = if only_use_last_segment
                    && let Some(path_span) = path.segments.last().map(|segment| segment.ident.span)
                {
                    path_span
                } else {
                    path.span
                };
                self.matches.insert(path_span.into(), LinkFromSrc::Local(clean::Span::new(span)));
            }
            Res::PrimTy(p) => {
                // FIXME: Doesn't handle "path-like" primitives like arrays or tuples.
                self.matches
                    .insert(path.span.into(), LinkFromSrc::Primitive(PrimitiveType::from(p)));
            }
            Res::Err => {}
            _ => {}
        }
    }

    /// Used to generate links on items' definition to go to their documentation page.
    pub(crate) fn extract_info_from_hir_id(&mut self, hir_id: HirId) {
        if let Node::Item(item) = self.tcx.hir_node(hir_id)
            && let Some(span) = self.tcx.def_ident_span(item.owner_id)
        {
            let cspan = clean::Span::new(span);
            // If the span isn't from the current crate, we ignore it.
            if cspan.inner().is_dummy() || cspan.cnum(self.tcx.sess) != LOCAL_CRATE {
                return;
            }
            self.matches.insert(span.into(), LinkFromSrc::Doc(item.owner_id.to_def_id()));
        }
    }

    /// Adds the macro call into the span map. Returns `true` if the `span` was inside a macro
    /// expansion, whether or not it was added to the span map.
    ///
    /// The idea for the macro support is to check if the current `Span` comes from expansion. If
    /// so, we loop until we find the macro definition by using `outer_expn_data` in a loop.
    /// Finally, we get the information about the macro itself (`span` if "local", `DefId`
    /// otherwise) and store it inside the span map.
    fn handle_macro(&mut self, span: rustc_span::Span) -> bool {
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
            ExpnKind::Macro(_, macro_name) => macro_name,
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
        self.matches.insert(new_span.into(), link_from_src);
        true
    }

    fn infer_id(&mut self, hir_id: HirId, expr_hir_id: Option<HirId>, span: Span) {
        let tcx = self.tcx;
        let body_id = tcx.hir_enclosing_body_owner(hir_id);
        // FIXME: this is showing error messages for parts of the code that are not
        // compiled (because of cfg)!
        //
        // See discussion in https://github.com/rust-lang/rust/issues/69426#issuecomment-1019412352
        let typeck_results = tcx.typeck_body(tcx.hir_body_owned_by(body_id).id());
        // Interestingly enough, for method calls, we need the whole expression whereas for static
        // method/function calls, we need the call expression specifically.
        if let Some(def_id) = typeck_results.type_dependent_def_id(expr_hir_id.unwrap_or(hir_id)) {
            let link = if def_id.as_local().is_some() {
                LinkFromSrc::Local(rustc_span(def_id, tcx))
            } else {
                LinkFromSrc::External(def_id)
            };
            self.matches.insert(span, link);
        }
    }
}

// This is a reimplementation of `hir_enclosing_body_owner` which allows to fail without
// panicking.
fn hir_enclosing_body_owner(tcx: TyCtxt<'_>, hir_id: HirId) -> Option<LocalDefId> {
    for (_, node) in tcx.hir_parent_iter(hir_id) {
        // FIXME: associated type impl items don't have an associated body, so we don't handle
        // them currently.
        if let Node::ImplItem(impl_item) = node
            && matches!(impl_item.kind, rustc_hir::ImplItemKind::Type(_))
        {
            return None;
        } else if let Some((def_id, _)) = node.associated_body() {
            return Some(def_id);
        }
    }
    None
}

impl<'tcx> Visitor<'tcx> for SpanMapVisitor<'tcx> {
    type NestedFilter = nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_path(&mut self, path: &rustc_hir::Path<'tcx>, _id: HirId) {
        if self.handle_macro(path.span) {
            return;
        }
        self.handle_path(path, false);
        intravisit::walk_path(self, path);
    }

    fn visit_qpath(&mut self, qpath: &QPath<'tcx>, id: HirId, _span: rustc_span::Span) {
        match *qpath {
            QPath::TypeRelative(qself, path) => {
                if matches!(path.res, Res::Err) {
                    let tcx = self.tcx;
                    if let Some(body_id) = hir_enclosing_body_owner(tcx, id) {
                        let typeck_results = tcx.typeck_body(tcx.hir_body_owned_by(body_id).id());
                        let path = rustc_hir::Path {
                            // We change the span to not include parens.
                            span: path.ident.span,
                            res: typeck_results.qpath_res(qpath, id),
                            segments: &[],
                        };
                        self.handle_path(&path, false);
                    }
                } else {
                    self.infer_id(path.hir_id, Some(id), path.ident.span.into());
                }

                rustc_ast::visit::try_visit!(self.visit_ty_unambig(qself));
                self.visit_path_segment(path);
            }
            QPath::Resolved(maybe_qself, path) => {
                self.handle_path(path, true);

                rustc_ast::visit::visit_opt!(self, visit_ty_unambig, maybe_qself);
                if !self.handle_macro(path.span) {
                    intravisit::walk_path(self, path);
                }
            }
        }
    }

    fn visit_mod(&mut self, m: &'tcx Mod<'tcx>, span: rustc_span::Span, id: HirId) {
        // To make the difference between "mod foo {}" and "mod foo;". In case we "import" another
        // file, we want to link to it. Otherwise no need to create a link.
        if !span.overlaps(m.spans.inner_span) {
            // Now that we confirmed it's a file import, we want to get the span for the module
            // name only and not all the "mod foo;".
            if let Node::Item(item) = self.tcx.hir_node(id) {
                let (ident, _) = item.expect_mod();
                self.matches.insert(
                    ident.span.into(),
                    LinkFromSrc::Local(clean::Span::new(m.spans.inner_span)),
                );
            }
        } else {
            // If it's a "mod foo {}", we want to look to its documentation page.
            self.extract_info_from_hir_id(id);
        }
        intravisit::walk_mod(self, m);
    }

    fn visit_expr(&mut self, expr: &'tcx rustc_hir::Expr<'tcx>) {
        match expr.kind {
            ExprKind::MethodCall(segment, ..) => {
                self.infer_id(segment.hir_id, Some(expr.hir_id), segment.ident.span.into())
            }
            ExprKind::Call(call, ..) => self.infer_id(call.hir_id, None, call.span.into()),
            _ => {
                if self.handle_macro(expr.span) {
                    // We don't want to go deeper into the macro.
                    return;
                }
            }
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        match item.kind {
            ItemKind::Static(..)
            | ItemKind::Const(..)
            | ItemKind::Fn { .. }
            | ItemKind::Macro(..)
            | ItemKind::TyAlias(..)
            | ItemKind::Enum(..)
            | ItemKind::Struct(..)
            | ItemKind::Union(..)
            | ItemKind::Trait(..)
            | ItemKind::TraitAlias(..) => self.extract_info_from_hir_id(item.hir_id()),
            ItemKind::Impl(_)
            | ItemKind::Use(..)
            | ItemKind::ExternCrate(..)
            | ItemKind::ForeignMod { .. }
            | ItemKind::GlobalAsm { .. }
            // We already have "visit_mod" above so no need to check it here.
            | ItemKind::Mod(..) => {}
        }
        intravisit::walk_item(self, item);
    }
}
