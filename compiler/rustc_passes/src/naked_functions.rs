use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{ErasedMap, FnKind, NestedVisitorMap, Visitor};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;
use rustc_span::Span;

fn check_mod_naked_functions(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut CheckNakedFunctions { tcx }.as_deep_visitor(),
    );
}

crate fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_naked_functions, ..*providers };
}

struct CheckNakedFunctions<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for CheckNakedFunctions<'tcx> {
    type Map = ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_fn(
        &mut self,
        fk: FnKind<'v>,
        _fd: &'tcx hir::FnDecl<'tcx>,
        body_id: hir::BodyId,
        _span: Span,
        _hir_id: hir::HirId,
    ) {
        match fk {
            // Rejected during attribute check. Do not validate further.
            FnKind::Closure(..) => return,
            FnKind::ItemFn(..) | FnKind::Method(..) => {}
        }

        let naked = fk.attrs().iter().any(|attr| attr.has_name(sym::naked));
        if naked {
            let body = self.tcx.hir().body(body_id);
            check_params(self.tcx, body);
            check_body(self.tcx, body);
        }
    }
}

/// Checks that parameters don't use patterns. Mirrors the checks for function declarations.
fn check_params(tcx: TyCtxt<'_>, body: &hir::Body<'_>) {
    for param in body.params {
        match param.pat.kind {
            hir::PatKind::Wild
            | hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, _, _, None) => {}
            _ => {
                tcx.sess
                    .struct_span_err(
                        param.pat.span,
                        "patterns not allowed in naked function parameters",
                    )
                    .emit();
            }
        }
    }
}

/// Checks that function parameters aren't referenced in the function body.
fn check_body<'tcx>(tcx: TyCtxt<'tcx>, body: &'tcx hir::Body<'tcx>) {
    let mut params = hir::HirIdSet::default();
    for param in body.params {
        param.pat.each_binding(|_binding_mode, hir_id, _span, _ident| {
            params.insert(hir_id);
        });
    }
    CheckBody { tcx, params }.visit_body(body);
}

struct CheckBody<'tcx> {
    tcx: TyCtxt<'tcx>,
    params: hir::HirIdSet,
}

impl<'tcx> Visitor<'tcx> for CheckBody<'tcx> {
    type Map = ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Path(hir::QPath::Resolved(
            _,
            hir::Path { res: hir::def::Res::Local(var_hir_id), .. },
        )) = expr.kind
        {
            if self.params.contains(var_hir_id) {
                self.tcx
                    .sess
                    .struct_span_err(
                        expr.span,
                        "use of parameters not allowed inside naked functions",
                    )
                    .emit();
            }
        }
        hir::intravisit::walk_expr(self, expr);
    }
}
