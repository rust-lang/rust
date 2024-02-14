use rustc_ast::{
    mut_visit::{self, MutVisitor},
    ptr::P,
    TraitObjectSyntax, Ty, TyKind,
};
use rustc_hir::def::DefKind;

use crate::{Res, Resolver};

struct LateDesugarVisitor<'a, 'b, 'tcx> {
    r: &'b mut Resolver<'a, 'tcx>,
}

impl<'a, 'b, 'tcx> LateDesugarVisitor<'a, 'b, 'tcx> {
    fn new(resolver: &'b mut Resolver<'a, 'tcx>) -> Self {
        LateDesugarVisitor { r: resolver }
    }
}

impl MutVisitor for LateDesugarVisitor<'_, '_, '_> {
    fn visit_ty(&mut self, ty: &mut P<Ty>) {
        if let TyKind::Path(None, _path) = &ty.kind
            && let Some(partial_res) = self.r.partial_res_map.get(&ty.id)
            && let Some(res @ Res::Def(DefKind::Trait | DefKind::TraitAlias, _)) =
                partial_res.full_res()
        {
            debug!("[Desugar][Ty][BareTrait][{:?}] {:?}\n{:#?}", ty.span, res, ty);

            let trait_obj = TyKind::TraitObject(vec![], TraitObjectSyntax::None);
            ty.kind = trait_obj;

            debug!("->\n{:#?}\n", ty);
        }

        mut_visit::noop_visit_ty(ty, self);
    }
}
