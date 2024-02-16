use rustc_ast::{
    mut_visit::{self, MutVisitor},
    ptr::P,
    Path, TraitObjectSyntax, Ty, TyKind,
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
        // If the type is a path, and that path resolves to a trate, desugar it
        // into a bare trait object.

        if let TyKind::Path(None, _path) = &ty.kind
            && let Some(partial_res) = self.r.partial_res_map.get(&ty.id)
            && let Some(res @ Res::Def(DefKind::Trait | DefKind::TraitAlias, _)) =
                partial_res.full_res()
        {
            debug!("[Desugar][Ty][BareTrait][{:?}] {:?}\n{:#?}", ty.span, res, ty);

            // TODO(axelmagn): extract bounds from path?
            let bounds = vec![];
            // TODO(axelmagn): is this the right choice? I don't think a trait
            // path would ever imply Dyn, so None is the only other option.
            let syntax = TraitObjectSyntax::None;
            let trait_obj = TyKind::TraitObject(bounds, syntax);
            ty.kind = trait_obj;

            // TODO(axelmagn): Do the type tokens need to be rewritten?  I would
            // assume so, for use in AST lowering.

            debug!("->\n{:#?}\n", ty);
        }

        mut_visit::noop_visit_ty(ty, self);
    }

    fn visit_path(&mut self, path: &mut Path) {
        // TODO(axelmagn): Desugar type-relative paths during resolution.
        // Transform a::b::c::d to <a::b::c>::d when a::b::c can be resolved to
        // a type and ::d cannot (for instance because it is a trait method).
        // (check rustc_ast_lowering::path::lower_qpath for current impl)

        mut_visit::noop_visit_path(path, self);
    }
}
