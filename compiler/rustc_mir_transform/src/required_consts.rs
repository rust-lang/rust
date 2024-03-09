use rustc_hir::LangItem;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, Const, ConstOperand, Location};
use rustc_middle::ty::{self, ConstKind, Instance, InstanceDef, TyCtxt};

pub struct RequiredConstsVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    required_consts: &'a mut Vec<ConstOperand<'tcx>>,
    required_fns: &'a mut Vec<Instance<'tcx>>,
}

impl<'a, 'tcx> RequiredConstsVisitor<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        body: &'a mir::Body<'tcx>,
        required_consts: &'a mut Vec<ConstOperand<'tcx>>,
        required_fns: &'a mut Vec<Instance<'tcx>>,
    ) -> Self {
        RequiredConstsVisitor { tcx, body, required_consts, required_fns }
    }
}

impl<'tcx> Visitor<'tcx> for RequiredConstsVisitor<'_, 'tcx> {
    fn visit_constant(&mut self, constant: &ConstOperand<'tcx>, _: Location) {
        let const_ = constant.const_;
        match const_ {
            Const::Ty(c) => match c.kind() {
                ConstKind::Param(_) | ConstKind::Error(_) | ConstKind::Value(_) => {}
                _ => bug!("only ConstKind::Param/Value should be encountered here, got {:#?}", c),
            },
            Const::Unevaluated(..) => self.required_consts.push(*constant),
            Const::Val(_val, ty) => {
                // This is how function items get referenced: via zero-sized constants of `FnDef` type
                if let ty::FnDef(def_id, args) = ty.kind() {
                    debug!("adding to required_fns: {def_id:?}");
                    // FIXME maybe we shouldn't use `Instance`? We can't use `Instance::new`, it is
                    // for codegen. But `Instance` feels like the right representation... Check what
                    // the regular collector does.
                    self.required_fns.push(Instance { def: InstanceDef::Item(*def_id), args });
                }
            }
        }
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match terminator.kind {
            // We don't need to handle `Call` as we already handled all function type operands in
            // `visit_constant`. But we do need to handle `Drop`.
            mir::TerminatorKind::Drop { place, .. } => {
                let ty = place.ty(self.body, self.tcx).ty;
                let def_id = self.tcx.require_lang_item(LangItem::DropInPlace, None);
                let args = self.tcx.mk_args(&[ty.into()]);
                // FIXME: same as above (we cannot use `Instance::resolve_drop_in_place` as this is
                // still generic).
                self.required_fns.push(Instance { def: InstanceDef::Item(def_id), args });
            }
            _ => {}
        }
    }
}
