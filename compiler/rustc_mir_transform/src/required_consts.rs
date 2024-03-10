use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, Const, ConstOperand, Location, RequiredItem};
use rustc_middle::ty::{self, ConstKind, TyCtxt};

pub struct RequiredConstsVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    required_consts: &'a mut Vec<ConstOperand<'tcx>>,
    required_items: &'a mut Vec<RequiredItem<'tcx>>,
}

impl<'a, 'tcx> RequiredConstsVisitor<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        body: &'a mir::Body<'tcx>,
        required_consts: &'a mut Vec<ConstOperand<'tcx>>,
        required_items: &'a mut Vec<RequiredItem<'tcx>>,
    ) -> Self {
        RequiredConstsVisitor { tcx, body, required_consts, required_items }
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
                    debug!("adding to required_items: {def_id:?}");
                    self.required_items.push(RequiredItem::Fn(*def_id, args));
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
                self.required_items.push(RequiredItem::Drop(ty));
            }
            _ => {}
        }
    }
}
