use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Constant, Location};
use rustc_middle::ty::ConstKind;

pub struct UnevalConstSetVisitor<'a, 'tcx> {
    uneval_consts: &'a mut Vec<Constant<'tcx>>,
}

impl<'a, 'tcx> UnevalConstSetVisitor<'a, 'tcx> {
    pub fn new(uneval_consts: &'a mut Vec<Constant<'tcx>>) -> Self {
        UnevalConstSetVisitor { uneval_consts }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UnevalConstSetVisitor<'a, 'tcx> {
    fn visit_constant(&mut self, constant: &Constant<'tcx>, _: Location) {
        let const_kind = constant.literal.val;

        if let ConstKind::Unevaluated(_, _, _) = const_kind {
            self.uneval_consts.push(*constant);
        }
    }
}
