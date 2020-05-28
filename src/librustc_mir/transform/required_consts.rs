use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Constant, Location};
use rustc_middle::ty::ConstKind;

pub struct RequiredConstsVisitor<'a, 'tcx> {
    required_consts: &'a mut Vec<Constant<'tcx>>,
}

impl<'a, 'tcx> RequiredConstsVisitor<'a, 'tcx> {
    pub fn new(required_consts: &'a mut Vec<Constant<'tcx>>) -> Self {
        RequiredConstsVisitor { required_consts }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for RequiredConstsVisitor<'a, 'tcx> {
    fn visit_constant(&mut self, constant: &Constant<'tcx>, _: Location) {
        let const_kind = constant.literal.val;

        if let ConstKind::Unevaluated(_, _, _) = const_kind {
            self.required_consts.push(*constant);
        }
    }
}
