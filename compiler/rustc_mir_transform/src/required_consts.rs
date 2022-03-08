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

impl<'tcx> Visitor<'tcx> for RequiredConstsVisitor<'_, 'tcx> {
    fn visit_constant(&mut self, constant: &Constant<'tcx>, _: Location) {
        let literal = constant.literal;
        if let Some(ct) = literal.const_for_ty() && let ConstKind::Unevaluated(_) = ct.val() {
            self.required_consts.push(*constant);
        }
    }
}
