use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{ConstOperand, Location};

pub struct RequiredConstsVisitor<'a, 'tcx> {
    required_consts: &'a mut Vec<ConstOperand<'tcx>>,
}

impl<'a, 'tcx> RequiredConstsVisitor<'a, 'tcx> {
    pub fn new(required_consts: &'a mut Vec<ConstOperand<'tcx>>) -> Self {
        RequiredConstsVisitor { required_consts }
    }
}

impl<'tcx> Visitor<'tcx> for RequiredConstsVisitor<'_, 'tcx> {
    fn visit_constant(&mut self, constant: &ConstOperand<'tcx>, _: Location) {
        if constant.const_.is_required_const() {
            self.required_consts.push(*constant);
        }
    }
}
