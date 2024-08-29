use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{traversal, Body, ConstOperand, Location};

pub struct RequiredConstsVisitor<'a, 'tcx> {
    required_consts: &'a mut Vec<ConstOperand<'tcx>>,
}

impl<'a, 'tcx> RequiredConstsVisitor<'a, 'tcx> {
    fn new(required_consts: &'a mut Vec<ConstOperand<'tcx>>) -> Self {
        RequiredConstsVisitor { required_consts }
    }

    pub fn compute_required_consts(body: &mut Body<'tcx>) {
        let mut required_consts = Vec::new();
        let mut required_consts_visitor = RequiredConstsVisitor::new(&mut required_consts);
        for (bb, bb_data) in traversal::reverse_postorder(&body) {
            required_consts_visitor.visit_basic_block_data(bb, bb_data);
        }
        body.set_required_consts(required_consts);
    }
}

impl<'tcx> Visitor<'tcx> for RequiredConstsVisitor<'_, 'tcx> {
    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, _: Location) {
        if constant.const_.is_required_const() {
            self.required_consts.push(*constant);
        }
    }
}
