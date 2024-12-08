use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Body, ConstOperand, Location, traversal};

pub(super) struct RequiredConstsVisitor<'tcx> {
    required_consts: Vec<ConstOperand<'tcx>>,
}

impl<'tcx> RequiredConstsVisitor<'tcx> {
    pub(super) fn compute_required_consts(body: &mut Body<'tcx>) {
        let mut visitor = RequiredConstsVisitor { required_consts: Vec::new() };
        for (bb, bb_data) in traversal::reverse_postorder(&body) {
            visitor.visit_basic_block_data(bb, bb_data);
        }
        body.set_required_consts(visitor.required_consts);
    }
}

impl<'tcx> Visitor<'tcx> for RequiredConstsVisitor<'tcx> {
    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, _: Location) {
        if constant.const_.is_required_const() {
            self.required_consts.push(*constant);
        }
    }
}
