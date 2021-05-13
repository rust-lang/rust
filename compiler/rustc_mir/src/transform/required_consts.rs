use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Constant, Location, Operand};
use rustc_middle::ty::ConstKind;
use rustc_span::Span;

pub struct RequiredConstsVisitor<'a, 'tcx> {
    required_consts: &'a mut Vec<(Span, Constant<'tcx>)>,
}

impl<'a, 'tcx> RequiredConstsVisitor<'a, 'tcx> {
    pub fn new(required_consts: &'a mut Vec<(Span, Constant<'tcx>)>) -> Self {
        RequiredConstsVisitor { required_consts }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for RequiredConstsVisitor<'a, 'tcx> {
    fn visit_operand(&mut self,
        operand: &Operand<'tcx>,
        _: Location) {
        if let Operand::Constant(box(span, constant)) = operand {
            if let Some(ct) = constant.literal.const_for_ty() {
                if let ConstKind::Unevaluated(_) = ct.val {
                    self.required_consts.push((*span, *constant));
                }
            }
        }
    }
}
