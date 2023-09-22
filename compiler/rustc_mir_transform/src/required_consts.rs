use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Const, ConstOperand, Location};
use rustc_middle::ty::ConstKind;

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
        let const_ = constant.const_;
        match const_ {
            Const::Ty(c) => match c.kind() {
                ConstKind::Param(_) | ConstKind::Error(_) | ConstKind::Value(_) => {}
                _ => bug!("only ConstKind::Param/Value should be encountered here, got {:#?}", c),
            },
            Const::Unevaluated(..) => self.required_consts.push(*constant),
            Const::Val(..) => {}
        }
    }
}
