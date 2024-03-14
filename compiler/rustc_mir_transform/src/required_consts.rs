use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Const, ConstOperand, Location};
use rustc_middle::ty;

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
        // Only unevaluated consts have to be added to `required_consts` as only those can possibly
        // still have latent const-eval errors.
        let is_required = match constant.const_ {
            Const::Ty(c) => match c.kind() {
                ty::ConstKind::Value(_) => false, // already a value, cannot error
                ty::ConstKind::Param(_) | ty::ConstKind::Error(_) => true, // these are errors or could be replaced by errors
                _ => bug!(
                    "only ConstKind::Param/Value/Error should be encountered here, got {:#?}",
                    c
                ),
            },
            Const::Unevaluated(..) => true,
            Const::Val(..) => false, // already a value, cannot error
        };
        if is_required {
            self.required_consts.push(*constant);
        }
    }
}
