use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Constant, ConstantKind, Location};
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
        match literal {
            ConstantKind::Ty(c) => match c.kind() {
                ConstKind::Unevaluated(uv) => {
                    let literal = ConstantKind::Unevaluated(uv.expand(), c.ty());
                    let new_constant =
                        Constant { span: constant.span, user_ty: constant.user_ty, literal };
                    self.required_consts.push(new_constant);
                }
                _ => {}
            },
            ConstantKind::Unevaluated(..) => self.required_consts.push(*constant),
            ConstantKind::Val(..) => {}
        }
    }
}
