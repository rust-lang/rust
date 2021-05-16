//! Constant evaluation details

use std::convert::TryInto;

use hir_def::{
    builtin_type::BuiltinUint,
    expr::{Expr, Literal},
    type_ref::ConstScalar,
};

use crate::{Const, ConstData, ConstValue, Interner, TyKind};

/// Extension trait for [`Const`]
pub trait ConstExt {
    /// Is a [`Const`] unknown?
    fn is_unknown(&self) -> bool;
}

impl ConstExt for Const {
    fn is_unknown(&self) -> bool {
        match self.data(&Interner).value {
            // interned Unknown
            chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst {
                interned: ConstScalar::Unknown,
            }) => true,

            // interned concrete anything else
            chalk_ir::ConstValue::Concrete(..) => false,

            _ => {
                log::error!("is_unknown was called on a non-concrete constant value! {:?}", self);
                true
            }
        }
    }
}

// FIXME: support more than just evaluating literals
pub fn eval_usize(expr: &Expr) -> Option<u64> {
    match expr {
        Expr::Literal(Literal::Uint(v, None))
        | Expr::Literal(Literal::Uint(v, Some(BuiltinUint::Usize))) => (*v).try_into().ok(),
        _ => None,
    }
}

/// Interns a possibly-unknown target usize
pub fn usize_const(value: Option<u64>) -> Const {
    ConstData {
        ty: TyKind::Scalar(chalk_ir::Scalar::Uint(chalk_ir::UintTy::Usize)).intern(&Interner),
        value: ConstValue::Concrete(chalk_ir::ConcreteConst {
            interned: value.map(|value| ConstScalar::Usize(value)).unwrap_or(ConstScalar::Unknown),
        }),
    }
    .intern(&Interner)
}
