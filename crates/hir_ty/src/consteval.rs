//! Constant evaluation details

use std::{collections::HashMap, convert::TryInto, fmt::Display};

use chalk_ir::{IntTy, Scalar};
use hir_def::{
    expr::{ArithOp, BinaryOp, Expr, Literal, Pat},
    type_ref::ConstScalar,
};
use hir_expand::name::Name;
use la_arena::{Arena, Idx};

use crate::{Const, ConstData, ConstValue, Interner, Ty, TyKind};

/// Extension trait for [`Const`]
pub trait ConstExt {
    /// Is a [`Const`] unknown?
    fn is_unknown(&self) -> bool;
}

impl ConstExt for Const {
    fn is_unknown(&self) -> bool {
        match self.data(Interner).value {
            // interned Unknown
            chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst {
                interned: ConstScalar::Unknown,
            }) => true,

            // interned concrete anything else
            chalk_ir::ConstValue::Concrete(..) => false,

            _ => {
                tracing::error!(
                    "is_unknown was called on a non-concrete constant value! {:?}",
                    self
                );
                true
            }
        }
    }
}

pub struct ConstEvalCtx<'a> {
    pub exprs: &'a Arena<Expr>,
    pub pats: &'a Arena<Pat>,
    pub local_data: HashMap<Name, ComputedExpr>,
    pub infer: &'a mut dyn FnMut(Idx<Expr>) -> Ty,
}

#[derive(Debug, Clone)]
pub enum ConstEvalError {
    NotSupported(&'static str),
    TypeError,
    IncompleteExpr,
    Panic(String),
}

#[derive(Debug, Clone)]
pub enum ComputedExpr {
    Literal(Literal),
    Tuple(Box<[ComputedExpr]>),
}

impl Display for ComputedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputedExpr::Literal(l) => match l {
                Literal::Int(x, _) => {
                    if *x >= 16 {
                        write!(f, "{} ({:#X})", x, x)
                    } else {
                        write!(f, "{}", x)
                    }
                }
                Literal::Uint(x, _) => {
                    if *x >= 16 {
                        write!(f, "{} ({:#X})", x, x)
                    } else {
                        write!(f, "{}", x)
                    }
                }
                Literal::Float(x, _) => write!(f, "{}", x),
                Literal::Bool(x) => write!(f, "{}", x),
                Literal::Char(x) => write!(f, "{:?}", x),
                Literal::String(x) => write!(f, "{:?}", x),
                Literal::ByteString(x) => write!(f, "{:?}", x),
            },
            ComputedExpr::Tuple(t) => {
                write!(f, "(")?;
                for x in &**t {
                    write!(f, "{}, ", x)?;
                }
                write!(f, ")")
            }
        }
    }
}

fn scalar_max(scalar: &Scalar) -> i128 {
    match scalar {
        Scalar::Bool => 1,
        Scalar::Char => u32::MAX as i128,
        Scalar::Int(x) => match x {
            IntTy::Isize => isize::MAX as i128,
            IntTy::I8 => i8::MAX as i128,
            IntTy::I16 => i16::MAX as i128,
            IntTy::I32 => i32::MAX as i128,
            IntTy::I64 => i64::MAX as i128,
            IntTy::I128 => i128::MAX as i128,
        },
        Scalar::Uint(x) => match x {
            chalk_ir::UintTy::Usize => usize::MAX as i128,
            chalk_ir::UintTy::U8 => u8::MAX as i128,
            chalk_ir::UintTy::U16 => u16::MAX as i128,
            chalk_ir::UintTy::U32 => u32::MAX as i128,
            chalk_ir::UintTy::U64 => u64::MAX as i128,
            chalk_ir::UintTy::U128 => i128::MAX as i128, // ignore too big u128 for now
        },
        Scalar::Float(_) => 0,
    }
}

fn is_valid(scalar: &Scalar, value: i128) -> bool {
    if value < 0 {
        !matches!(scalar, Scalar::Uint(_)) && -scalar_max(scalar) - 1 <= value
    } else {
        value <= scalar_max(scalar)
    }
}

pub fn eval_const(expr: &Expr, ctx: &mut ConstEvalCtx<'_>) -> Result<ComputedExpr, ConstEvalError> {
    match expr {
        Expr::Literal(l) => Ok(ComputedExpr::Literal(l.clone())),
        &Expr::UnaryOp { expr, op } => {
            let ty = &(ctx.infer)(expr);
            let ev = eval_const(&ctx.exprs[expr], ctx)?;
            match op {
                hir_def::expr::UnaryOp::Deref => Err(ConstEvalError::NotSupported("deref")),
                hir_def::expr::UnaryOp::Not => {
                    let v = match ev {
                        ComputedExpr::Literal(Literal::Bool(b)) => {
                            return Ok(ComputedExpr::Literal(Literal::Bool(!b)))
                        }
                        ComputedExpr::Literal(Literal::Int(v, _)) => v,
                        ComputedExpr::Literal(Literal::Uint(v, _)) => v
                            .try_into()
                            .map_err(|_| ConstEvalError::NotSupported("too big u128"))?,
                        _ => return Err(ConstEvalError::NotSupported("this kind of operator")),
                    };
                    let r = match ty.kind(Interner) {
                        TyKind::Scalar(Scalar::Uint(x)) => match x {
                            chalk_ir::UintTy::U8 => !(v as u8) as i128,
                            chalk_ir::UintTy::U16 => !(v as u16) as i128,
                            chalk_ir::UintTy::U32 => !(v as u32) as i128,
                            chalk_ir::UintTy::U64 => !(v as u64) as i128,
                            chalk_ir::UintTy::U128 => {
                                return Err(ConstEvalError::NotSupported("negation of u128"))
                            }
                            chalk_ir::UintTy::Usize => !(v as usize) as i128,
                        },
                        TyKind::Scalar(Scalar::Int(x)) => match x {
                            chalk_ir::IntTy::I8 => !(v as i8) as i128,
                            chalk_ir::IntTy::I16 => !(v as i16) as i128,
                            chalk_ir::IntTy::I32 => !(v as i32) as i128,
                            chalk_ir::IntTy::I64 => !(v as i64) as i128,
                            chalk_ir::IntTy::I128 => !v,
                            chalk_ir::IntTy::Isize => !(v as isize) as i128,
                        },
                        _ => return Err(ConstEvalError::NotSupported("unreachable?")),
                    };
                    Ok(ComputedExpr::Literal(Literal::Int(r, None)))
                }
                hir_def::expr::UnaryOp::Neg => {
                    let v = match ev {
                        ComputedExpr::Literal(Literal::Int(v, _)) => v,
                        ComputedExpr::Literal(Literal::Uint(v, _)) => v
                            .try_into()
                            .map_err(|_| ConstEvalError::NotSupported("too big u128"))?,
                        _ => return Err(ConstEvalError::NotSupported("this kind of operator")),
                    };
                    Ok(ComputedExpr::Literal(Literal::Int(
                        v.checked_neg().ok_or_else(|| {
                            ConstEvalError::Panic("overflow in negation".to_string())
                        })?,
                        None,
                    )))
                }
            }
        }
        &Expr::BinaryOp { lhs, rhs, op } => {
            let ty = &(ctx.infer)(lhs);
            let lhs = eval_const(&ctx.exprs[lhs], ctx)?;
            let rhs = eval_const(&ctx.exprs[rhs], ctx)?;
            let op = op.ok_or(ConstEvalError::IncompleteExpr)?;
            let v1 = match lhs {
                ComputedExpr::Literal(Literal::Int(v, _)) => v,
                ComputedExpr::Literal(Literal::Uint(v, _)) => {
                    v.try_into().map_err(|_| ConstEvalError::NotSupported("too big u128"))?
                }
                _ => return Err(ConstEvalError::NotSupported("this kind of operator")),
            };
            let v2 = match rhs {
                ComputedExpr::Literal(Literal::Int(v, _)) => v,
                ComputedExpr::Literal(Literal::Uint(v, _)) => {
                    v.try_into().map_err(|_| ConstEvalError::NotSupported("too big u128"))?
                }
                _ => return Err(ConstEvalError::NotSupported("this kind of operator")),
            };
            match op {
                BinaryOp::ArithOp(b) => {
                    let panic_arith = ConstEvalError::Panic(
                        "attempt to run invalid arithmetic operation".to_string(),
                    );
                    let r = match b {
                        ArithOp::Add => v1.checked_add(v2).ok_or_else(|| panic_arith.clone())?,
                        ArithOp::Mul => v1.checked_mul(v2).ok_or_else(|| panic_arith.clone())?,
                        ArithOp::Sub => v1.checked_sub(v2).ok_or_else(|| panic_arith.clone())?,
                        ArithOp::Div => v1.checked_div(v2).ok_or_else(|| panic_arith.clone())?,
                        ArithOp::Rem => v1.checked_rem(v2).ok_or_else(|| panic_arith.clone())?,
                        ArithOp::Shl => v1
                            .checked_shl(v2.try_into().map_err(|_| panic_arith.clone())?)
                            .ok_or_else(|| panic_arith.clone())?,
                        ArithOp::Shr => v1
                            .checked_shr(v2.try_into().map_err(|_| panic_arith.clone())?)
                            .ok_or_else(|| panic_arith.clone())?,
                        ArithOp::BitXor => v1 ^ v2,
                        ArithOp::BitOr => v1 | v2,
                        ArithOp::BitAnd => v1 & v2,
                    };
                    if let TyKind::Scalar(s) = ty.kind(Interner) {
                        if !is_valid(s, r) {
                            return Err(panic_arith);
                        }
                    }
                    Ok(ComputedExpr::Literal(Literal::Int(r, None)))
                }
                BinaryOp::LogicOp(_) => Err(ConstEvalError::TypeError),
                _ => Err(ConstEvalError::NotSupported("bin op on this operators")),
            }
        }
        Expr::Block { statements, tail, .. } => {
            let mut prev_values = HashMap::<Name, Option<ComputedExpr>>::default();
            for statement in &**statements {
                match *statement {
                    hir_def::expr::Statement::Let { pat, initializer, .. } => {
                        let pat = &ctx.pats[pat];
                        let name = match pat {
                            Pat::Bind { name, subpat, .. } if subpat.is_none() => name.clone(),
                            _ => {
                                return Err(ConstEvalError::NotSupported("complex patterns in let"))
                            }
                        };
                        let value = match initializer {
                            Some(x) => eval_const(&ctx.exprs[x], ctx)?,
                            None => continue,
                        };
                        if !prev_values.contains_key(&name) {
                            let prev = ctx.local_data.insert(name.clone(), value);
                            prev_values.insert(name, prev);
                        } else {
                            ctx.local_data.insert(name, value);
                        }
                    }
                    hir_def::expr::Statement::Expr { .. } => {
                        return Err(ConstEvalError::NotSupported("this kind of statement"))
                    }
                }
            }
            let r = match tail {
                &Some(x) => eval_const(&ctx.exprs[x], ctx),
                None => Ok(ComputedExpr::Tuple(Box::new([]))),
            };
            // clean up local data, so caller will receive the exact map that passed to us
            for (name, val) in prev_values {
                match val {
                    Some(x) => ctx.local_data.insert(name, x),
                    None => ctx.local_data.remove(&name),
                };
            }
            r
        }
        Expr::Path(p) => {
            let name = p.mod_path().as_ident().ok_or(ConstEvalError::NotSupported("big paths"))?;
            let r = ctx
                .local_data
                .get(name)
                .ok_or(ConstEvalError::NotSupported("Non local name resolution"))?;
            Ok(r.clone())
        }
        _ => Err(ConstEvalError::NotSupported("This kind of expression")),
    }
}

pub fn eval_usize(expr: Idx<Expr>, mut ctx: ConstEvalCtx<'_>) -> Option<u64> {
    let expr = &ctx.exprs[expr];
    if let Ok(ce) = eval_const(expr, &mut ctx) {
        match ce {
            ComputedExpr::Literal(Literal::Int(x, _)) => return x.try_into().ok(),
            ComputedExpr::Literal(Literal::Uint(x, _)) => return x.try_into().ok(),
            _ => {}
        }
    }
    None
}

/// Interns a possibly-unknown target usize
pub fn usize_const(value: Option<u64>) -> Const {
    ConstData {
        ty: TyKind::Scalar(chalk_ir::Scalar::Uint(chalk_ir::UintTy::Usize)).intern(Interner),
        value: ConstValue::Concrete(chalk_ir::ConcreteConst {
            interned: value.map(ConstScalar::Usize).unwrap_or(ConstScalar::Unknown),
        }),
    }
    .intern(Interner)
}
