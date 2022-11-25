//! Constant evaluation details

use std::{
    collections::HashMap,
    fmt::{Display, Write},
};

use chalk_ir::{BoundVar, DebruijnIndex, GenericArgData, IntTy, Scalar};
use hir_def::{
    expr::{ArithOp, BinaryOp, Expr, ExprId, Literal, Pat, PatId},
    path::ModPath,
    resolver::{resolver_for_expr, ResolveValueResult, Resolver, ValueNs},
    type_ref::ConstScalar,
    ConstId, DefWithBodyId,
};
use la_arena::{Arena, Idx};
use stdx::never;

use crate::{
    db::HirDatabase, infer::InferenceContext, lower::ParamLoweringMode, to_placeholder_idx,
    utils::Generics, Const, ConstData, ConstValue, GenericArg, InferenceResult, Interner, Ty,
    TyBuilder, TyKind,
};

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
    pub db: &'a dyn HirDatabase,
    pub owner: DefWithBodyId,
    pub exprs: &'a Arena<Expr>,
    pub pats: &'a Arena<Pat>,
    pub local_data: HashMap<PatId, ComputedExpr>,
    infer: &'a InferenceResult,
}

impl ConstEvalCtx<'_> {
    fn expr_ty(&mut self, expr: ExprId) -> Ty {
        self.infer[expr].clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstEvalError {
    NotSupported(&'static str),
    SemanticError(&'static str),
    Loop,
    IncompleteExpr,
    Panic(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputedExpr {
    Literal(Literal),
    Tuple(Box<[ComputedExpr]>),
}

impl Display for ComputedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputedExpr::Literal(l) => match l {
                Literal::Int(x, _) => {
                    if *x >= 10 {
                        write!(f, "{} ({:#X})", x, x)
                    } else {
                        x.fmt(f)
                    }
                }
                Literal::Uint(x, _) => {
                    if *x >= 10 {
                        write!(f, "{} ({:#X})", x, x)
                    } else {
                        x.fmt(f)
                    }
                }
                Literal::Float(x, _) => x.fmt(f),
                Literal::Bool(x) => x.fmt(f),
                Literal::Char(x) => std::fmt::Debug::fmt(x, f),
                Literal::String(x) => std::fmt::Debug::fmt(x, f),
                Literal::ByteString(x) => std::fmt::Debug::fmt(x, f),
            },
            ComputedExpr::Tuple(t) => {
                f.write_char('(')?;
                for x in &**t {
                    x.fmt(f)?;
                    f.write_str(", ")?;
                }
                f.write_char(')')
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

pub fn eval_const(
    expr_id: ExprId,
    ctx: &mut ConstEvalCtx<'_>,
) -> Result<ComputedExpr, ConstEvalError> {
    let expr = &ctx.exprs[expr_id];
    match expr {
        Expr::Missing => Err(ConstEvalError::IncompleteExpr),
        Expr::Literal(l) => Ok(ComputedExpr::Literal(l.clone())),
        &Expr::UnaryOp { expr, op } => {
            let ty = &ctx.expr_ty(expr);
            let ev = eval_const(expr, ctx)?;
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
            let ty = &ctx.expr_ty(lhs);
            let lhs = eval_const(lhs, ctx)?;
            let rhs = eval_const(rhs, ctx)?;
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
                BinaryOp::LogicOp(_) => Err(ConstEvalError::SemanticError("logic op on numbers")),
                _ => Err(ConstEvalError::NotSupported("bin op on this operators")),
            }
        }
        Expr::Block { statements, tail, .. } => {
            let mut prev_values = HashMap::<PatId, Option<ComputedExpr>>::default();
            for statement in &**statements {
                match *statement {
                    hir_def::expr::Statement::Let { pat: pat_id, initializer, .. } => {
                        let pat = &ctx.pats[pat_id];
                        match pat {
                            Pat::Bind { subpat, .. } if subpat.is_none() => (),
                            _ => {
                                return Err(ConstEvalError::NotSupported("complex patterns in let"))
                            }
                        };
                        let value = match initializer {
                            Some(x) => eval_const(x, ctx)?,
                            None => continue,
                        };
                        if !prev_values.contains_key(&pat_id) {
                            let prev = ctx.local_data.insert(pat_id, value);
                            prev_values.insert(pat_id, prev);
                        } else {
                            ctx.local_data.insert(pat_id, value);
                        }
                    }
                    hir_def::expr::Statement::Expr { .. } => {
                        return Err(ConstEvalError::NotSupported("this kind of statement"))
                    }
                }
            }
            let r = match tail {
                &Some(x) => eval_const(x, ctx),
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
            let resolver = resolver_for_expr(ctx.db.upcast(), ctx.owner, expr_id);
            let pr = resolver
                .resolve_path_in_value_ns(ctx.db.upcast(), p.mod_path())
                .ok_or(ConstEvalError::SemanticError("unresolved path"))?;
            let pr = match pr {
                ResolveValueResult::ValueNs(v) => v,
                ResolveValueResult::Partial(..) => {
                    return match ctx
                        .infer
                        .assoc_resolutions_for_expr(expr_id)
                        .ok_or(ConstEvalError::SemanticError("unresolved assoc item"))?
                    {
                        hir_def::AssocItemId::FunctionId(_) => {
                            Err(ConstEvalError::NotSupported("assoc function"))
                        }
                        hir_def::AssocItemId::ConstId(c) => ctx.db.const_eval(c),
                        hir_def::AssocItemId::TypeAliasId(_) => {
                            Err(ConstEvalError::NotSupported("assoc type alias"))
                        }
                    }
                }
            };
            match pr {
                ValueNs::LocalBinding(pat_id) => {
                    let r = ctx
                        .local_data
                        .get(&pat_id)
                        .ok_or(ConstEvalError::NotSupported("Unexpected missing local"))?;
                    Ok(r.clone())
                }
                ValueNs::ConstId(id) => ctx.db.const_eval(id),
                ValueNs::GenericParam(_) => {
                    Err(ConstEvalError::NotSupported("const generic without substitution"))
                }
                _ => Err(ConstEvalError::NotSupported("path that are not const or local")),
            }
        }
        _ => Err(ConstEvalError::NotSupported("This kind of expression")),
    }
}

pub(crate) fn path_to_const(
    db: &dyn HirDatabase,
    resolver: &Resolver,
    path: &ModPath,
    mode: ParamLoweringMode,
    args_lazy: impl FnOnce() -> Generics,
    debruijn: DebruijnIndex,
) -> Option<Const> {
    match resolver.resolve_path_in_value_ns_fully(db.upcast(), &path) {
        Some(ValueNs::GenericParam(p)) => {
            let ty = db.const_param_ty(p);
            let args = args_lazy();
            let value = match mode {
                ParamLoweringMode::Placeholder => {
                    ConstValue::Placeholder(to_placeholder_idx(db, p.into()))
                }
                ParamLoweringMode::Variable => match args.param_idx(p.into()) {
                    Some(x) => ConstValue::BoundVar(BoundVar::new(debruijn, x)),
                    None => {
                        never!(
                            "Generic list doesn't contain this param: {:?}, {}, {:?}",
                            args,
                            path,
                            p
                        );
                        return None;
                    }
                },
            };
            Some(ConstData { ty, value }.intern(Interner))
        }
        _ => None,
    }
}

pub fn unknown_const(ty: Ty) -> Const {
    ConstData {
        ty,
        value: ConstValue::Concrete(chalk_ir::ConcreteConst { interned: ConstScalar::Unknown }),
    }
    .intern(Interner)
}

pub fn unknown_const_as_generic(ty: Ty) -> GenericArg {
    GenericArgData::Const(unknown_const(ty)).intern(Interner)
}

/// Interns a constant scalar with the given type
pub fn intern_const_scalar(value: ConstScalar, ty: Ty) -> Const {
    ConstData { ty, value: ConstValue::Concrete(chalk_ir::ConcreteConst { interned: value }) }
        .intern(Interner)
}

/// Interns a possibly-unknown target usize
pub fn usize_const(value: Option<u128>) -> Const {
    intern_const_scalar(value.map_or(ConstScalar::Unknown, ConstScalar::UInt), TyBuilder::usize())
}

pub(crate) fn const_eval_recover(
    _: &dyn HirDatabase,
    _: &[String],
    _: &ConstId,
) -> Result<ComputedExpr, ConstEvalError> {
    Err(ConstEvalError::Loop)
}

pub(crate) fn const_eval_query(
    db: &dyn HirDatabase,
    const_id: ConstId,
) -> Result<ComputedExpr, ConstEvalError> {
    let def = const_id.into();
    let body = db.body(def);
    let infer = &db.infer(def);
    let result = eval_const(
        body.body_expr,
        &mut ConstEvalCtx {
            db,
            owner: const_id.into(),
            exprs: &body.exprs,
            pats: &body.pats,
            local_data: HashMap::default(),
            infer,
        },
    );
    result
}

pub(crate) fn eval_to_const<'a>(
    expr: Idx<Expr>,
    mode: ParamLoweringMode,
    ctx: &mut InferenceContext<'a>,
    args: impl FnOnce() -> Generics,
    debruijn: DebruijnIndex,
) -> Const {
    if let Expr::Path(p) = &ctx.body.exprs[expr] {
        let db = ctx.db;
        let resolver = &ctx.resolver;
        if let Some(c) = path_to_const(db, resolver, p.mod_path(), mode, args, debruijn) {
            return c;
        }
    }
    let body = ctx.body.clone();
    let mut ctx = ConstEvalCtx {
        db: ctx.db,
        owner: ctx.owner,
        exprs: &body.exprs,
        pats: &body.pats,
        local_data: HashMap::default(),
        infer: &ctx.result,
    };
    let computed_expr = eval_const(expr, &mut ctx);
    let const_scalar = match computed_expr {
        Ok(ComputedExpr::Literal(literal)) => literal.into(),
        _ => ConstScalar::Unknown,
    };
    intern_const_scalar(const_scalar, TyBuilder::usize())
}

#[cfg(test)]
mod tests;
