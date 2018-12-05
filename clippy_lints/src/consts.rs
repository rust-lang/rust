// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![allow(clippy::float_cmp)]

use crate::rustc::hir::def::Def;
use crate::rustc::hir::*;
use crate::rustc::lint::LateContext;
use crate::rustc::ty::subst::{Subst, Substs};
use crate::rustc::ty::{self, Instance, Ty, TyCtxt};
use crate::rustc::{bug, span_bug};
use crate::syntax::ast::{FloatTy, LitKind};
use crate::syntax::ptr::P;
use crate::utils::{clip, sext, unsext};
use std::cmp::Ordering::{self, Equal};
use std::cmp::PartialOrd;
use std::convert::TryInto;
use std::hash::{Hash, Hasher};
use std::mem;
use std::rc::Rc;

/// A `LitKind`-like enum to fold constant `Expr`s into.
#[derive(Debug, Clone)]
pub enum Constant {
    /// a String "abc"
    Str(String),
    /// a Binary String b"abc"
    Binary(Rc<Vec<u8>>),
    /// a single char 'a'
    Char(char),
    /// an integer's bit representation
    Int(u128),
    /// an f32
    F32(f32),
    /// an f64
    F64(f64),
    /// true or false
    Bool(bool),
    /// an array of constants
    Vec(Vec<Constant>),
    /// also an array, but with only one constant, repeated N times
    Repeat(Box<Constant>, u64),
    /// a tuple of constants
    Tuple(Vec<Constant>),
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&Constant::Str(ref ls), &Constant::Str(ref rs)) => ls == rs,
            (&Constant::Binary(ref l), &Constant::Binary(ref r)) => l == r,
            (&Constant::Char(l), &Constant::Char(r)) => l == r,
            (&Constant::Int(l), &Constant::Int(r)) => l == r,
            (&Constant::F64(l), &Constant::F64(r)) => {
                // we want `Fw32 == FwAny` and `FwAny == Fw64`, by transitivity we must have
                // `Fw32 == Fw64` so don’t compare them
                // mem::transmute is required to catch non-matching 0.0, -0.0, and NaNs
                unsafe { mem::transmute::<f64, u64>(l) == mem::transmute::<f64, u64>(r) }
            },
            (&Constant::F32(l), &Constant::F32(r)) => {
                // we want `Fw32 == FwAny` and `FwAny == Fw64`, by transitivity we must have
                // `Fw32 == Fw64` so don’t compare them
                // mem::transmute is required to catch non-matching 0.0, -0.0, and NaNs
                unsafe { mem::transmute::<f64, u64>(f64::from(l)) == mem::transmute::<f64, u64>(f64::from(r)) }
            },
            (&Constant::Bool(l), &Constant::Bool(r)) => l == r,
            (&Constant::Vec(ref l), &Constant::Vec(ref r)) | (&Constant::Tuple(ref l), &Constant::Tuple(ref r)) => {
                l == r
            },
            (&Constant::Repeat(ref lv, ref ls), &Constant::Repeat(ref rv, ref rs)) => ls == rs && lv == rv,
            _ => false, // TODO: Are there inter-type equalities?
        }
    }
}

impl Hash for Constant {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match *self {
            Constant::Str(ref s) => {
                s.hash(state);
            },
            Constant::Binary(ref b) => {
                b.hash(state);
            },
            Constant::Char(c) => {
                c.hash(state);
            },
            Constant::Int(i) => {
                i.hash(state);
            },
            Constant::F32(f) => {
                unsafe { mem::transmute::<f64, u64>(f64::from(f)) }.hash(state);
            },
            Constant::F64(f) => {
                unsafe { mem::transmute::<f64, u64>(f) }.hash(state);
            },
            Constant::Bool(b) => {
                b.hash(state);
            },
            Constant::Vec(ref v) | Constant::Tuple(ref v) => {
                v.hash(state);
            },
            Constant::Repeat(ref c, l) => {
                c.hash(state);
                l.hash(state);
            },
        }
    }
}

impl Constant {
    pub fn partial_cmp(tcx: TyCtxt<'_, '_, '_>, cmp_type: ty::Ty<'_>, left: &Self, right: &Self) -> Option<Ordering> {
        match (left, right) {
            (&Constant::Str(ref ls), &Constant::Str(ref rs)) => Some(ls.cmp(rs)),
            (&Constant::Char(ref l), &Constant::Char(ref r)) => Some(l.cmp(r)),
            (&Constant::Int(l), &Constant::Int(r)) => {
                if let ty::Int(int_ty) = cmp_type.sty {
                    Some(sext(tcx, l, int_ty).cmp(&sext(tcx, r, int_ty)))
                } else {
                    Some(l.cmp(&r))
                }
            },
            (&Constant::F64(l), &Constant::F64(r)) => l.partial_cmp(&r),
            (&Constant::F32(l), &Constant::F32(r)) => l.partial_cmp(&r),
            (&Constant::Bool(ref l), &Constant::Bool(ref r)) => Some(l.cmp(r)),
            (&Constant::Tuple(ref l), &Constant::Tuple(ref r)) | (&Constant::Vec(ref l), &Constant::Vec(ref r)) => l
                .iter()
                .zip(r.iter())
                .map(|(li, ri)| Self::partial_cmp(tcx, cmp_type, li, ri))
                .find(|r| r.map_or(true, |o| o != Ordering::Equal))
                .unwrap_or_else(|| Some(l.len().cmp(&r.len()))),
            (&Constant::Repeat(ref lv, ref ls), &Constant::Repeat(ref rv, ref rs)) => {
                match Self::partial_cmp(tcx, cmp_type, lv, rv) {
                    Some(Equal) => Some(ls.cmp(rs)),
                    x => x,
                }
            },
            _ => None, // TODO: Are there any useful inter-type orderings?
        }
    }
}

/// parse a `LitKind` to a `Constant`
pub fn lit_to_constant<'tcx>(lit: &LitKind, ty: Ty<'tcx>) -> Constant {
    use crate::syntax::ast::*;

    match *lit {
        LitKind::Str(ref is, _) => Constant::Str(is.to_string()),
        LitKind::Byte(b) => Constant::Int(u128::from(b)),
        LitKind::ByteStr(ref s) => Constant::Binary(Rc::clone(s)),
        LitKind::Char(c) => Constant::Char(c),
        LitKind::Int(n, _) => Constant::Int(n),
        LitKind::Float(ref is, _) | LitKind::FloatUnsuffixed(ref is) => match ty.sty {
            ty::Float(FloatTy::F32) => Constant::F32(is.as_str().parse().unwrap()),
            ty::Float(FloatTy::F64) => Constant::F64(is.as_str().parse().unwrap()),
            _ => bug!(),
        },
        LitKind::Bool(b) => Constant::Bool(b),
    }
}

pub fn constant<'c, 'cc>(
    lcx: &LateContext<'c, 'cc>,
    tables: &'c ty::TypeckTables<'cc>,
    e: &Expr,
) -> Option<(Constant, bool)> {
    let mut cx = ConstEvalLateContext {
        tcx: lcx.tcx,
        tables,
        param_env: lcx.param_env,
        needed_resolution: false,
        substs: lcx.tcx.intern_substs(&[]),
    };
    cx.expr(e).map(|cst| (cst, cx.needed_resolution))
}

pub fn constant_simple<'c, 'cc>(
    lcx: &LateContext<'c, 'cc>,
    tables: &'c ty::TypeckTables<'cc>,
    e: &Expr,
) -> Option<Constant> {
    constant(lcx, tables, e).and_then(|(cst, res)| if res { None } else { Some(cst) })
}

/// Creates a `ConstEvalLateContext` from the given `LateContext` and `TypeckTables`
pub fn constant_context<'c, 'cc>(
    lcx: &LateContext<'c, 'cc>,
    tables: &'c ty::TypeckTables<'cc>,
) -> ConstEvalLateContext<'c, 'cc> {
    ConstEvalLateContext {
        tcx: lcx.tcx,
        tables,
        param_env: lcx.param_env,
        needed_resolution: false,
        substs: lcx.tcx.intern_substs(&[]),
    }
}

pub struct ConstEvalLateContext<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    needed_resolution: bool,
    substs: &'tcx Substs<'tcx>,
}

impl<'c, 'cc> ConstEvalLateContext<'c, 'cc> {
    /// simple constant folding: Insert an expression, get a constant or none.
    pub fn expr(&mut self, e: &Expr) -> Option<Constant> {
        match e.node {
            ExprKind::Path(ref qpath) => self.fetch_path(qpath, e.hir_id),
            ExprKind::Block(ref block, _) => self.block(block),
            ExprKind::If(ref cond, ref then, ref otherwise) => self.ifthenelse(cond, then, otherwise),
            ExprKind::Lit(ref lit) => Some(lit_to_constant(&lit.node, self.tables.expr_ty(e))),
            ExprKind::Array(ref vec) => self.multi(vec).map(Constant::Vec),
            ExprKind::Tup(ref tup) => self.multi(tup).map(Constant::Tuple),
            ExprKind::Repeat(ref value, _) => {
                let n = match self.tables.expr_ty(e).sty {
                    ty::Array(_, n) => n.assert_usize(self.tcx).expect("array length"),
                    _ => span_bug!(e.span, "typeck error"),
                };
                self.expr(value).map(|v| Constant::Repeat(Box::new(v), n))
            },
            ExprKind::Unary(op, ref operand) => self.expr(operand).and_then(|o| match op {
                UnNot => self.constant_not(&o, self.tables.expr_ty(e)),
                UnNeg => self.constant_negate(&o, self.tables.expr_ty(e)),
                UnDeref => Some(o),
            }),
            ExprKind::Binary(op, ref left, ref right) => self.binop(op, left, right),
            // TODO: add other expressions
            _ => None,
        }
    }

    #[allow(clippy::cast_possible_wrap)]
    fn constant_not(&self, o: &Constant, ty: ty::Ty<'_>) -> Option<Constant> {
        use self::Constant::*;
        match *o {
            Bool(b) => Some(Bool(!b)),
            Int(value) => {
                let value = !value;
                match ty.sty {
                    ty::Int(ity) => Some(Int(unsext(self.tcx, value as i128, ity))),
                    ty::Uint(ity) => Some(Int(clip(self.tcx, value, ity))),
                    _ => None,
                }
            },
            _ => None,
        }
    }

    fn constant_negate(&self, o: &Constant, ty: ty::Ty<'_>) -> Option<Constant> {
        use self::Constant::*;
        match *o {
            Int(value) => {
                let ity = match ty.sty {
                    ty::Int(ity) => ity,
                    _ => return None,
                };
                // sign extend
                let value = sext(self.tcx, value, ity);
                let value = value.checked_neg()?;
                // clear unused bits
                Some(Int(unsext(self.tcx, value, ity)))
            },
            F32(f) => Some(F32(-f)),
            F64(f) => Some(F64(-f)),
            _ => None,
        }
    }

    /// create `Some(Vec![..])` of all constants, unless there is any
    /// non-constant part
    fn multi(&mut self, vec: &[Expr]) -> Option<Vec<Constant>> {
        vec.iter().map(|elem| self.expr(elem)).collect::<Option<_>>()
    }

    /// lookup a possibly constant expression from a ExprKind::Path
    fn fetch_path(&mut self, qpath: &QPath, id: HirId) -> Option<Constant> {
        use crate::rustc::mir::interpret::GlobalId;

        let def = self.tables.qpath_def(qpath, id);
        match def {
            Def::Const(def_id) | Def::AssociatedConst(def_id) => {
                let substs = self.tables.node_substs(id);
                let substs = if self.substs.is_empty() {
                    substs
                } else {
                    substs.subst(self.tcx, self.substs)
                };
                let instance = Instance::resolve(self.tcx, self.param_env, def_id, substs)?;
                let gid = GlobalId {
                    instance,
                    promoted: None,
                };

                let result = self.tcx.const_eval(self.param_env.and(gid)).ok()?;
                let ret = miri_to_const(self.tcx, result);
                if ret.is_some() {
                    self.needed_resolution = true;
                }
                return ret;
            },
            _ => {},
        }
        None
    }

    /// A block can only yield a constant if it only has one constant expression
    fn block(&mut self, block: &Block) -> Option<Constant> {
        if block.stmts.is_empty() {
            block.expr.as_ref().and_then(|b| self.expr(b))
        } else {
            None
        }
    }

    fn ifthenelse(&mut self, cond: &Expr, then: &P<Expr>, otherwise: &Option<P<Expr>>) -> Option<Constant> {
        if let Some(Constant::Bool(b)) = self.expr(cond) {
            if b {
                self.expr(&**then)
            } else {
                otherwise.as_ref().and_then(|expr| self.expr(expr))
            }
        } else {
            None
        }
    }

    fn binop(&mut self, op: BinOp, left: &Expr, right: &Expr) -> Option<Constant> {
        let l = self.expr(left)?;
        let r = self.expr(right);
        match (l, r) {
            (Constant::Int(l), Some(Constant::Int(r))) => match self.tables.expr_ty(left).sty {
                ty::Int(ity) => {
                    let l = sext(self.tcx, l, ity);
                    let r = sext(self.tcx, r, ity);
                    let zext = |n: i128| Constant::Int(unsext(self.tcx, n, ity));
                    match op.node {
                        BinOpKind::Add => l.checked_add(r).map(zext),
                        BinOpKind::Sub => l.checked_sub(r).map(zext),
                        BinOpKind::Mul => l.checked_mul(r).map(zext),
                        BinOpKind::Div if r != 0 => l.checked_div(r).map(zext),
                        BinOpKind::Rem if r != 0 => l.checked_rem(r).map(zext),
                        BinOpKind::Shr => l.checked_shr(r.try_into().expect("invalid shift")).map(zext),
                        BinOpKind::Shl => l.checked_shl(r.try_into().expect("invalid shift")).map(zext),
                        BinOpKind::BitXor => Some(zext(l ^ r)),
                        BinOpKind::BitOr => Some(zext(l | r)),
                        BinOpKind::BitAnd => Some(zext(l & r)),
                        BinOpKind::Eq => Some(Constant::Bool(l == r)),
                        BinOpKind::Ne => Some(Constant::Bool(l != r)),
                        BinOpKind::Lt => Some(Constant::Bool(l < r)),
                        BinOpKind::Le => Some(Constant::Bool(l <= r)),
                        BinOpKind::Ge => Some(Constant::Bool(l >= r)),
                        BinOpKind::Gt => Some(Constant::Bool(l > r)),
                        _ => None,
                    }
                },
                ty::Uint(_) => match op.node {
                    BinOpKind::Add => l.checked_add(r).map(Constant::Int),
                    BinOpKind::Sub => l.checked_sub(r).map(Constant::Int),
                    BinOpKind::Mul => l.checked_mul(r).map(Constant::Int),
                    BinOpKind::Div => l.checked_div(r).map(Constant::Int),
                    BinOpKind::Rem => l.checked_rem(r).map(Constant::Int),
                    BinOpKind::Shr => l.checked_shr(r.try_into().expect("shift too large")).map(Constant::Int),
                    BinOpKind::Shl => l.checked_shl(r.try_into().expect("shift too large")).map(Constant::Int),
                    BinOpKind::BitXor => Some(Constant::Int(l ^ r)),
                    BinOpKind::BitOr => Some(Constant::Int(l | r)),
                    BinOpKind::BitAnd => Some(Constant::Int(l & r)),
                    BinOpKind::Eq => Some(Constant::Bool(l == r)),
                    BinOpKind::Ne => Some(Constant::Bool(l != r)),
                    BinOpKind::Lt => Some(Constant::Bool(l < r)),
                    BinOpKind::Le => Some(Constant::Bool(l <= r)),
                    BinOpKind::Ge => Some(Constant::Bool(l >= r)),
                    BinOpKind::Gt => Some(Constant::Bool(l > r)),
                    _ => None,
                },
                _ => None,
            },
            (Constant::F32(l), Some(Constant::F32(r))) => match op.node {
                BinOpKind::Add => Some(Constant::F32(l + r)),
                BinOpKind::Sub => Some(Constant::F32(l - r)),
                BinOpKind::Mul => Some(Constant::F32(l * r)),
                BinOpKind::Div => Some(Constant::F32(l / r)),
                BinOpKind::Rem => Some(Constant::F32(l % r)),
                BinOpKind::Eq => Some(Constant::Bool(l == r)),
                BinOpKind::Ne => Some(Constant::Bool(l != r)),
                BinOpKind::Lt => Some(Constant::Bool(l < r)),
                BinOpKind::Le => Some(Constant::Bool(l <= r)),
                BinOpKind::Ge => Some(Constant::Bool(l >= r)),
                BinOpKind::Gt => Some(Constant::Bool(l > r)),
                _ => None,
            },
            (Constant::F64(l), Some(Constant::F64(r))) => match op.node {
                BinOpKind::Add => Some(Constant::F64(l + r)),
                BinOpKind::Sub => Some(Constant::F64(l - r)),
                BinOpKind::Mul => Some(Constant::F64(l * r)),
                BinOpKind::Div => Some(Constant::F64(l / r)),
                BinOpKind::Rem => Some(Constant::F64(l % r)),
                BinOpKind::Eq => Some(Constant::Bool(l == r)),
                BinOpKind::Ne => Some(Constant::Bool(l != r)),
                BinOpKind::Lt => Some(Constant::Bool(l < r)),
                BinOpKind::Le => Some(Constant::Bool(l <= r)),
                BinOpKind::Ge => Some(Constant::Bool(l >= r)),
                BinOpKind::Gt => Some(Constant::Bool(l > r)),
                _ => None,
            },
            (l, r) => match (op.node, l, r) {
                (BinOpKind::And, Constant::Bool(false), _) => Some(Constant::Bool(false)),
                (BinOpKind::Or, Constant::Bool(true), _) => Some(Constant::Bool(true)),
                (BinOpKind::And, Constant::Bool(true), Some(r)) | (BinOpKind::Or, Constant::Bool(false), Some(r)) => {
                    Some(r)
                },
                (BinOpKind::BitXor, Constant::Bool(l), Some(Constant::Bool(r))) => Some(Constant::Bool(l ^ r)),
                (BinOpKind::BitAnd, Constant::Bool(l), Some(Constant::Bool(r))) => Some(Constant::Bool(l & r)),
                (BinOpKind::BitOr, Constant::Bool(l), Some(Constant::Bool(r))) => Some(Constant::Bool(l | r)),
                _ => None,
            },
        }
    }
}

pub fn miri_to_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, result: &ty::Const<'tcx>) -> Option<Constant> {
    use crate::rustc::mir::interpret::{ConstValue, Scalar};
    match result.val {
        ConstValue::Scalar(Scalar::Bits { bits: b, .. }) => match result.ty.sty {
            ty::Bool => Some(Constant::Bool(b == 1)),
            ty::Uint(_) | ty::Int(_) => Some(Constant::Int(b)),
            ty::Float(FloatTy::F32) => Some(Constant::F32(f32::from_bits(
                b.try_into().expect("invalid f32 bit representation"),
            ))),
            ty::Float(FloatTy::F64) => Some(Constant::F64(f64::from_bits(
                b.try_into().expect("invalid f64 bit representation"),
            ))),
            // FIXME: implement other conversion
            _ => None,
        },
        ConstValue::ScalarPair(Scalar::Ptr(ptr), Scalar::Bits { bits: n, .. }) => match result.ty.sty {
            ty::Ref(_, tam, _) => match tam.sty {
                ty::Str => {
                    let alloc = tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id);
                    let offset = ptr.offset.bytes().try_into().expect("too-large pointer offset");
                    let n = n as usize;
                    String::from_utf8(alloc.bytes[offset..(offset + n)].to_owned())
                        .ok()
                        .map(Constant::Str)
                },
                _ => None,
            },
            _ => None,
        },
        // FIXME: implement other conversions
        _ => None,
    }
}
