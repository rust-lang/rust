#![allow(clippy::float_cmp)]

use crate::{clip, is_direct_expn_of, sext, unsext};
use if_chain::if_chain;
use rustc_ast::ast::{self, LitFloatType, LitKind};
use rustc_data_structures::sync::Lrc;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{BinOp, BinOpKind, Block, Expr, ExprKind, HirId, Item, ItemKind, Node, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::mir;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::ty::SubstsRef;
use rustc_middle::ty::{self, EarlyBinder, FloatTy, ScalarInt, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::symbol::Symbol;
use std::cmp::Ordering::{self, Equal};
use std::hash::{Hash, Hasher};
use std::iter;

/// A `LitKind`-like enum to fold constant `Expr`s into.
#[derive(Debug, Clone)]
pub enum Constant {
    /// A `String` (e.g., "abc").
    Str(String),
    /// A binary string (e.g., `b"abc"`).
    Binary(Lrc<[u8]>),
    /// A single `char` (e.g., `'a'`).
    Char(char),
    /// An integer's bit representation.
    Int(u128),
    /// An `f32`.
    F32(f32),
    /// An `f64`.
    F64(f64),
    /// `true` or `false`.
    Bool(bool),
    /// An array of constants.
    Vec(Vec<Constant>),
    /// Also an array, but with only one constant, repeated N times.
    Repeat(Box<Constant>, u64),
    /// A tuple of constants.
    Tuple(Vec<Constant>),
    /// A raw pointer.
    RawPtr(u128),
    /// A reference
    Ref(Box<Constant>),
    /// A literal with syntax error.
    Err,
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Str(ls), Self::Str(rs)) => ls == rs,
            (Self::Binary(l), Self::Binary(r)) => l == r,
            (&Self::Char(l), &Self::Char(r)) => l == r,
            (&Self::Int(l), &Self::Int(r)) => l == r,
            (&Self::F64(l), &Self::F64(r)) => {
                // We want `Fw32 == FwAny` and `FwAny == Fw64`, and by transitivity we must have
                // `Fw32 == Fw64`, so don’t compare them.
                // `to_bits` is required to catch non-matching 0.0, -0.0, and NaNs.
                l.to_bits() == r.to_bits()
            },
            (&Self::F32(l), &Self::F32(r)) => {
                // We want `Fw32 == FwAny` and `FwAny == Fw64`, and by transitivity we must have
                // `Fw32 == Fw64`, so don’t compare them.
                // `to_bits` is required to catch non-matching 0.0, -0.0, and NaNs.
                f64::from(l).to_bits() == f64::from(r).to_bits()
            },
            (&Self::Bool(l), &Self::Bool(r)) => l == r,
            (&Self::Vec(ref l), &Self::Vec(ref r)) | (&Self::Tuple(ref l), &Self::Tuple(ref r)) => l == r,
            (Self::Repeat(lv, ls), Self::Repeat(rv, rs)) => ls == rs && lv == rv,
            (Self::Ref(lb), Self::Ref(rb)) => *lb == *rb,
            // TODO: are there inter-type equalities?
            _ => false,
        }
    }
}

impl Hash for Constant {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        std::mem::discriminant(self).hash(state);
        match *self {
            Self::Str(ref s) => {
                s.hash(state);
            },
            Self::Binary(ref b) => {
                b.hash(state);
            },
            Self::Char(c) => {
                c.hash(state);
            },
            Self::Int(i) => {
                i.hash(state);
            },
            Self::F32(f) => {
                f64::from(f).to_bits().hash(state);
            },
            Self::F64(f) => {
                f.to_bits().hash(state);
            },
            Self::Bool(b) => {
                b.hash(state);
            },
            Self::Vec(ref v) | Self::Tuple(ref v) => {
                v.hash(state);
            },
            Self::Repeat(ref c, l) => {
                c.hash(state);
                l.hash(state);
            },
            Self::RawPtr(u) => {
                u.hash(state);
            },
            Self::Ref(ref r) => {
                r.hash(state);
            },
            Self::Err => {},
        }
    }
}

impl Constant {
    pub fn partial_cmp(tcx: TyCtxt<'_>, cmp_type: Ty<'_>, left: &Self, right: &Self) -> Option<Ordering> {
        match (left, right) {
            (Self::Str(ls), Self::Str(rs)) => Some(ls.cmp(rs)),
            (Self::Char(l), Self::Char(r)) => Some(l.cmp(r)),
            (&Self::Int(l), &Self::Int(r)) => match *cmp_type.kind() {
                ty::Int(int_ty) => Some(sext(tcx, l, int_ty).cmp(&sext(tcx, r, int_ty))),
                ty::Uint(_) => Some(l.cmp(&r)),
                _ => bug!("Not an int type"),
            },
            (&Self::F64(l), &Self::F64(r)) => l.partial_cmp(&r),
            (&Self::F32(l), &Self::F32(r)) => l.partial_cmp(&r),
            (Self::Bool(l), Self::Bool(r)) => Some(l.cmp(r)),
            (Self::Tuple(l), Self::Tuple(r)) if l.len() == r.len() => match *cmp_type.kind() {
                ty::Tuple(tys) if tys.len() == l.len() => l
                    .iter()
                    .zip(r)
                    .zip(tys)
                    .map(|((li, ri), cmp_type)| Self::partial_cmp(tcx, cmp_type, li, ri))
                    .find(|r| r.map_or(true, |o| o != Ordering::Equal))
                    .unwrap_or_else(|| Some(l.len().cmp(&r.len()))),
                _ => None,
            },
            (Self::Vec(l), Self::Vec(r)) => {
                let (ty::Array(cmp_type, _) | ty::Slice(cmp_type)) = *cmp_type.kind() else {
                    return None
                };
                iter::zip(l, r)
                    .map(|(li, ri)| Self::partial_cmp(tcx, cmp_type, li, ri))
                    .find(|r| r.map_or(true, |o| o != Ordering::Equal))
                    .unwrap_or_else(|| Some(l.len().cmp(&r.len())))
            },
            (Self::Repeat(lv, ls), Self::Repeat(rv, rs)) => {
                match Self::partial_cmp(
                    tcx,
                    match *cmp_type.kind() {
                        ty::Array(ty, _) => ty,
                        _ => return None,
                    },
                    lv,
                    rv,
                ) {
                    Some(Equal) => Some(ls.cmp(rs)),
                    x => x,
                }
            },
            (Self::Ref(lb), Self::Ref(rb)) => Self::partial_cmp(
                tcx,
                match *cmp_type.kind() {
                    ty::Ref(_, ty, _) => ty,
                    _ => return None,
                },
                lb,
                rb,
            ),
            // TODO: are there any useful inter-type orderings?
            _ => None,
        }
    }

    /// Returns the integer value or `None` if `self` or `val_type` is not integer type.
    pub fn int_value(&self, cx: &LateContext<'_>, val_type: Ty<'_>) -> Option<FullInt> {
        if let Constant::Int(const_int) = *self {
            match *val_type.kind() {
                ty::Int(ity) => Some(FullInt::S(sext(cx.tcx, const_int, ity))),
                ty::Uint(_) => Some(FullInt::U(const_int)),
                _ => None,
            }
        } else {
            None
        }
    }

    #[must_use]
    pub fn peel_refs(mut self) -> Self {
        while let Constant::Ref(r) = self {
            self = *r;
        }
        self
    }
}

/// Parses a `LitKind` to a `Constant`.
pub fn lit_to_mir_constant(lit: &LitKind, ty: Option<Ty<'_>>) -> Constant {
    match *lit {
        LitKind::Str(ref is, _) => Constant::Str(is.to_string()),
        LitKind::Byte(b) => Constant::Int(u128::from(b)),
        LitKind::ByteStr(ref s, _) => Constant::Binary(Lrc::clone(s)),
        LitKind::Char(c) => Constant::Char(c),
        LitKind::Int(n, _) => Constant::Int(n),
        LitKind::Float(ref is, LitFloatType::Suffixed(fty)) => match fty {
            ast::FloatTy::F32 => Constant::F32(is.as_str().parse().unwrap()),
            ast::FloatTy::F64 => Constant::F64(is.as_str().parse().unwrap()),
        },
        LitKind::Float(ref is, LitFloatType::Unsuffixed) => match ty.expect("type of float is known").kind() {
            ty::Float(FloatTy::F32) => Constant::F32(is.as_str().parse().unwrap()),
            ty::Float(FloatTy::F64) => Constant::F64(is.as_str().parse().unwrap()),
            _ => bug!(),
        },
        LitKind::Bool(b) => Constant::Bool(b),
        LitKind::Err => Constant::Err,
    }
}

pub fn constant<'tcx>(
    lcx: &LateContext<'tcx>,
    typeck_results: &ty::TypeckResults<'tcx>,
    e: &Expr<'_>,
) -> Option<(Constant, bool)> {
    let mut cx = ConstEvalLateContext {
        lcx,
        typeck_results,
        param_env: lcx.param_env,
        needed_resolution: false,
        substs: lcx.tcx.intern_substs(&[]),
    };
    cx.expr(e).map(|cst| (cst, cx.needed_resolution))
}

pub fn constant_simple<'tcx>(
    lcx: &LateContext<'tcx>,
    typeck_results: &ty::TypeckResults<'tcx>,
    e: &Expr<'_>,
) -> Option<Constant> {
    constant(lcx, typeck_results, e).and_then(|(cst, res)| if res { None } else { Some(cst) })
}

pub fn constant_full_int<'tcx>(
    lcx: &LateContext<'tcx>,
    typeck_results: &ty::TypeckResults<'tcx>,
    e: &Expr<'_>,
) -> Option<FullInt> {
    constant_simple(lcx, typeck_results, e)?.int_value(lcx, typeck_results.expr_ty(e))
}

#[derive(Copy, Clone, Debug, Eq)]
pub enum FullInt {
    S(i128),
    U(u128),
}

impl PartialEq for FullInt {
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl PartialOrd for FullInt {
    #[must_use]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FullInt {
    #[must_use]
    fn cmp(&self, other: &Self) -> Ordering {
        use FullInt::{S, U};

        fn cmp_s_u(s: i128, u: u128) -> Ordering {
            u128::try_from(s).map_or(Ordering::Less, |x| x.cmp(&u))
        }

        match (*self, *other) {
            (S(s), S(o)) => s.cmp(&o),
            (U(s), U(o)) => s.cmp(&o),
            (S(s), U(o)) => cmp_s_u(s, o),
            (U(s), S(o)) => cmp_s_u(o, s).reverse(),
        }
    }
}

/// Creates a `ConstEvalLateContext` from the given `LateContext` and `TypeckResults`.
pub fn constant_context<'a, 'tcx>(
    lcx: &'a LateContext<'tcx>,
    typeck_results: &'a ty::TypeckResults<'tcx>,
) -> ConstEvalLateContext<'a, 'tcx> {
    ConstEvalLateContext {
        lcx,
        typeck_results,
        param_env: lcx.param_env,
        needed_resolution: false,
        substs: lcx.tcx.intern_substs(&[]),
    }
}

pub struct ConstEvalLateContext<'a, 'tcx> {
    lcx: &'a LateContext<'tcx>,
    typeck_results: &'a ty::TypeckResults<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    needed_resolution: bool,
    substs: SubstsRef<'tcx>,
}

impl<'a, 'tcx> ConstEvalLateContext<'a, 'tcx> {
    /// Simple constant folding: Insert an expression, get a constant or none.
    pub fn expr(&mut self, e: &Expr<'_>) -> Option<Constant> {
        match e.kind {
            ExprKind::Path(ref qpath) => self.fetch_path(qpath, e.hir_id, self.typeck_results.expr_ty(e)),
            ExprKind::Block(block, _) => self.block(block),
            ExprKind::Lit(ref lit) => {
                if is_direct_expn_of(e.span, "cfg").is_some() {
                    None
                } else {
                    Some(lit_to_mir_constant(&lit.node, self.typeck_results.expr_ty_opt(e)))
                }
            },
            ExprKind::Array(vec) => self.multi(vec).map(Constant::Vec),
            ExprKind::Tup(tup) => self.multi(tup).map(Constant::Tuple),
            ExprKind::Repeat(value, _) => {
                let n = match self.typeck_results.expr_ty(e).kind() {
                    ty::Array(_, n) => n.try_eval_usize(self.lcx.tcx, self.lcx.param_env)?,
                    _ => span_bug!(e.span, "typeck error"),
                };
                self.expr(value).map(|v| Constant::Repeat(Box::new(v), n))
            },
            ExprKind::Unary(op, operand) => self.expr(operand).and_then(|o| match op {
                UnOp::Not => self.constant_not(&o, self.typeck_results.expr_ty(e)),
                UnOp::Neg => self.constant_negate(&o, self.typeck_results.expr_ty(e)),
                UnOp::Deref => Some(if let Constant::Ref(r) = o { *r } else { o }),
            }),
            ExprKind::If(cond, then, ref otherwise) => self.ifthenelse(cond, then, *otherwise),
            ExprKind::Binary(op, left, right) => self.binop(op, left, right),
            ExprKind::Call(callee, args) => {
                // We only handle a few const functions for now.
                if_chain! {
                    if args.is_empty();
                    if let ExprKind::Path(qpath) = &callee.kind;
                    let res = self.typeck_results.qpath_res(qpath, callee.hir_id);
                    if let Some(def_id) = res.opt_def_id();
                    let def_path = self.lcx.get_def_path(def_id);
                    let def_path: Vec<&str> = def_path.iter().take(4).map(Symbol::as_str).collect();
                    if let ["core", "num", int_impl, "max_value"] = *def_path;
                    then {
                        let value = match int_impl {
                            "<impl i8>" => i8::MAX as u128,
                            "<impl i16>" => i16::MAX as u128,
                            "<impl i32>" => i32::MAX as u128,
                            "<impl i64>" => i64::MAX as u128,
                            "<impl i128>" => i128::MAX as u128,
                            _ => return None,
                        };
                        Some(Constant::Int(value))
                    } else {
                        None
                    }
                }
            },
            ExprKind::Index(arr, index) => self.index(arr, index),
            ExprKind::AddrOf(_, _, inner) => self.expr(inner).map(|r| Constant::Ref(Box::new(r))),
            // TODO: add other expressions.
            _ => None,
        }
    }

    #[expect(clippy::cast_possible_wrap)]
    fn constant_not(&self, o: &Constant, ty: Ty<'_>) -> Option<Constant> {
        use self::Constant::{Bool, Int};
        match *o {
            Bool(b) => Some(Bool(!b)),
            Int(value) => {
                let value = !value;
                match *ty.kind() {
                    ty::Int(ity) => Some(Int(unsext(self.lcx.tcx, value as i128, ity))),
                    ty::Uint(ity) => Some(Int(clip(self.lcx.tcx, value, ity))),
                    _ => None,
                }
            },
            _ => None,
        }
    }

    fn constant_negate(&self, o: &Constant, ty: Ty<'_>) -> Option<Constant> {
        use self::Constant::{Int, F32, F64};
        match *o {
            Int(value) => {
                let ty::Int(ity) = *ty.kind() else { return None };
                // sign extend
                let value = sext(self.lcx.tcx, value, ity);
                let value = value.checked_neg()?;
                // clear unused bits
                Some(Int(unsext(self.lcx.tcx, value, ity)))
            },
            F32(f) => Some(F32(-f)),
            F64(f) => Some(F64(-f)),
            _ => None,
        }
    }

    /// Create `Some(Vec![..])` of all constants, unless there is any
    /// non-constant part.
    fn multi(&mut self, vec: &[Expr<'_>]) -> Option<Vec<Constant>> {
        vec.iter().map(|elem| self.expr(elem)).collect::<Option<_>>()
    }

    /// Lookup a possibly constant expression from an `ExprKind::Path`.
    fn fetch_path(&mut self, qpath: &QPath<'_>, id: HirId, ty: Ty<'tcx>) -> Option<Constant> {
        let res = self.typeck_results.qpath_res(qpath, id);
        match res {
            Res::Def(DefKind::Const | DefKind::AssocConst, def_id) => {
                // Check if this constant is based on `cfg!(..)`,
                // which is NOT constant for our purposes.
                if let Some(node) = self.lcx.tcx.hir().get_if_local(def_id) &&
                let Node::Item(&Item {
                    kind: ItemKind::Const(_, body_id),
                    ..
                }) = node &&
                let Node::Expr(&Expr {
                    kind: ExprKind::Lit(_),
                    span,
                    ..
                }) = self.lcx.tcx.hir().get(body_id.hir_id) &&
                is_direct_expn_of(span, "cfg").is_some() {
                    return None;
                }

                let substs = self.typeck_results.node_substs(id);
                let substs = if self.substs.is_empty() {
                    substs
                } else {
                    EarlyBinder(substs).subst(self.lcx.tcx, self.substs)
                };

                let result = self
                    .lcx
                    .tcx
                    .const_eval_resolve(
                        self.param_env,
                        mir::UnevaluatedConst::new(ty::WithOptConstParam::unknown(def_id), substs),
                        None,
                    )
                    .ok()
                    .map(|val| rustc_middle::mir::ConstantKind::from_value(val, ty))?;
                let result = miri_to_const(self.lcx.tcx, result);
                if result.is_some() {
                    self.needed_resolution = true;
                }
                result
            },
            // FIXME: cover all usable cases.
            _ => None,
        }
    }

    fn index(&mut self, lhs: &'_ Expr<'_>, index: &'_ Expr<'_>) -> Option<Constant> {
        let lhs = self.expr(lhs);
        let index = self.expr(index);

        match (lhs, index) {
            (Some(Constant::Vec(vec)), Some(Constant::Int(index))) => match vec.get(index as usize) {
                Some(Constant::F32(x)) => Some(Constant::F32(*x)),
                Some(Constant::F64(x)) => Some(Constant::F64(*x)),
                _ => None,
            },
            (Some(Constant::Vec(vec)), _) => {
                if !vec.is_empty() && vec.iter().all(|x| *x == vec[0]) {
                    match vec.get(0) {
                        Some(Constant::F32(x)) => Some(Constant::F32(*x)),
                        Some(Constant::F64(x)) => Some(Constant::F64(*x)),
                        _ => None,
                    }
                } else {
                    None
                }
            },
            _ => None,
        }
    }

    /// A block can only yield a constant if it only has one constant expression.
    fn block(&mut self, block: &Block<'_>) -> Option<Constant> {
        if block.stmts.is_empty() {
            block.expr.as_ref().and_then(|b| self.expr(b))
        } else {
            None
        }
    }

    fn ifthenelse(&mut self, cond: &Expr<'_>, then: &Expr<'_>, otherwise: Option<&Expr<'_>>) -> Option<Constant> {
        if let Some(Constant::Bool(b)) = self.expr(cond) {
            if b {
                self.expr(then)
            } else {
                otherwise.as_ref().and_then(|expr| self.expr(expr))
            }
        } else {
            None
        }
    }

    fn binop(&mut self, op: BinOp, left: &Expr<'_>, right: &Expr<'_>) -> Option<Constant> {
        let l = self.expr(left)?;
        let r = self.expr(right);
        match (l, r) {
            (Constant::Int(l), Some(Constant::Int(r))) => match *self.typeck_results.expr_ty_opt(left)?.kind() {
                ty::Int(ity) => {
                    let l = sext(self.lcx.tcx, l, ity);
                    let r = sext(self.lcx.tcx, r, ity);
                    let zext = |n: i128| Constant::Int(unsext(self.lcx.tcx, n, ity));
                    match op.node {
                        BinOpKind::Add => l.checked_add(r).map(zext),
                        BinOpKind::Sub => l.checked_sub(r).map(zext),
                        BinOpKind::Mul => l.checked_mul(r).map(zext),
                        BinOpKind::Div if r != 0 => l.checked_div(r).map(zext),
                        BinOpKind::Rem if r != 0 => l.checked_rem(r).map(zext),
                        BinOpKind::Shr => l.checked_shr(r.try_into().ok()?).map(zext),
                        BinOpKind::Shl => l.checked_shl(r.try_into().ok()?).map(zext),
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
                    BinOpKind::Shr => l.checked_shr(r.try_into().ok()?).map(Constant::Int),
                    BinOpKind::Shl => l.checked_shl(r.try_into().ok()?).map(Constant::Int),
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

pub fn miri_to_const<'tcx>(tcx: TyCtxt<'tcx>, result: mir::ConstantKind<'tcx>) -> Option<Constant> {
    use rustc_middle::mir::interpret::ConstValue;
    match result {
        mir::ConstantKind::Val(ConstValue::Scalar(Scalar::Int(int)), _) => {
            match result.ty().kind() {
                ty::Bool => Some(Constant::Bool(int == ScalarInt::TRUE)),
                ty::Uint(_) | ty::Int(_) => Some(Constant::Int(int.assert_bits(int.size()))),
                ty::Float(FloatTy::F32) => Some(Constant::F32(f32::from_bits(
                    int.try_into().expect("invalid f32 bit representation"),
                ))),
                ty::Float(FloatTy::F64) => Some(Constant::F64(f64::from_bits(
                    int.try_into().expect("invalid f64 bit representation"),
                ))),
                ty::RawPtr(type_and_mut) => {
                    if let ty::Uint(_) = type_and_mut.ty.kind() {
                        return Some(Constant::RawPtr(int.assert_bits(int.size())));
                    }
                    None
                },
                // FIXME: implement other conversions.
                _ => None,
            }
        },
        mir::ConstantKind::Val(ConstValue::Slice { data, start, end }, _) => match result.ty().kind() {
            ty::Ref(_, tam, _) => match tam.kind() {
                ty::Str => String::from_utf8(
                    data.inner()
                        .inspect_with_uninit_and_ptr_outside_interpreter(start..end)
                        .to_owned(),
                )
                .ok()
                .map(Constant::Str),
                _ => None,
            },
            _ => None,
        },
        mir::ConstantKind::Val(ConstValue::ByRef { alloc, offset: _ }, _) => match result.ty().kind() {
            ty::Array(sub_type, len) => match sub_type.kind() {
                ty::Float(FloatTy::F32) => match len.kind().try_to_machine_usize(tcx) {
                    Some(len) => alloc
                        .inner()
                        .inspect_with_uninit_and_ptr_outside_interpreter(0..(4 * usize::try_from(len).unwrap()))
                        .to_owned()
                        .array_chunks::<4>()
                        .map(|&chunk| Some(Constant::F32(f32::from_le_bytes(chunk))))
                        .collect::<Option<Vec<Constant>>>()
                        .map(Constant::Vec),
                    _ => None,
                },
                ty::Float(FloatTy::F64) => match len.kind().try_to_machine_usize(tcx) {
                    Some(len) => alloc
                        .inner()
                        .inspect_with_uninit_and_ptr_outside_interpreter(0..(8 * usize::try_from(len).unwrap()))
                        .to_owned()
                        .array_chunks::<8>()
                        .map(|&chunk| Some(Constant::F64(f64::from_le_bytes(chunk))))
                        .collect::<Option<Vec<Constant>>>()
                        .map(Constant::Vec),
                    _ => None,
                },
                // FIXME: implement other array type conversions.
                _ => None,
            },
            _ => None,
        },
        // FIXME: implement other conversions.
        _ => None,
    }
}
