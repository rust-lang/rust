//! A simple const eval API, for use on arbitrary HIR expressions.
//!
//! This cannot use rustc's const eval, aka miri, as arbitrary HIR expressions cannot be lowered to
//! executable MIR bodies, so we have to do this instead.
#![allow(clippy::float_cmp)]

use std::sync::Arc;

use crate::source::{SpanRangeExt, walk_span_to_context};
use crate::{clip, is_direct_expn_of, sext, unsext};

use rustc_abi::Size;
use rustc_apfloat::Float;
use rustc_apfloat::ieee::{Half, Quad};
use rustc_ast::ast::{self, LitFloatType, LitKind};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{
    BinOpKind, Block, ConstBlock, Expr, ExprKind, HirId, Item, ItemKind, Node, PatExpr, PatExprKind, QPath, UnOp,
};
use rustc_lexer::tokenize;
use rustc_lint::LateContext;
use rustc_middle::mir::ConstValue;
use rustc_middle::mir::interpret::{Scalar, alloc_range};
use rustc_middle::ty::{self, FloatTy, IntTy, ScalarInt, Ty, TyCtxt, TypeckResults, UintTy};
use rustc_middle::{bug, mir, span_bug};
use rustc_span::def_id::DefId;
use rustc_span::symbol::Ident;
use rustc_span::{SyntaxContext, sym};
use std::cell::Cell;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::iter;

/// A `LitKind`-like enum to fold constant `Expr`s into.
#[derive(Debug, Clone)]
pub enum Constant<'tcx> {
    Adt(mir::Const<'tcx>),
    /// A `String` (e.g., "abc").
    Str(String),
    /// A binary string (e.g., `b"abc"`).
    Binary(Arc<[u8]>),
    /// A single `char` (e.g., `'a'`).
    Char(char),
    /// An integer's bit representation.
    Int(u128),
    /// An `f16` bitcast to a `u16`.
    // FIXME(f16_f128): use `f16` once builtins are available on all host tools platforms.
    F16(u16),
    /// An `f32`.
    F32(f32),
    /// An `f64`.
    F64(f64),
    /// An `f128` bitcast to a `u128`.
    // FIXME(f16_f128): use `f128` once builtins are available on all host tools platforms.
    F128(u128),
    /// `true` or `false`.
    Bool(bool),
    /// An array of constants.
    Vec(Vec<Constant<'tcx>>),
    /// Also an array, but with only one constant, repeated N times.
    Repeat(Box<Constant<'tcx>>, u64),
    /// A tuple of constants.
    Tuple(Vec<Constant<'tcx>>),
    /// A raw pointer.
    RawPtr(u128),
    /// A reference
    Ref(Box<Constant<'tcx>>),
    /// A literal with syntax error.
    Err,
}

trait IntTypeBounds: Sized {
    type Output: PartialOrd;

    fn min_max(self) -> Option<(Self::Output, Self::Output)>;
    fn bits(self) -> Self::Output;
    fn ensure_fits(self, val: Self::Output) -> Option<Self::Output> {
        let (min, max) = self.min_max()?;
        (min <= val && val <= max).then_some(val)
    }
}
impl IntTypeBounds for UintTy {
    type Output = u128;
    fn min_max(self) -> Option<(Self::Output, Self::Output)> {
        Some(match self {
            UintTy::U8 => (u8::MIN.into(), u8::MAX.into()),
            UintTy::U16 => (u16::MIN.into(), u16::MAX.into()),
            UintTy::U32 => (u32::MIN.into(), u32::MAX.into()),
            UintTy::U64 => (u64::MIN.into(), u64::MAX.into()),
            UintTy::U128 => (u128::MIN, u128::MAX),
            UintTy::Usize => (usize::MIN.try_into().ok()?, usize::MAX.try_into().ok()?),
        })
    }
    fn bits(self) -> Self::Output {
        match self {
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
            UintTy::Usize => usize::BITS.into(),
        }
    }
}
impl IntTypeBounds for IntTy {
    type Output = i128;
    fn min_max(self) -> Option<(Self::Output, Self::Output)> {
        Some(match self {
            IntTy::I8 => (i8::MIN.into(), i8::MAX.into()),
            IntTy::I16 => (i16::MIN.into(), i16::MAX.into()),
            IntTy::I32 => (i32::MIN.into(), i32::MAX.into()),
            IntTy::I64 => (i64::MIN.into(), i64::MAX.into()),
            IntTy::I128 => (i128::MIN, i128::MAX),
            IntTy::Isize => (isize::MIN.try_into().ok()?, isize::MAX.try_into().ok()?),
        })
    }
    fn bits(self) -> Self::Output {
        match self {
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
            IntTy::Isize => isize::BITS.into(),
        }
    }
}

impl PartialEq for Constant<'_> {
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

impl Hash for Constant<'_> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        std::mem::discriminant(self).hash(state);
        match *self {
            Self::Adt(ref elem) => {
                elem.hash(state);
            },
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
            Self::F16(f) => {
                // FIXME(f16_f128): once conversions to/from `f128` are available on all platforms,
                f.hash(state);
            },
            Self::F32(f) => {
                f64::from(f).to_bits().hash(state);
            },
            Self::F64(f) => {
                f.to_bits().hash(state);
            },
            Self::F128(f) => {
                f.hash(state);
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

impl Constant<'_> {
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
                    .find(|r| r.is_none_or(|o| o != Ordering::Equal))
                    .unwrap_or_else(|| Some(l.len().cmp(&r.len()))),
                _ => None,
            },
            (Self::Vec(l), Self::Vec(r)) => {
                let (ty::Array(cmp_type, _) | ty::Slice(cmp_type)) = *cmp_type.kind() else {
                    return None;
                };
                iter::zip(l, r)
                    .map(|(li, ri)| Self::partial_cmp(tcx, cmp_type, li, ri))
                    .find(|r| r.is_none_or(|o| o != Ordering::Equal))
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
                    Some(Ordering::Equal) => Some(ls.cmp(rs)),
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
    pub fn int_value(&self, tcx: TyCtxt<'_>, val_type: Ty<'_>) -> Option<FullInt> {
        if let Constant::Int(const_int) = *self {
            match *val_type.kind() {
                ty::Int(ity) => Some(FullInt::S(sext(tcx, const_int, ity))),
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

    fn parse_f16(s: &str) -> Self {
        let f: Half = s.parse().unwrap();
        Self::F16(f.to_bits().try_into().unwrap())
    }

    fn parse_f128(s: &str) -> Self {
        let f: Quad = s.parse().unwrap();
        Self::F128(f.to_bits())
    }
}

/// Parses a `LitKind` to a `Constant`.
pub fn lit_to_mir_constant<'tcx>(lit: &LitKind, ty: Option<Ty<'tcx>>) -> Constant<'tcx> {
    match *lit {
        LitKind::Str(ref is, _) => Constant::Str(is.to_string()),
        LitKind::Byte(b) => Constant::Int(u128::from(b)),
        LitKind::ByteStr(ref s, _) | LitKind::CStr(ref s, _) => Constant::Binary(Arc::clone(s)),
        LitKind::Char(c) => Constant::Char(c),
        LitKind::Int(n, _) => Constant::Int(n.get()),
        LitKind::Float(ref is, LitFloatType::Suffixed(fty)) => match fty {
            // FIXME(f16_f128): just use `parse()` directly when available for `f16`/`f128`
            ast::FloatTy::F16 => Constant::parse_f16(is.as_str()),
            ast::FloatTy::F32 => Constant::F32(is.as_str().parse().unwrap()),
            ast::FloatTy::F64 => Constant::F64(is.as_str().parse().unwrap()),
            ast::FloatTy::F128 => Constant::parse_f128(is.as_str()),
        },
        LitKind::Float(ref is, LitFloatType::Unsuffixed) => match ty.expect("type of float is known").kind() {
            ty::Float(FloatTy::F16) => Constant::parse_f16(is.as_str()),
            ty::Float(FloatTy::F32) => Constant::F32(is.as_str().parse().unwrap()),
            ty::Float(FloatTy::F64) => Constant::F64(is.as_str().parse().unwrap()),
            ty::Float(FloatTy::F128) => Constant::parse_f128(is.as_str()),
            _ => bug!(),
        },
        LitKind::Bool(b) => Constant::Bool(b),
        LitKind::Err(_) => Constant::Err,
    }
}

/// The source of a constant value.
#[derive(Clone, Copy)]
pub enum ConstantSource {
    /// The value is determined solely from the expression.
    Local,
    /// The value is dependent on a defined constant.
    Constant,
    /// The value is dependent on a constant defined in `core` crate.
    CoreConstant,
}
impl ConstantSource {
    pub fn is_local(self) -> bool {
        matches!(self, Self::Local)
    }
}

#[derive(Copy, Clone, Debug, Eq)]
pub enum FullInt {
    S(i128),
    U(u128),
}

impl PartialEq for FullInt {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl PartialOrd for FullInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FullInt {
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

/// The context required to evaluate a constant expression.
///
/// This is currently limited to constant folding and reading the value of named constants.
///
/// See the module level documentation for some context.
pub struct ConstEvalCtxt<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    typeck: &'tcx TypeckResults<'tcx>,
    source: Cell<ConstantSource>,
}

impl<'tcx> ConstEvalCtxt<'tcx> {
    /// Creates the evaluation context from the lint context. This requires the lint context to be
    /// in a body (i.e. `cx.enclosing_body.is_some()`).
    pub fn new(cx: &LateContext<'tcx>) -> Self {
        Self {
            tcx: cx.tcx,
            typing_env: cx.typing_env(),
            typeck: cx.typeck_results(),
            source: Cell::new(ConstantSource::Local),
        }
    }

    /// Creates an evaluation context.
    pub fn with_env(tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>, typeck: &'tcx TypeckResults<'tcx>) -> Self {
        Self {
            tcx,
            typing_env,
            typeck,
            source: Cell::new(ConstantSource::Local),
        }
    }

    /// Attempts to evaluate the expression and returns both the value and whether it's dependant on
    /// other items.
    pub fn eval_with_source(&self, e: &Expr<'_>) -> Option<(Constant<'tcx>, ConstantSource)> {
        self.source.set(ConstantSource::Local);
        self.expr(e).map(|c| (c, self.source.get()))
    }

    /// Attempts to evaluate the expression.
    pub fn eval(&self, e: &Expr<'_>) -> Option<Constant<'tcx>> {
        self.expr(e)
    }

    /// Attempts to evaluate the expression without accessing other items.
    pub fn eval_simple(&self, e: &Expr<'_>) -> Option<Constant<'tcx>> {
        match self.eval_with_source(e) {
            Some((x, ConstantSource::Local)) => Some(x),
            _ => None,
        }
    }

    /// Attempts to evaluate the expression as an integer without accessing other items.
    pub fn eval_full_int(&self, e: &Expr<'_>) -> Option<FullInt> {
        match self.eval_with_source(e) {
            Some((x, ConstantSource::Local)) => x.int_value(self.tcx, self.typeck.expr_ty(e)),
            _ => None,
        }
    }

    pub fn eval_pat_expr(&self, pat_expr: &PatExpr<'_>) -> Option<Constant<'tcx>> {
        match &pat_expr.kind {
            PatExprKind::Lit { lit, negated } => {
                let ty = self.typeck.node_type_opt(pat_expr.hir_id);
                let val = lit_to_mir_constant(&lit.node, ty);
                if *negated {
                    self.constant_negate(&val, ty?)
                } else {
                    Some(val)
                }
            },
            PatExprKind::ConstBlock(ConstBlock { body, .. }) => self.expr(self.tcx.hir_body(*body).value),
            PatExprKind::Path(qpath) => self.qpath(qpath, pat_expr.hir_id),
        }
    }

    fn qpath(&self, qpath: &QPath<'_>, hir_id: HirId) -> Option<Constant<'tcx>> {
        let is_core_crate = if let Some(def_id) = self.typeck.qpath_res(qpath, hir_id).opt_def_id() {
            self.tcx.crate_name(def_id.krate) == sym::core
        } else {
            false
        };
        self.fetch_path_and_apply(qpath, hir_id, self.typeck.node_type(hir_id), |self_, result| {
            let result = mir_to_const(self_.tcx, result)?;
            // If source is already Constant we wouldn't want to override it with CoreConstant
            self_.source.set(
                if is_core_crate && !matches!(self_.source.get(), ConstantSource::Constant) {
                    ConstantSource::CoreConstant
                } else {
                    ConstantSource::Constant
                },
            );
            Some(result)
        })
    }

    /// Simple constant folding: Insert an expression, get a constant or none.
    fn expr(&self, e: &Expr<'_>) -> Option<Constant<'tcx>> {
        match e.kind {
            ExprKind::ConstBlock(ConstBlock { body, .. }) => self.expr(self.tcx.hir_body(body).value),
            ExprKind::DropTemps(e) => self.expr(e),
            ExprKind::Path(ref qpath) => self.qpath(qpath, e.hir_id),
            ExprKind::Block(block, _) => self.block(block),
            ExprKind::Lit(lit) => {
                if is_direct_expn_of(e.span, "cfg").is_some() {
                    None
                } else {
                    Some(lit_to_mir_constant(&lit.node, self.typeck.expr_ty_opt(e)))
                }
            },
            ExprKind::Array(vec) => self.multi(vec).map(Constant::Vec),
            ExprKind::Tup(tup) => self.multi(tup).map(Constant::Tuple),
            ExprKind::Repeat(value, _) => {
                let n = match self.typeck.expr_ty(e).kind() {
                    ty::Array(_, n) => n.try_to_target_usize(self.tcx)?,
                    _ => span_bug!(e.span, "typeck error"),
                };
                self.expr(value).map(|v| Constant::Repeat(Box::new(v), n))
            },
            ExprKind::Unary(op, operand) => self.expr(operand).and_then(|o| match op {
                UnOp::Not => self.constant_not(&o, self.typeck.expr_ty(e)),
                UnOp::Neg => self.constant_negate(&o, self.typeck.expr_ty(e)),
                UnOp::Deref => Some(if let Constant::Ref(r) = o { *r } else { o }),
            }),
            ExprKind::If(cond, then, ref otherwise) => self.ifthenelse(cond, then, *otherwise),
            ExprKind::Binary(op, left, right) => self.binop(op.node, left, right),
            ExprKind::Call(callee, []) => {
                // We only handle a few const functions for now.
                if let ExprKind::Path(qpath) = &callee.kind
                    && let Some(did) = self.typeck.qpath_res(qpath, callee.hir_id).opt_def_id()
                {
                    match self.tcx.get_diagnostic_name(did) {
                        Some(sym::i8_legacy_fn_max_value) => Some(Constant::Int(i8::MAX as u128)),
                        Some(sym::i16_legacy_fn_max_value) => Some(Constant::Int(i16::MAX as u128)),
                        Some(sym::i32_legacy_fn_max_value) => Some(Constant::Int(i32::MAX as u128)),
                        Some(sym::i64_legacy_fn_max_value) => Some(Constant::Int(i64::MAX as u128)),
                        Some(sym::i128_legacy_fn_max_value) => Some(Constant::Int(i128::MAX as u128)),
                        _ => None,
                    }
                } else {
                    None
                }
            },
            ExprKind::Index(arr, index, _) => self.index(arr, index),
            ExprKind::AddrOf(_, _, inner) => self.expr(inner).map(|r| Constant::Ref(Box::new(r))),
            ExprKind::Field(local_expr, ref field) => {
                let result = self.expr(local_expr);
                if let Some(Constant::Adt(constant)) = &self.expr(local_expr)
                    && let ty::Adt(adt_def, _) = constant.ty().kind()
                    && adt_def.is_struct()
                    && let Some(desired_field) = field_of_struct(*adt_def, self.tcx, *constant, field)
                {
                    mir_to_const(self.tcx, desired_field)
                } else {
                    result
                }
            },
            _ => None,
        }
    }

    /// Simple constant folding to determine if an expression is an empty slice, str, array, …
    /// `None` will be returned if the constness cannot be determined, or if the resolution
    /// leaves the local crate.
    pub fn eval_is_empty(&self, e: &Expr<'_>) -> Option<bool> {
        match e.kind {
            ExprKind::ConstBlock(ConstBlock { body, .. }) => self.eval_is_empty(self.tcx.hir_body(body).value),
            ExprKind::DropTemps(e) => self.eval_is_empty(e),
            ExprKind::Path(ref qpath) => {
                if !self
                    .typeck
                    .qpath_res(qpath, e.hir_id)
                    .opt_def_id()
                    .is_some_and(DefId::is_local)
                {
                    return None;
                }
                self.fetch_path_and_apply(qpath, e.hir_id, self.typeck.expr_ty(e), |self_, result| {
                    mir_is_empty(self_.tcx, result)
                })
            },
            ExprKind::Lit(lit) => {
                if is_direct_expn_of(e.span, "cfg").is_some() {
                    None
                } else {
                    match &lit.node {
                        LitKind::Str(is, _) => Some(is.is_empty()),
                        LitKind::ByteStr(s, _) | LitKind::CStr(s, _) => Some(s.is_empty()),
                        _ => None,
                    }
                }
            },
            ExprKind::Array(vec) => self.multi(vec).map(|v| v.is_empty()),
            ExprKind::Repeat(..) => {
                if let ty::Array(_, n) = self.typeck.expr_ty(e).kind() {
                    Some(n.try_to_target_usize(self.tcx)? == 0)
                } else {
                    span_bug!(e.span, "typeck error");
                }
            },
            _ => None,
        }
    }

    #[expect(clippy::cast_possible_wrap)]
    fn constant_not(&self, o: &Constant<'tcx>, ty: Ty<'_>) -> Option<Constant<'tcx>> {
        use self::Constant::{Bool, Int};
        match *o {
            Bool(b) => Some(Bool(!b)),
            Int(value) => {
                let value = !value;
                match *ty.kind() {
                    ty::Int(ity) => Some(Int(unsext(self.tcx, value as i128, ity))),
                    ty::Uint(ity) => Some(Int(clip(self.tcx, value, ity))),
                    _ => None,
                }
            },
            _ => None,
        }
    }

    fn constant_negate(&self, o: &Constant<'tcx>, ty: Ty<'_>) -> Option<Constant<'tcx>> {
        use self::Constant::{F32, F64, Int};
        match *o {
            Int(value) => {
                let ty::Int(ity) = *ty.kind() else { return None };
                let (min, _) = ity.min_max()?;
                // sign extend
                let value = sext(self.tcx, value, ity);

                // Applying unary - to the most negative value of any signed integer type panics.
                if value == min {
                    return None;
                }

                let value = value.checked_neg()?;
                // clear unused bits
                Some(Int(unsext(self.tcx, value, ity)))
            },
            F32(f) => Some(F32(-f)),
            F64(f) => Some(F64(-f)),
            _ => None,
        }
    }

    /// Create `Some(Vec![..])` of all constants, unless there is any
    /// non-constant part.
    fn multi(&self, vec: &[Expr<'_>]) -> Option<Vec<Constant<'tcx>>> {
        vec.iter().map(|elem| self.expr(elem)).collect::<Option<_>>()
    }

    /// Lookup a possibly constant expression from an `ExprKind::Path` and apply a function on it.
    fn fetch_path_and_apply<T, F>(&self, qpath: &QPath<'_>, id: HirId, ty: Ty<'tcx>, f: F) -> Option<T>
    where
        F: FnOnce(&Self, mir::Const<'tcx>) -> Option<T>,
    {
        let res = self.typeck.qpath_res(qpath, id);
        match res {
            Res::Def(DefKind::Const | DefKind::AssocConst, def_id) => {
                // Check if this constant is based on `cfg!(..)`,
                // which is NOT constant for our purposes.
                if let Some(node) = self.tcx.hir_get_if_local(def_id)
                    && let Node::Item(Item {
                        kind: ItemKind::Const(.., body_id),
                        ..
                    }) = node
                    && let Node::Expr(Expr {
                        kind: ExprKind::Lit(_),
                        span,
                        ..
                    }) = self.tcx.hir_node(body_id.hir_id)
                    && is_direct_expn_of(*span, "cfg").is_some()
                {
                    return None;
                }

                let args = self.typeck.node_args(id);
                let result = self
                    .tcx
                    .const_eval_resolve(self.typing_env, mir::UnevaluatedConst::new(def_id, args), qpath.span())
                    .ok()
                    .map(|val| mir::Const::from_value(val, ty))?;
                f(self, result)
            },
            _ => None,
        }
    }

    fn index(&self, lhs: &'_ Expr<'_>, index: &'_ Expr<'_>) -> Option<Constant<'tcx>> {
        let lhs = self.expr(lhs);
        let index = self.expr(index);

        match (lhs, index) {
            (Some(Constant::Vec(vec)), Some(Constant::Int(index))) => match vec.get(index as usize) {
                Some(Constant::F16(x)) => Some(Constant::F16(*x)),
                Some(Constant::F32(x)) => Some(Constant::F32(*x)),
                Some(Constant::F64(x)) => Some(Constant::F64(*x)),
                Some(Constant::F128(x)) => Some(Constant::F128(*x)),
                _ => None,
            },
            (Some(Constant::Vec(vec)), _) => {
                if !vec.is_empty() && vec.iter().all(|x| *x == vec[0]) {
                    match vec.first() {
                        Some(Constant::F16(x)) => Some(Constant::F16(*x)),
                        Some(Constant::F32(x)) => Some(Constant::F32(*x)),
                        Some(Constant::F64(x)) => Some(Constant::F64(*x)),
                        Some(Constant::F128(x)) => Some(Constant::F128(*x)),
                        _ => None,
                    }
                } else {
                    None
                }
            },
            _ => None,
        }
    }

    /// A block can only yield a constant if it has exactly one constant expression.
    fn block(&self, block: &Block<'_>) -> Option<Constant<'tcx>> {
        if block.stmts.is_empty()
            && let Some(expr) = block.expr
        {
            // Try to detect any `cfg`ed statements or empty macro expansions.
            let span = block.span.data();
            if span.ctxt == SyntaxContext::root() {
                if let Some(expr_span) = walk_span_to_context(expr.span, span.ctxt)
                    && let expr_lo = expr_span.lo()
                    && expr_lo >= span.lo
                    && let Some(src) = (span.lo..expr_lo).get_source_range(&self.tcx)
                    && let Some(src) = src.as_str()
                {
                    use rustc_lexer::TokenKind::{BlockComment, LineComment, OpenBrace, Semi, Whitespace};
                    if !tokenize(src)
                        .map(|t| t.kind)
                        .filter(|t| !matches!(t, Whitespace | LineComment { .. } | BlockComment { .. } | Semi))
                        .eq([OpenBrace])
                    {
                        self.source.set(ConstantSource::Constant);
                    }
                } else {
                    // Unable to access the source. Assume a non-local dependency.
                    self.source.set(ConstantSource::Constant);
                }
            }

            self.expr(expr)
        } else {
            None
        }
    }

    fn ifthenelse(&self, cond: &Expr<'_>, then: &Expr<'_>, otherwise: Option<&Expr<'_>>) -> Option<Constant<'tcx>> {
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

    fn binop(&self, op: BinOpKind, left: &Expr<'_>, right: &Expr<'_>) -> Option<Constant<'tcx>> {
        let l = self.expr(left)?;
        let r = self.expr(right);
        match (l, r) {
            (Constant::Int(l), Some(Constant::Int(r))) => match *self.typeck.expr_ty_opt(left)?.kind() {
                ty::Int(ity) => {
                    let (ty_min_value, _) = ity.min_max()?;
                    let bits = ity.bits();
                    let l = sext(self.tcx, l, ity);
                    let r = sext(self.tcx, r, ity);

                    // Using / or %, where the left-hand argument is the smallest integer of a signed integer type and
                    // the right-hand argument is -1 always panics, even with overflow-checks disabled
                    if let BinOpKind::Div | BinOpKind::Rem = op
                        && l == ty_min_value
                        && r == -1
                    {
                        return None;
                    }

                    let zext = |n: i128| Constant::Int(unsext(self.tcx, n, ity));
                    match op {
                        // When +, * or binary - create a value greater than the maximum value, or less than
                        // the minimum value that can be stored, it panics.
                        BinOpKind::Add => l.checked_add(r).and_then(|n| ity.ensure_fits(n)).map(zext),
                        BinOpKind::Sub => l.checked_sub(r).and_then(|n| ity.ensure_fits(n)).map(zext),
                        BinOpKind::Mul => l.checked_mul(r).and_then(|n| ity.ensure_fits(n)).map(zext),
                        BinOpKind::Div if r != 0 => l.checked_div(r).map(zext),
                        BinOpKind::Rem if r != 0 => l.checked_rem(r).map(zext),
                        // Using << or >> where the right-hand argument is greater than or equal to the number of bits
                        // in the type of the left-hand argument, or is negative panics.
                        BinOpKind::Shr if r < bits && !r.is_negative() => l.checked_shr(r.try_into().ok()?).map(zext),
                        BinOpKind::Shl if r < bits && !r.is_negative() => l.checked_shl(r.try_into().ok()?).map(zext),
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
                ty::Uint(ity) => {
                    let bits = ity.bits();

                    match op {
                        BinOpKind::Add => l.checked_add(r).and_then(|n| ity.ensure_fits(n)).map(Constant::Int),
                        BinOpKind::Sub => l.checked_sub(r).and_then(|n| ity.ensure_fits(n)).map(Constant::Int),
                        BinOpKind::Mul => l.checked_mul(r).and_then(|n| ity.ensure_fits(n)).map(Constant::Int),
                        BinOpKind::Div => l.checked_div(r).map(Constant::Int),
                        BinOpKind::Rem => l.checked_rem(r).map(Constant::Int),
                        BinOpKind::Shr if r < bits => l.checked_shr(r.try_into().ok()?).map(Constant::Int),
                        BinOpKind::Shl if r < bits => l.checked_shl(r.try_into().ok()?).map(Constant::Int),
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
                    }
                },
                _ => None,
            },
            // FIXME(f16_f128): add these types when binary operations are available on all platforms
            (Constant::F32(l), Some(Constant::F32(r))) => match op {
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
            (Constant::F64(l), Some(Constant::F64(r))) => match op {
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
            (l, r) => match (op, l, r) {
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

pub fn mir_to_const<'tcx>(tcx: TyCtxt<'tcx>, result: mir::Const<'tcx>) -> Option<Constant<'tcx>> {
    let mir::Const::Val(val, _) = result else {
        // We only work on evaluated consts.
        return None;
    };
    match (val, result.ty().kind()) {
        (ConstValue::Scalar(Scalar::Int(int)), _) => match result.ty().kind() {
            ty::Adt(adt_def, _) if adt_def.is_struct() => Some(Constant::Adt(result)),
            ty::Bool => Some(Constant::Bool(int == ScalarInt::TRUE)),
            ty::Uint(_) | ty::Int(_) => Some(Constant::Int(int.to_bits(int.size()))),
            ty::Float(FloatTy::F16) => Some(Constant::F16(int.into())),
            ty::Float(FloatTy::F32) => Some(Constant::F32(f32::from_bits(int.into()))),
            ty::Float(FloatTy::F64) => Some(Constant::F64(f64::from_bits(int.into()))),
            ty::Float(FloatTy::F128) => Some(Constant::F128(int.into())),
            ty::RawPtr(_, _) => Some(Constant::RawPtr(int.to_bits(int.size()))),
            _ => None,
        },
        (_, ty::Ref(_, inner_ty, _)) if matches!(inner_ty.kind(), ty::Str) => {
            let data = val.try_get_slice_bytes_for_diagnostics(tcx)?;
            String::from_utf8(data.to_owned()).ok().map(Constant::Str)
        },
        (_, ty::Adt(adt_def, _)) if adt_def.is_struct() => Some(Constant::Adt(result)),
        (ConstValue::Indirect { alloc_id, offset }, ty::Array(sub_type, len)) => {
            let alloc = tcx.global_alloc(alloc_id).unwrap_memory().inner();
            let len = len.try_to_target_usize(tcx)?;
            let ty::Float(flt) = sub_type.kind() else {
                return None;
            };
            let size = Size::from_bits(flt.bit_width());
            let mut res = Vec::new();
            for idx in 0..len {
                let range = alloc_range(offset + size * idx, size);
                let val = alloc.read_scalar(&tcx, range, /* read_provenance */ false).ok()?;
                res.push(match flt {
                    FloatTy::F16 => Constant::F16(val.to_u16().discard_err()?),
                    FloatTy::F32 => Constant::F32(f32::from_bits(val.to_u32().discard_err()?)),
                    FloatTy::F64 => Constant::F64(f64::from_bits(val.to_u64().discard_err()?)),
                    FloatTy::F128 => Constant::F128(val.to_u128().discard_err()?),
                });
            }
            Some(Constant::Vec(res))
        },
        _ => None,
    }
}

fn mir_is_empty<'tcx>(tcx: TyCtxt<'tcx>, result: mir::Const<'tcx>) -> Option<bool> {
    let mir::Const::Val(val, _) = result else {
        // We only work on evaluated consts.
        return None;
    };
    match (val, result.ty().kind()) {
        (_, ty::Ref(_, inner_ty, _)) => match inner_ty.kind() {
            ty::Str | ty::Slice(_) => {
                if let ConstValue::Indirect { alloc_id, offset } = val {
                    // Get the length from the slice, using the same formula as
                    // [`ConstValue::try_get_slice_bytes_for_diagnostics`].
                    let a = tcx.global_alloc(alloc_id).unwrap_memory().inner();
                    let ptr_size = tcx.data_layout.pointer_size;
                    if a.size() < offset + 2 * ptr_size {
                        // (partially) dangling reference
                        return None;
                    }
                    let len = a
                        .read_scalar(&tcx, alloc_range(offset + ptr_size, ptr_size), false)
                        .ok()?
                        .to_target_usize(&tcx)
                        .discard_err()?;
                    Some(len == 0)
                } else {
                    None
                }
            },
            ty::Array(_, len) => Some(len.try_to_target_usize(tcx)? == 0),
            _ => None,
        },
        (ConstValue::Indirect { .. }, ty::Array(_, len)) => Some(len.try_to_target_usize(tcx)? == 0),
        (ConstValue::ZeroSized, _) => Some(true),
        _ => None,
    }
}

fn field_of_struct<'tcx>(
    adt_def: ty::AdtDef<'tcx>,
    tcx: TyCtxt<'tcx>,
    result: mir::Const<'tcx>,
    field: &Ident,
) -> Option<mir::Const<'tcx>> {
    if let mir::Const::Val(result, ty) = result
        && let Some(dc) = tcx.try_destructure_mir_constant_for_user_output(result, ty)
        && let Some(dc_variant) = dc.variant
        && let Some(variant) = adt_def.variants().get(dc_variant)
        && let Some(field_idx) = variant.fields.iter().position(|el| el.name == field.name)
        && let Some(&(val, ty)) = dc.fields.get(field_idx)
    {
        Some(mir::Const::Val(val, ty))
    } else {
        None
    }
}
