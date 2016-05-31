#![allow(cast_possible_truncation)]

use rustc::lint::LateContext;
use rustc::hir::def::{Def, PathResolution};
use rustc_const_eval::lookup_const_by_id;
use rustc_const_math::{ConstInt, ConstUsize, ConstIsize};
use rustc::hir::*;
use std::cmp::Ordering::{self, Equal};
use std::cmp::PartialOrd;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::rc::Rc;
use syntax::ast::{FloatTy, LitIntType, LitKind, StrStyle, UintTy, IntTy};
use syntax::ptr::P;

#[derive(Debug, Copy, Clone)]
pub enum FloatWidth {
    F32,
    F64,
    Any,
}

impl From<FloatTy> for FloatWidth {
    fn from(ty: FloatTy) -> FloatWidth {
        match ty {
            FloatTy::F32 => FloatWidth::F32,
            FloatTy::F64 => FloatWidth::F64,
        }
    }
}

/// A `LitKind`-like enum to fold constant `Expr`s into.
#[derive(Debug, Clone)]
pub enum Constant {
    /// a String "abc"
    Str(String, StrStyle),
    /// a Binary String b"abc"
    Binary(Rc<Vec<u8>>),
    /// a single char 'a'
    Char(char),
    /// an integer, third argument is whether the value is negated
    Int(ConstInt),
    /// a float with given type
    Float(String, FloatWidth),
    /// true or false
    Bool(bool),
    /// an array of constants
    Vec(Vec<Constant>),
    /// also an array, but with only one constant, repeated N times
    Repeat(Box<Constant>, usize),
    /// a tuple of constants
    Tuple(Vec<Constant>),
}

impl Constant {
    /// convert to u64 if possible
    ///
    /// # panics
    ///
    /// if the constant could not be converted to u64 losslessly
    fn as_u64(&self) -> u64 {
        if let Constant::Int(val) = *self {
            val.to_u64().expect("negative constant can't be casted to u64")
        } else {
            panic!("Could not convert a {:?} to u64", self);
        }
    }

    /// convert this constant to a f64, if possible
    #[allow(cast_precision_loss, cast_possible_wrap)]
    pub fn as_float(&self) -> Option<f64> {
        match *self {
            Constant::Float(ref s, _) => s.parse().ok(),
            Constant::Int(i) if i.is_negative() => Some(i.to_u64_unchecked() as i64 as f64),
            Constant::Int(i) => Some(i.to_u64_unchecked() as f64),
            _ => None,
        }
    }
}

impl PartialEq for Constant {
    fn eq(&self, other: &Constant) -> bool {
        match (self, other) {
            (&Constant::Str(ref ls, ref l_sty), &Constant::Str(ref rs, ref r_sty)) => ls == rs && l_sty == r_sty,
            (&Constant::Binary(ref l), &Constant::Binary(ref r)) => l == r,
            (&Constant::Char(l), &Constant::Char(r)) => l == r,
            (&Constant::Int(l), &Constant::Int(r)) => {
                l.is_negative() == r.is_negative() && l.to_u64_unchecked() == r.to_u64_unchecked()
            }
            (&Constant::Float(ref ls, _), &Constant::Float(ref rs, _)) => {
                // we want `Fw32 == FwAny` and `FwAny == Fw64`, by transitivity we must have
                // `Fw32 == Fw64` so don’t compare them
                match (ls.parse::<f64>(), rs.parse::<f64>()) {
                    (Ok(l), Ok(r)) => l.eq(&r),
                    _ => false,
                }
            }
            (&Constant::Bool(l), &Constant::Bool(r)) => l == r,
            (&Constant::Vec(ref l), &Constant::Vec(ref r)) => l == r,
            (&Constant::Repeat(ref lv, ref ls), &Constant::Repeat(ref rv, ref rs)) => ls == rs && lv == rv,
            (&Constant::Tuple(ref l), &Constant::Tuple(ref r)) => l == r,
            _ => false, //TODO: Are there inter-type equalities?
        }
    }
}

impl Hash for Constant {
    fn hash<H>(&self, state: &mut H)
        where H: Hasher
    {
        match *self {
            Constant::Str(ref s, ref k) => {
                s.hash(state);
                k.hash(state);
            }
            Constant::Binary(ref b) => {
                b.hash(state);
            }
            Constant::Char(c) => {
                c.hash(state);
            }
            Constant::Int(i) => {
                i.to_u64_unchecked().hash(state);
                i.is_negative().hash(state);
            }
            Constant::Float(ref f, _) => {
                // don’t use the width here because of PartialEq implementation
                if let Ok(f) = f.parse::<f64>() {
                    unsafe { mem::transmute::<f64, u64>(f) }.hash(state);
                }
            }
            Constant::Bool(b) => {
                b.hash(state);
            }
            Constant::Vec(ref v) |
            Constant::Tuple(ref v) => {
                v.hash(state);
            }
            Constant::Repeat(ref c, l) => {
                c.hash(state);
                l.hash(state);
            }
        }
    }
}

impl PartialOrd for Constant {
    fn partial_cmp(&self, other: &Constant) -> Option<Ordering> {
        match (self, other) {
            (&Constant::Str(ref ls, ref l_sty), &Constant::Str(ref rs, ref r_sty)) => {
                if l_sty == r_sty {
                    Some(ls.cmp(rs))
                } else {
                    None
                }
            }
            (&Constant::Char(ref l), &Constant::Char(ref r)) => Some(l.cmp(r)),
            (&Constant::Int(l), &Constant::Int(r)) => Some(l.cmp(&r)),
            (&Constant::Float(ref ls, _), &Constant::Float(ref rs, _)) => {
                match (ls.parse::<f64>(), rs.parse::<f64>()) {
                    (Ok(ref l), Ok(ref r)) => l.partial_cmp(r),
                    _ => None,
                }
            }
            (&Constant::Bool(ref l), &Constant::Bool(ref r)) => Some(l.cmp(r)),
            (&Constant::Tuple(ref l), &Constant::Tuple(ref r)) |
            (&Constant::Vec(ref l), &Constant::Vec(ref r)) => l.partial_cmp(r),
            (&Constant::Repeat(ref lv, ref ls), &Constant::Repeat(ref rv, ref rs)) => {
                match lv.partial_cmp(rv) {
                    Some(Equal) => Some(ls.cmp(rs)),
                    x => x,
                }
            }
            _ => None, //TODO: Are there any useful inter-type orderings?
        }
    }
}

/// parse a `LitKind` to a `Constant`
#[allow(cast_possible_wrap)]
pub fn lit_to_constant(lit: &LitKind) -> Constant {
    match *lit {
        LitKind::Str(ref is, style) => Constant::Str(is.to_string(), style),
        LitKind::Byte(b) => Constant::Int(ConstInt::U8(b)),
        LitKind::ByteStr(ref s) => Constant::Binary(s.clone()),
        LitKind::Char(c) => Constant::Char(c),
        LitKind::Int(value, LitIntType::Unsuffixed) => Constant::Int(ConstInt::Infer(value)),
        LitKind::Int(value, LitIntType::Unsigned(UintTy::U8)) => Constant::Int(ConstInt::U8(value as u8)),
        LitKind::Int(value, LitIntType::Unsigned(UintTy::U16)) => Constant::Int(ConstInt::U16(value as u16)),
        LitKind::Int(value, LitIntType::Unsigned(UintTy::U32)) => Constant::Int(ConstInt::U32(value as u32)),
        LitKind::Int(value, LitIntType::Unsigned(UintTy::U64)) => Constant::Int(ConstInt::U64(value as u64)),
        LitKind::Int(value, LitIntType::Unsigned(UintTy::Us)) => {
            Constant::Int(ConstInt::Usize(ConstUsize::Us32(value as u32)))
        }
        LitKind::Int(value, LitIntType::Signed(IntTy::I8)) => Constant::Int(ConstInt::I8(value as i8)),
        LitKind::Int(value, LitIntType::Signed(IntTy::I16)) => Constant::Int(ConstInt::I16(value as i16)),
        LitKind::Int(value, LitIntType::Signed(IntTy::I32)) => Constant::Int(ConstInt::I32(value as i32)),
        LitKind::Int(value, LitIntType::Signed(IntTy::I64)) => Constant::Int(ConstInt::I64(value as i64)),
        LitKind::Int(value, LitIntType::Signed(IntTy::Is)) => {
            Constant::Int(ConstInt::Isize(ConstIsize::Is32(value as i32)))
        }
        LitKind::Float(ref is, ty) => Constant::Float(is.to_string(), ty.into()),
        LitKind::FloatUnsuffixed(ref is) => Constant::Float(is.to_string(), FloatWidth::Any),
        LitKind::Bool(b) => Constant::Bool(b),
    }
}

fn constant_not(o: Constant) -> Option<Constant> {
    use self::Constant::*;
    match o {
        Bool(b) => Some(Bool(!b)),
        Int(value) => (!value).ok().map(Int),
        _ => None,
    }
}

fn constant_negate(o: Constant) -> Option<Constant> {
    use self::Constant::*;
    match o {
        Int(value) => (-value).ok().map(Int),
        Float(is, ty) => Some(Float(neg_float_str(is), ty)),
        _ => None,
    }
}

fn neg_float_str(s: String) -> String {
    if s.starts_with('-') {
        s[1..].to_owned()
    } else {
        format!("-{}", s)
    }
}

pub fn constant(lcx: &LateContext, e: &Expr) -> Option<(Constant, bool)> {
    let mut cx = ConstEvalLateContext {
        lcx: Some(lcx),
        needed_resolution: false,
    };
    cx.expr(e).map(|cst| (cst, cx.needed_resolution))
}

pub fn constant_simple(e: &Expr) -> Option<Constant> {
    let mut cx = ConstEvalLateContext {
        lcx: None,
        needed_resolution: false,
    };
    cx.expr(e)
}

struct ConstEvalLateContext<'c, 'cc: 'c> {
    lcx: Option<&'c LateContext<'c, 'cc>>,
    needed_resolution: bool,
}

impl<'c, 'cc> ConstEvalLateContext<'c, 'cc> {
    /// simple constant folding: Insert an expression, get a constant or none.
    fn expr(&mut self, e: &Expr) -> Option<Constant> {
        match e.node {
            ExprPath(_, _) => self.fetch_path(e),
            ExprBlock(ref block) => self.block(block),
            ExprIf(ref cond, ref then, ref otherwise) => self.ifthenelse(cond, then, otherwise),
            ExprLit(ref lit) => Some(lit_to_constant(&lit.node)),
            ExprVec(ref vec) => self.multi(vec).map(Constant::Vec),
            ExprTup(ref tup) => self.multi(tup).map(Constant::Tuple),
            ExprRepeat(ref value, ref number) => {
                self.binop_apply(value, number, |v, n| Some(Constant::Repeat(Box::new(v), n.as_u64() as usize)))
            }
            ExprUnary(op, ref operand) => {
                self.expr(operand).and_then(|o| {
                    match op {
                        UnNot => constant_not(o),
                        UnNeg => constant_negate(o),
                        UnDeref => Some(o),
                    }
                })
            }
            ExprBinary(op, ref left, ref right) => self.binop(op, left, right),
            // TODO: add other expressions
            _ => None,
        }
    }

    /// create `Some(Vec![..])` of all constants, unless there is any
    /// non-constant part
    fn multi<E: Deref<Target = Expr> + Sized>(&mut self, vec: &[E]) -> Option<Vec<Constant>> {
        vec.iter()
           .map(|elem| self.expr(elem))
           .collect::<Option<_>>()
    }

    /// lookup a possibly constant expression from a ExprPath
    fn fetch_path(&mut self, e: &Expr) -> Option<Constant> {
        if let Some(lcx) = self.lcx {
            let mut maybe_id = None;
            if let Some(&PathResolution { base_def: Def::Const(id), .. }) = lcx.tcx.def_map.borrow().get(&e.id) {
                maybe_id = Some(id);
            }
            // separate if lets to avoid double borrowing the def_map
            if let Some(id) = maybe_id {
                if let Some((const_expr, _ty)) = lookup_const_by_id(lcx.tcx, id, None) {
                    let ret = self.expr(const_expr);
                    if ret.is_some() {
                        self.needed_resolution = true;
                    }
                    return ret;
                }
            }
        }
        None
    }

    /// A block can only yield a constant if it only has one constant expression
    fn block(&mut self, block: &Block) -> Option<Constant> {
        if block.stmts.is_empty() {
            block.expr.as_ref().and_then(|ref b| self.expr(b))
        } else {
            None
        }
    }

    fn ifthenelse(&mut self, cond: &Expr, then: &Block, otherwise: &Option<P<Expr>>) -> Option<Constant> {
        if let Some(Constant::Bool(b)) = self.expr(cond) {
            if b {
                self.block(then)
            } else {
                otherwise.as_ref().and_then(|expr| self.expr(expr))
            }
        } else {
            None
        }
    }

    fn binop(&mut self, op: BinOp, left: &Expr, right: &Expr) -> Option<Constant> {
        let l = if let Some(l) = self.expr(left) {
            l
        } else {
            return None;
        };
        let r = self.expr(right);
        match (op.node, l, r) {
            (BiAdd, Constant::Int(l), Some(Constant::Int(r))) => (l + r).ok().map(Constant::Int),
            (BiSub, Constant::Int(l), Some(Constant::Int(r))) => (l - r).ok().map(Constant::Int),
            (BiMul, Constant::Int(l), Some(Constant::Int(r))) => (l * r).ok().map(Constant::Int),
            (BiDiv, Constant::Int(l), Some(Constant::Int(r))) => (l / r).ok().map(Constant::Int),
            (BiRem, Constant::Int(l), Some(Constant::Int(r))) => (l % r).ok().map(Constant::Int),
            (BiAnd, Constant::Bool(false), _) => Some(Constant::Bool(false)),
            (BiOr, Constant::Bool(true), _) => Some(Constant::Bool(true)),
            (BiAnd, Constant::Bool(true), Some(r)) |
            (BiOr, Constant::Bool(false), Some(r)) => Some(r),
            (BiBitXor, Constant::Bool(l), Some(Constant::Bool(r))) => Some(Constant::Bool(l ^ r)),
            (BiBitXor, Constant::Int(l), Some(Constant::Int(r))) => (l ^ r).ok().map(Constant::Int),
            (BiBitAnd, Constant::Bool(l), Some(Constant::Bool(r))) => Some(Constant::Bool(l & r)),
            (BiBitAnd, Constant::Int(l), Some(Constant::Int(r))) => (l & r).ok().map(Constant::Int),
            (BiBitOr, Constant::Bool(l), Some(Constant::Bool(r))) => Some(Constant::Bool(l | r)),
            (BiBitOr, Constant::Int(l), Some(Constant::Int(r))) => (l | r).ok().map(Constant::Int),
            (BiShl, Constant::Int(l), Some(Constant::Int(r))) => (l << r).ok().map(Constant::Int),
            (BiShr, Constant::Int(l), Some(Constant::Int(r))) => (l >> r).ok().map(Constant::Int),
            (BiEq, Constant::Int(l), Some(Constant::Int(r))) => Some(Constant::Bool(l == r)),
            (BiNe, Constant::Int(l), Some(Constant::Int(r))) => Some(Constant::Bool(l != r)),
            (BiLt, Constant::Int(l), Some(Constant::Int(r))) => Some(Constant::Bool(l < r)),
            (BiLe, Constant::Int(l), Some(Constant::Int(r))) => Some(Constant::Bool(l <= r)),
            (BiGe, Constant::Int(l), Some(Constant::Int(r))) => Some(Constant::Bool(l >= r)),
            (BiGt, Constant::Int(l), Some(Constant::Int(r))) => Some(Constant::Bool(l > r)),
            _ => None,
        }
    }


    fn binop_apply<F>(&mut self, left: &Expr, right: &Expr, op: F) -> Option<Constant>
        where F: Fn(Constant, Constant) -> Option<Constant>
    {
        if let (Some(lc), Some(rc)) = (self.expr(left), self.expr(right)) {
            op(lc, rc)
        } else {
            None
        }
    }
}
