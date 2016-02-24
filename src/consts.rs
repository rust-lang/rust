#![allow(cast_possible_truncation)]

use rustc::lint::LateContext;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::{Def, PathResolution};
use rustc_front::hir::*;
use std::cmp::Ordering::{self, Greater, Less, Equal};
use std::cmp::PartialOrd;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::rc::Rc;
use syntax::ast::{FloatTy, LitIntType, LitKind, StrStyle, UintTy};
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

#[derive(Copy, Eq, Debug, Clone, PartialEq, Hash)]
pub enum Sign {
    Plus,
    Minus,
}

/// a Lit_-like enum to fold constant `Expr`s into
#[derive(Debug, Clone)]
pub enum Constant {
    /// a String "abc"
    Str(String, StrStyle),
    /// a Binary String b"abc"
    Binary(Rc<Vec<u8>>),
    /// a single byte b'a'
    Byte(u8),
    /// a single char 'a'
    Char(char),
    /// an integer, third argument is whether the value is negated
    Int(u64, LitIntType, Sign),
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
        if let Constant::Int(val, _, _) = *self {
            val // TODO we may want to check the sign if any
        } else {
            panic!("Could not convert a {:?} to u64", self);
        }
    }

    /// convert this constant to a f64, if possible
    #[allow(cast_precision_loss)]
    pub fn as_float(&self) -> Option<f64> {
        match *self {
            Constant::Byte(b) => Some(b as f64),
            Constant::Float(ref s, _) => s.parse().ok(),
            Constant::Int(i, _, Sign::Minus) => Some(-(i as f64)),
            Constant::Int(i, _, Sign::Plus) => Some(i as f64),
            _ => None,
        }
    }
}

impl PartialEq for Constant {
    fn eq(&self, other: &Constant) -> bool {
        match (self, other) {
            (&Constant::Str(ref ls, ref lsty), &Constant::Str(ref rs, ref rsty)) => ls == rs && lsty == rsty,
            (&Constant::Binary(ref l), &Constant::Binary(ref r)) => l == r,
            (&Constant::Byte(l), &Constant::Byte(r)) => l == r,
            (&Constant::Char(l), &Constant::Char(r)) => l == r,
            (&Constant::Int(0, _, _), &Constant::Int(0, _, _)) => true,
            (&Constant::Int(lv, _, lneg), &Constant::Int(rv, _, rneg)) => lv == rv && lneg == rneg,
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
            Constant::Byte(u) => {
                u.hash(state);
            }
            Constant::Char(c) => {
                c.hash(state);
            }
            Constant::Int(u, _, t) => {
                u.hash(state);
                t.hash(state);
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
            Constant::Vec(ref v) | Constant::Tuple(ref v) => {
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
            (&Constant::Str(ref ls, ref lsty), &Constant::Str(ref rs, ref rsty)) => {
                if lsty != rsty {
                    None
                } else {
                    Some(ls.cmp(rs))
                }
            }
            (&Constant::Byte(ref l), &Constant::Byte(ref r)) => Some(l.cmp(r)),
            (&Constant::Char(ref l), &Constant::Char(ref r)) => Some(l.cmp(r)),
            (&Constant::Int(0, _, _), &Constant::Int(0, _, _)) => Some(Equal),
            (&Constant::Int(ref lv, _, Sign::Plus), &Constant::Int(ref rv, _, Sign::Plus)) => Some(lv.cmp(rv)),
            (&Constant::Int(ref lv, _, Sign::Minus), &Constant::Int(ref rv, _, Sign::Minus)) => Some(rv.cmp(lv)),
            (&Constant::Int(_, _, Sign::Minus), &Constant::Int(_, _, Sign::Plus)) => Some(Less),
            (&Constant::Int(_, _, Sign::Plus), &Constant::Int(_, _, Sign::Minus)) => Some(Greater),
            (&Constant::Float(ref ls, _), &Constant::Float(ref rs, _)) => {
                match (ls.parse::<f64>(), rs.parse::<f64>()) {
                    (Ok(ref l), Ok(ref r)) => l.partial_cmp(r),
                    _ => None,
                }
            }
            (&Constant::Bool(ref l), &Constant::Bool(ref r)) => Some(l.cmp(r)),
            (&Constant::Vec(ref l), &Constant::Vec(ref r)) => l.partial_cmp(&r),
            (&Constant::Repeat(ref lv, ref ls), &Constant::Repeat(ref rv, ref rs)) => {
                match lv.partial_cmp(rv) {
                    Some(Equal) => Some(ls.cmp(rs)),
                    x => x,
                }
            }
            (&Constant::Tuple(ref l), &Constant::Tuple(ref r)) => l.partial_cmp(r),
            _ => None, //TODO: Are there any useful inter-type orderings?
        }
    }
}

fn lit_to_constant(lit: &LitKind) -> Constant {
    match *lit {
        LitKind::Str(ref is, style) => Constant::Str(is.to_string(), style),
        LitKind::Byte(b) => Constant::Byte(b),
        LitKind::ByteStr(ref s) => Constant::Binary(s.clone()),
        LitKind::Char(c) => Constant::Char(c),
        LitKind::Int(value, ty) => Constant::Int(value, ty, Sign::Plus),
        LitKind::Float(ref is, ty) => Constant::Float(is.to_string(), ty.into()),
        LitKind::FloatUnsuffixed(ref is) => Constant::Float(is.to_string(), FloatWidth::Any),
        LitKind::Bool(b) => Constant::Bool(b),
    }
}

fn constant_not(o: Constant) -> Option<Constant> {
    use syntax::ast::LitIntType::*;
    use self::Constant::*;
    match o {
        Bool(b) => Some(Bool(!b)),
        Int(value, LitIntType::Signed(ity), Sign::Plus) if value != ::std::u64::MAX => {
            Some(Int(value + 1, LitIntType::Signed(ity), Sign::Minus))
        }
        Int(0, LitIntType::Signed(ity), Sign::Minus) => Some(Int(1, LitIntType::Signed(ity), Sign::Minus)),
        Int(value, LitIntType::Signed(ity), Sign::Minus) => Some(Int(value - 1, LitIntType::Signed(ity), Sign::Plus)),
        Int(value, LitIntType::Unsigned(ity), Sign::Plus) => {
            let mask = match ity {
                UintTy::U8 => ::std::u8::MAX as u64,
                UintTy::U16 => ::std::u16::MAX as u64,
                UintTy::U32 => ::std::u32::MAX as u64,
                UintTy::U64 => ::std::u64::MAX,
                UintTy::Us => {
                    return None;
                }  // refuse to guess
            };
            Some(Int(!value & mask, LitIntType::Unsigned(ity), Sign::Plus))
        }
        _ => None,
    }
}

fn constant_negate(o: Constant) -> Option<Constant> {
    use syntax::ast::LitIntType::*;
    use self::Constant::*;
    match o {
        Int(value, LitIntType::Signed(ity), sign) => Some(Int(value, LitIntType::Signed(ity), neg_sign(sign))),
        Int(value, LitIntType::Unsuffixed, sign) => Some(Int(value, LitIntType::Unsuffixed, neg_sign(sign))),
        Float(is, ty) => Some(Float(neg_float_str(is), ty)),
        _ => None,
    }
}

fn neg_sign(s: Sign) -> Sign {
    match s {
        Sign::Plus => Sign::Minus,
        Sign::Minus => Sign::Plus,
    }
}

fn neg_float_str(s: String) -> String {
    if s.starts_with('-') {
        s[1..].to_owned()
    } else {
        format!("-{}", s)
    }
}

fn unify_int_type(l: LitIntType, r: LitIntType) -> Option<LitIntType> {
    use syntax::ast::LitIntType::*;
    match (l, r) {
        (Signed(lty), Signed(rty)) => {
            if lty == rty {
                Some(LitIntType::Signed(lty))
            } else {
                None
            }
        }
        (Unsigned(lty), Unsigned(rty)) => {
            if lty == rty {
                Some(LitIntType::Unsigned(lty))
            } else {
                None
            }
        }
        (Unsuffixed, Unsuffixed) => Some(Unsuffixed),
        (Signed(lty), Unsuffixed) => Some(Signed(lty)),
        (Unsigned(lty), Unsuffixed) => Some(Unsigned(lty)),
        (Unsuffixed, Signed(rty)) => Some(Signed(rty)),
        (Unsuffixed, Unsigned(rty)) => Some(Unsigned(rty)),
        _ => None,
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
            if let Some(&PathResolution { base_def: Def::Const(id), ..}) = lcx.tcx.def_map.borrow().get(&e.id) {
                maybe_id = Some(id);
            }
            // separate if lets to avoid double borrowing the def_map
            if let Some(id) = maybe_id {
                if let Some(const_expr) = lookup_const_by_id(lcx.tcx, id, None, None) {
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
        match op.node {
            BiAdd => {
                self.binop_apply(left, right, |l, r| {
                    match (l, r) {
                        (Constant::Byte(l8), Constant::Byte(r8)) => l8.checked_add(r8).map(Constant::Byte),
                        (Constant::Int(l64, lty, lsign), Constant::Int(r64, rty, rsign)) => {
                            add_ints(l64, r64, lty, rty, lsign, rsign)
                        }
                        // TODO: float (would need bignum library?)
                        _ => None,
                    }
                })
            }
            BiSub => {
                self.binop_apply(left, right, |l, r| {
                    match (l, r) {
                        (Constant::Byte(l8), Constant::Byte(r8)) => {
                            if r8 > l8 {
                                None
                            } else {
                                Some(Constant::Byte(l8 - r8))
                            }
                        }
                        (Constant::Int(l64, lty, lsign), Constant::Int(r64, rty, rsign)) => {
                            add_ints(l64, r64, lty, rty, lsign, neg_sign(rsign))
                        }
                        _ => None,
                    }
                })
            }
            BiMul => self.divmul(left, right, u64::checked_mul),
            BiDiv => self.divmul(left, right, u64::checked_div),
            // BiRem,
            BiAnd => self.short_circuit(left, right, false),
            BiOr => self.short_circuit(left, right, true),
            BiBitXor => self.bitop(left, right, |x, y| x ^ y),
            BiBitAnd => self.bitop(left, right, |x, y| x & y),
            BiBitOr => self.bitop(left, right, |x, y| (x | y)),
            BiShl => self.bitop(left, right, |x, y| x << y),
            BiShr => self.bitop(left, right, |x, y| x >> y),
            BiEq => self.binop_apply(left, right, |l, r| Some(Constant::Bool(l == r))),
            BiNe => self.binop_apply(left, right, |l, r| Some(Constant::Bool(l != r))),
            BiLt => self.cmp(left, right, Less, true),
            BiLe => self.cmp(left, right, Greater, false),
            BiGe => self.cmp(left, right, Less, false),
            BiGt => self.cmp(left, right, Greater, true),
            _ => None,
        }
    }

    fn divmul<F>(&mut self, left: &Expr, right: &Expr, f: F) -> Option<Constant>
        where F: Fn(u64, u64) -> Option<u64>
    {
        self.binop_apply(left, right, |l, r| {
            match (l, r) {
                (Constant::Int(l64, lty, lsign), Constant::Int(r64, rty, rsign)) => {
                    f(l64, r64).and_then(|value| {
                        let sign = if lsign == rsign {
                            Sign::Plus
                        } else {
                            Sign::Minus
                        };
                        unify_int_type(lty, rty).map(|ty| Constant::Int(value, ty, sign))
                    })
                }
                _ => None,
            }
        })
    }

    fn bitop<F>(&mut self, left: &Expr, right: &Expr, f: F) -> Option<Constant>
        where F: Fn(u64, u64) -> u64
    {
        self.binop_apply(left, right, |l, r| {
            match (l, r) {
                (Constant::Bool(l), Constant::Bool(r)) => Some(Constant::Bool(f(l as u64, r as u64) != 0)),
                (Constant::Byte(l8), Constant::Byte(r8)) => Some(Constant::Byte(f(l8 as u64, r8 as u64) as u8)),
                (Constant::Int(l, lty, lsign), Constant::Int(r, rty, rsign)) => {
                    if lsign == Sign::Plus && rsign == Sign::Plus {
                        unify_int_type(lty, rty).map(|ty| Constant::Int(f(l, r), ty, Sign::Plus))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
    }

    fn cmp(&mut self, left: &Expr, right: &Expr, ordering: Ordering, b: bool) -> Option<Constant> {
        self.binop_apply(left,
                         right,
                         |l, r| l.partial_cmp(&r).map(|o| Constant::Bool(b == (o == ordering))))
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

    fn short_circuit(&mut self, left: &Expr, right: &Expr, b: bool) -> Option<Constant> {
        self.expr(left).and_then(|left| {
            if let Constant::Bool(lbool) = left {
                if lbool == b {
                    Some(left)
                } else {
                    self.expr(right).and_then(|right| {
                        if let Constant::Bool(_) = right {
                            Some(right)
                        } else {
                            None
                        }
                    })
                }
            } else {
                None
            }
        })
    }
}

fn add_ints(l64: u64, r64: u64, lty: LitIntType, rty: LitIntType, lsign: Sign, rsign: Sign) -> Option<Constant> {
    let ty = if let Some(ty) = unify_int_type(lty, rty) {
        ty
    } else {
        return None;
    };

    match (lsign, rsign) {
        (Sign::Plus, Sign::Plus) => l64.checked_add(r64).map(|v| Constant::Int(v, ty, Sign::Plus)),
        (Sign::Plus, Sign::Minus) => {
            if r64 > l64 {
                Some(Constant::Int(r64 - l64, ty, Sign::Minus))
            } else {
                Some(Constant::Int(l64 - r64, ty, Sign::Plus))
            }
        }
        (Sign::Minus, Sign::Minus) => l64.checked_add(r64).map(|v| Constant::Int(v, ty, Sign::Minus)),
        (Sign::Minus, Sign::Plus) => {
            if l64 > r64 {
                Some(Constant::Int(l64 - r64, ty, Sign::Minus))
            } else {
                Some(Constant::Int(r64 - l64, ty, Sign::Plus))
            }
        }
    }
}
