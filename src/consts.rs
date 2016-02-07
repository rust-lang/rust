#![allow(cast_possible_truncation)]

use rustc::lint::LateContext;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::PathResolution;
use rustc::middle::def::Def;
use rustc_front::hir::*;
use syntax::ptr::P;
use std::cmp::PartialOrd;
use std::cmp::Ordering::{self, Greater, Less, Equal};
use std::rc::Rc;
use std::ops::Deref;

use syntax::ast::Lit_;
use syntax::ast::LitIntType;
use syntax::ast::{UintTy, FloatTy, StrStyle};
use syntax::ast::Sign::{self, Plus, Minus};


#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum FloatWidth {
    Fw32,
    Fw64,
    FwAny,
}

impl From<FloatTy> for FloatWidth {
    fn from(ty: FloatTy) -> FloatWidth {
        match ty {
            FloatTy::TyF32 => FloatWidth::Fw32,
            FloatTy::TyF64 => FloatWidth::Fw64,
        }
    }
}

/// a Lit_-like enum to fold constant `Expr`s into
#[derive(Eq, Debug, Clone)]
pub enum Constant {
    /// a String "abc"
    Str(String, StrStyle),
    /// a Binary String b"abc"
    Binary(Rc<Vec<u8>>),
    /// a single byte b'a'
    Byte(u8),
    /// a single char 'a'
    Char(char),
    /// an integer
    Int(u64, LitIntType),
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
        if let Constant::Int(val, _) = *self {
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
            Constant::Int(i, ty) => {
                Some(if is_negative(ty) {
                    -(i as f64)
                } else {
                    i as f64
                })
            }
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
            (&Constant::Int(lv, lty), &Constant::Int(rv, rty)) => {
                lv == rv && (is_negative(lty) & (lv != 0)) == (is_negative(rty) & (rv != 0))
            }
            (&Constant::Float(ref ls, lw), &Constant::Float(ref rs, rw)) => {
                use self::FloatWidth::*;
                if match (lw, rw) {
                    (FwAny, _) | (_, FwAny) | (Fw32, Fw32) | (Fw64, Fw64) => true,
                    _ => false,
                } {
                    match (ls.parse::<f64>(), rs.parse::<f64>()) {
                        (Ok(l), Ok(r)) => l.eq(&r),
                        _ => false,
                    }
                } else {
                    false
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
            (&Constant::Int(ref lv, lty), &Constant::Int(ref rv, rty)) => {
                Some(match (is_negative(lty) && *lv != 0, is_negative(rty) && *rv != 0) {
                    (true, true) => rv.cmp(lv),
                    (false, false) => lv.cmp(rv),
                    (true, false) => Less,
                    (false, true) => Greater,
                })
            }
            (&Constant::Float(ref ls, lw), &Constant::Float(ref rs, rw)) => {
                use self::FloatWidth::*;
                if match (lw, rw) {
                    (FwAny, _) | (_, FwAny) | (Fw32, Fw32) | (Fw64, Fw64) => true,
                    _ => false,
                } {
                    match (ls.parse::<f64>(), rs.parse::<f64>()) {
                        (Ok(ref l), Ok(ref r)) => l.partial_cmp(r),
                        _ => None,
                    }
                } else {
                    None
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

fn lit_to_constant(lit: &Lit_) -> Constant {
    match *lit {
        Lit_::LitStr(ref is, style) => Constant::Str(is.to_string(), style),
        Lit_::LitByte(b) => Constant::Byte(b),
        Lit_::LitByteStr(ref s) => Constant::Binary(s.clone()),
        Lit_::LitChar(c) => Constant::Char(c),
        Lit_::LitInt(value, ty) => Constant::Int(value, ty),
        Lit_::LitFloat(ref is, ty) => Constant::Float(is.to_string(), ty.into()),
        Lit_::LitFloatUnsuffixed(ref is) => Constant::Float(is.to_string(), FloatWidth::FwAny),
        Lit_::LitBool(b) => Constant::Bool(b),
    }
}

fn constant_not(o: Constant) -> Option<Constant> {
    use syntax::ast::LitIntType::*;
    use self::Constant::*;
    match o {
        Bool(b) => Some(Bool(!b)),
        Int(::std::u64::MAX, SignedIntLit(_, Plus)) => None,
        Int(value, SignedIntLit(ity, Plus)) => Some(Int(value + 1, SignedIntLit(ity, Minus))),
        Int(0, SignedIntLit(ity, Minus)) => Some(Int(1, SignedIntLit(ity, Minus))),
        Int(value, SignedIntLit(ity, Minus)) => Some(Int(value - 1, SignedIntLit(ity, Plus))),
        Int(value, UnsignedIntLit(ity)) => {
            let mask = match ity {
                UintTy::TyU8 => ::std::u8::MAX as u64,
                UintTy::TyU16 => ::std::u16::MAX as u64,
                UintTy::TyU32 => ::std::u32::MAX as u64,
                UintTy::TyU64 => ::std::u64::MAX,
                UintTy::TyUs => {
                    return None;
                }  // refuse to guess
            };
            Some(Int(!value & mask, UnsignedIntLit(ity)))
        },
        _ => None,
    }
}

fn constant_negate(o: Constant) -> Option<Constant> {
    use syntax::ast::LitIntType::*;
    use self::Constant::*;
    match o {
        Int(value, SignedIntLit(ity, sign)) => Some(Int(value, SignedIntLit(ity, neg_sign(sign)))),
        Int(value, UnsuffixedIntLit(sign)) => Some(Int(value, UnsuffixedIntLit(neg_sign(sign)))),
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

/// is the given LitIntType negative?
///
/// Examples
///
/// ```
/// assert!(is_negative(UnsuffixedIntLit(Minus)));
/// ```
pub fn is_negative(ty: LitIntType) -> bool {
    match ty {
        LitIntType::SignedIntLit(_, sign) | LitIntType::UnsuffixedIntLit(sign) => sign == Minus,
        LitIntType::UnsignedIntLit(_) => false,
    }
}

fn unify_int_type(l: LitIntType, r: LitIntType, s: Sign) -> Option<LitIntType> {
    use syntax::ast::LitIntType::*;
    match (l, r) {
        (SignedIntLit(lty, _), SignedIntLit(rty, _)) => {
            if lty == rty {
                Some(SignedIntLit(lty, s))
            } else {
                None
            }
        }
        (UnsignedIntLit(lty), UnsignedIntLit(rty)) => {
            if s == Plus && lty == rty {
                Some(UnsignedIntLit(lty))
            } else {
                None
            }
        }
        (UnsuffixedIntLit(_), UnsuffixedIntLit(_)) => Some(UnsuffixedIntLit(s)),
        (SignedIntLit(lty, _), UnsuffixedIntLit(_)) => Some(SignedIntLit(lty, s)),
        (UnsignedIntLit(lty), UnsuffixedIntLit(rs)) => {
            if rs == Plus {
                Some(UnsignedIntLit(lty))
            } else {
                None
            }
        }
        (UnsuffixedIntLit(_), SignedIntLit(rty, _)) => Some(SignedIntLit(rty, s)),
        (UnsuffixedIntLit(ls), UnsignedIntLit(rty)) => {
            if ls == Plus {
                Some(UnsignedIntLit(rty))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn add_neg_int(pos: u64, pty: LitIntType, neg: u64, nty: LitIntType) -> Option<Constant> {
    if neg > pos {
        unify_int_type(nty, pty, Minus).map(|ty| Constant::Int(neg - pos, ty))
    } else {
        unify_int_type(nty, pty, Plus).map(|ty| Constant::Int(pos - neg, ty))
    }
}

fn sub_int(l: u64, lty: LitIntType, r: u64, rty: LitIntType, neg: bool) -> Option<Constant> {
    unify_int_type(lty,
                   rty,
                   if neg {
                       Minus
                   } else {
                       Plus
                   })
        .and_then(|ty| l.checked_sub(r).map(|v| Constant::Int(v, ty)))
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
                        (Constant::Int(l64, lty), Constant::Int(r64, rty)) => {
                            let (ln, rn) = (is_negative(lty), is_negative(rty));
                            if ln == rn {
                                unify_int_type(lty,
                                               rty,
                                               if ln {
                                                   Minus
                                               } else {
                                                   Plus
                                               })
                                    .and_then(|ty| l64.checked_add(r64).map(|v| Constant::Int(v, ty)))
                            } else if ln {
                                add_neg_int(r64, rty, l64, lty)
                            } else {
                                add_neg_int(l64, lty, r64, rty)
                            }
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
                        (Constant::Int(l64, lty), Constant::Int(r64, rty)) => {
                            match (is_negative(lty), is_negative(rty)) {
                                (false, false) => sub_int(l64, lty, r64, rty, r64 > l64),
                                (true, true) => sub_int(l64, lty, r64, rty, l64 > r64),
                                (true, false) => {
                                    unify_int_type(lty, rty, Minus)
                                        .and_then(|ty| l64.checked_add(r64).map(|v| Constant::Int(v, ty)))
                                }
                                (false, true) => {
                                    unify_int_type(lty, rty, Plus)
                                        .and_then(|ty| l64.checked_add(r64).map(|v| Constant::Int(v, ty)))
                                }
                            }
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
                (Constant::Int(l64, lty), Constant::Int(r64, rty)) => {
                    f(l64, r64).and_then(|value| {
                        unify_int_type(lty,
                                       rty,
                                       if is_negative(lty) == is_negative(rty) {
                                           Plus
                                       } else {
                                           Minus
                                       })
                            .map(|ty| Constant::Int(value, ty))
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
                (Constant::Int(l, lty), Constant::Int(r, rty)) => {
                    unify_int_type(lty, rty, Plus).map(|ty| Constant::Int(f(l, r), ty))
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
