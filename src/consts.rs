#![allow(cast_possible_truncation)]

use rustc::lint::LateContext;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::PathResolution;
use rustc::middle::def::Def::*;
use rustc_front::hir::*;
use syntax::ptr::P;
use std::char;
use std::cmp::PartialOrd;
use std::cmp::Ordering::{self, Greater, Less, Equal};
use std::rc::Rc;
use std::ops::Deref;
use std::fmt;
use self::Constant::*;
use self::FloatWidth::*;

use syntax::ast::Lit_::*;
use syntax::ast::Lit_;
use syntax::ast::LitIntType::*;
use syntax::ast::LitIntType;
use syntax::ast::{UintTy, FloatTy, StrStyle};
use syntax::ast::FloatTy::*;
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
            TyF32 => Fw32,
            TyF64 => Fw64,
        }
    }
}

/// a Lit_-like enum to fold constant `Expr`s into
#[derive(Eq, Debug, Clone)]
pub enum Constant {
    /// a String "abc"
    ConstantStr(String, StrStyle),
    /// a Binary String b"abc"
    ConstantBinary(Rc<Vec<u8>>),
    /// a single byte b'a'
    ConstantByte(u8),
    /// a single char 'a'
    ConstantChar(char),
    /// an integer
    ConstantInt(u64, LitIntType),
    /// a float with given type
    ConstantFloat(String, FloatWidth),
    /// true or false
    ConstantBool(bool),
    /// an array of constants
    ConstantVec(Vec<Constant>),
    /// also an array, but with only one constant, repeated N times
    ConstantRepeat(Box<Constant>, usize),
    /// a tuple of constants
    ConstantTuple(Vec<Constant>),
}

impl Constant {
    /// convert to u64 if possible
    ///
    /// # panics
    ///
    /// if the constant could not be converted to u64 losslessly
    fn as_u64(&self) -> u64 {
        if let ConstantInt(val, _) = *self {
            val // TODO we may want to check the sign if any
        } else {
            panic!("Could not convert a {:?} to u64", self);
        }
    }

    /// convert this constant to a f64, if possible
    #[allow(cast_precision_loss)]
    pub fn as_float(&self) -> Option<f64> {
        match *self {
            ConstantByte(b) => Some(b as f64),
            ConstantFloat(ref s, _) => s.parse().ok(),
            ConstantInt(i, ty) => {
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
            (&ConstantStr(ref ls, ref lsty), &ConstantStr(ref rs, ref rsty)) => ls == rs && lsty == rsty,
            (&ConstantBinary(ref l), &ConstantBinary(ref r)) => l == r,
            (&ConstantByte(l), &ConstantByte(r)) => l == r,
            (&ConstantChar(l), &ConstantChar(r)) => l == r,
            (&ConstantInt(lv, lty), &ConstantInt(rv, rty)) => {
                lv == rv && (is_negative(lty) & (lv != 0)) == (is_negative(rty) & (rv != 0))
            }
            (&ConstantFloat(ref ls, lw), &ConstantFloat(ref rs, rw)) => {
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
            (&ConstantBool(l), &ConstantBool(r)) => l == r,
            (&ConstantVec(ref l), &ConstantVec(ref r)) => l == r,
            (&ConstantRepeat(ref lv, ref ls), &ConstantRepeat(ref rv, ref rs)) => ls == rs && lv == rv,
            (&ConstantTuple(ref l), &ConstantTuple(ref r)) => l == r,
            _ => false, //TODO: Are there inter-type equalities?
        }
    }
}

impl PartialOrd for Constant {
    fn partial_cmp(&self, other: &Constant) -> Option<Ordering> {
        match (self, other) {
            (&ConstantStr(ref ls, ref lsty), &ConstantStr(ref rs, ref rsty)) => {
                if lsty != rsty {
                    None
                } else {
                    Some(ls.cmp(rs))
                }
            }
            (&ConstantByte(ref l), &ConstantByte(ref r)) => Some(l.cmp(r)),
            (&ConstantChar(ref l), &ConstantChar(ref r)) => Some(l.cmp(r)),
            (&ConstantInt(ref lv, lty), &ConstantInt(ref rv, rty)) => {
                Some(match (is_negative(lty) && *lv != 0, is_negative(rty) && *rv != 0) {
                    (true, true) => rv.cmp(lv),
                    (false, false) => lv.cmp(rv),
                    (true, false) => Less,
                    (false, true) => Greater,
                })
            }
            (&ConstantFloat(ref ls, lw), &ConstantFloat(ref rs, rw)) => {
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
            (&ConstantBool(ref l), &ConstantBool(ref r)) => Some(l.cmp(r)),
            (&ConstantVec(ref l), &ConstantVec(ref r)) => l.partial_cmp(&r),
            (&ConstantRepeat(ref lv, ref ls), &ConstantRepeat(ref rv, ref rs)) => {
                match lv.partial_cmp(rv) {
                    Some(Equal) => Some(ls.cmp(rs)),
                    x => x,
                }
            }
            (&ConstantTuple(ref l), &ConstantTuple(ref r)) => l.partial_cmp(r),
            _ => None, //TODO: Are there any useful inter-type orderings?
        }
    }
}

fn format_byte(fmt: &mut fmt::Formatter, b: u8) -> fmt::Result {
    if b == b'\\' {
        write!(fmt, "\\\\")
    } else if 0x20 <= b && b <= 0x7e {
        write!(fmt, "{}", char::from_u32(b as u32).expect("all u8 are valid char"))
    } else if b == 0x0a {
        write!(fmt, "\\n")
    } else if b == 0x0d {
        write!(fmt, "\\r")
    } else {
        write!(fmt, "\\x{:02x}", b)
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ConstantStr(ref s, _) => write!(fmt, "{:?}", s),
            ConstantByte(ref b) => {
                write!(fmt, "b'")
                    .and_then(|_| format_byte(fmt, *b))
                    .and_then(|_| write!(fmt, "'"))
            }
            ConstantBinary(ref bs) => {
                try!(write!(fmt, "b\""));
                for b in bs.iter() {
                    try!(format_byte(fmt, *b));
                }
                write!(fmt, "\"")
            }
            ConstantChar(ref c) => write!(fmt, "'{}'", c),
            ConstantInt(ref i, ref ity) => {
                let (sign, suffix) = match *ity {
                    LitIntType::SignedIntLit(ref sity, ref sign) => {
                        (if let Sign::Minus = *sign {
                            "-"
                        } else {
                            ""
                        },
                         sity.ty_to_string())
                    }
                    LitIntType::UnsignedIntLit(ref uity) => ("", uity.ty_to_string()),
                    LitIntType::UnsuffixedIntLit(ref sign) => {
                        (if let Sign::Minus = *sign {
                            "-"
                        } else {
                            ""
                        },
                         "".into())
                    }
                };
                write!(fmt, "{}{}{}", sign, i, suffix)
            }
            ConstantFloat(ref s, ref fw) => {
                let suffix = match *fw {
                    FloatWidth::Fw32 => "f32",
                    FloatWidth::Fw64 => "f64",
                    FloatWidth::FwAny => "",
                };
                write!(fmt, "{}{}", s, suffix)
            }
            ConstantBool(ref b) => write!(fmt, "{}", b),
            ConstantRepeat(ref c, ref n) => write!(fmt, "[{}; {}]", c, n),
            ConstantVec(ref v) => {
                write!(fmt,
                       "[{}]",
                       v.iter()
                        .map(|i| format!("{}", i))
                        .collect::<Vec<_>>()
                        .join(", "))
            }
            ConstantTuple(ref t) => {
                write!(fmt,
                       "({})",
                       t.iter()
                        .map(|i| format!("{}", i))
                        .collect::<Vec<_>>()
                        .join(", "))
            }
        }
    }
}


fn lit_to_constant(lit: &Lit_) -> Constant {
    match *lit {
        LitStr(ref is, style) => ConstantStr(is.to_string(), style),
        LitByte(b) => ConstantByte(b),
        LitByteStr(ref s) => ConstantBinary(s.clone()),
        LitChar(c) => ConstantChar(c),
        LitInt(value, ty) => ConstantInt(value, ty),
        LitFloat(ref is, ty) => ConstantFloat(is.to_string(), ty.into()),
        LitFloatUnsuffixed(ref is) => ConstantFloat(is.to_string(), FwAny),
        LitBool(b) => ConstantBool(b),
    }
}

fn constant_not(o: Constant) -> Option<Constant> {
    Some(match o {
        ConstantBool(b) => ConstantBool(!b),
        ConstantInt(value, ty) => {
            let (nvalue, nty) = match ty {
                SignedIntLit(ity, Plus) => {
                    if value == ::std::u64::MAX {
                        return None;
                    }
                    (value + 1, SignedIntLit(ity, Minus))
                }
                SignedIntLit(ity, Minus) => {
                    if value == 0 {
                        (1, SignedIntLit(ity, Minus))
                    } else {
                        (value - 1, SignedIntLit(ity, Plus))
                    }
                }
                UnsignedIntLit(ity) => {
                    let mask = match ity {
                        UintTy::TyU8 => ::std::u8::MAX as u64,
                        UintTy::TyU16 => ::std::u16::MAX as u64,
                        UintTy::TyU32 => ::std::u32::MAX as u64,
                        UintTy::TyU64 => ::std::u64::MAX,
                        UintTy::TyUs => {
                            return None;
                        }  // refuse to guess
                    };
                    (!value & mask, UnsignedIntLit(ity))
                }
                UnsuffixedIntLit(_) => {
                    return None;
                }  // refuse to guess
            };
            ConstantInt(nvalue, nty)
        }
        _ => {
            return None;
        }
    })
}

fn constant_negate(o: Constant) -> Option<Constant> {
    Some(match o {
        ConstantInt(value, ty) => {
            ConstantInt(value,
                        match ty {
                            SignedIntLit(ity, sign) => SignedIntLit(ity, neg_sign(sign)),
                            UnsuffixedIntLit(sign) => UnsuffixedIntLit(neg_sign(sign)),
                            _ => {
                                return None;
                            }
                        })
        }
        ConstantFloat(is, ty) => ConstantFloat(neg_float_str(is), ty),
        _ => {
            return None;
        }
    })
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
        SignedIntLit(_, sign) | UnsuffixedIntLit(sign) => sign == Minus,
        UnsignedIntLit(_) => false,
    }
}

fn unify_int_type(l: LitIntType, r: LitIntType, s: Sign) -> Option<LitIntType> {
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
        unify_int_type(nty, pty, Minus).map(|ty| ConstantInt(neg - pos, ty))
    } else {
        unify_int_type(nty, pty, Plus).map(|ty| ConstantInt(pos - neg, ty))
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
        .and_then(|ty| l.checked_sub(r).map(|v| ConstantInt(v, ty)))
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
            ExprVec(ref vec) => self.multi(vec).map(ConstantVec),
            ExprTup(ref tup) => self.multi(tup).map(ConstantTuple),
            ExprRepeat(ref value, ref number) => {
                self.binop_apply(value, number, |v, n| Some(ConstantRepeat(Box::new(v), n.as_u64() as usize)))
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
            if let Some(&PathResolution { base_def: DefConst(id), ..}) = lcx.tcx.def_map.borrow().get(&e.id) {
                maybe_id = Some(id);
            }
            // separate if lets to avoid doubleborrowing the defmap
            if let Some(id) = maybe_id {
                if let Some(const_expr) = lookup_const_by_id(lcx.tcx, id, None) {
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
        if let Some(ConstantBool(b)) = self.expr(cond) {
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
                        (ConstantByte(l8), ConstantByte(r8)) => l8.checked_add(r8).map(ConstantByte),
                        (ConstantInt(l64, lty), ConstantInt(r64, rty)) => {
                            let (ln, rn) = (is_negative(lty), is_negative(rty));
                            if ln == rn {
                                unify_int_type(lty,
                                               rty,
                                               if ln {
                                                   Minus
                                               } else {
                                                   Plus
                                               })
                                    .and_then(|ty| l64.checked_add(r64).map(|v| ConstantInt(v, ty)))
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
                        (ConstantByte(l8), ConstantByte(r8)) => {
                            if r8 > l8 {
                                None
                            } else {
                                Some(ConstantByte(l8 - r8))
                            }
                        }
                        (ConstantInt(l64, lty), ConstantInt(r64, rty)) => {
                            match (is_negative(lty), is_negative(rty)) {
                                (false, false) => sub_int(l64, lty, r64, rty, r64 > l64),
                                (true, true) => sub_int(l64, lty, r64, rty, l64 > r64),
                                (true, false) => {
                                    unify_int_type(lty, rty, Minus)
                                        .and_then(|ty| l64.checked_add(r64).map(|v| ConstantInt(v, ty)))
                                }
                                (false, true) => {
                                    unify_int_type(lty, rty, Plus)
                                        .and_then(|ty| l64.checked_add(r64).map(|v| ConstantInt(v, ty)))
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
            BiEq => self.binop_apply(left, right, |l, r| Some(ConstantBool(l == r))),
            BiNe => self.binop_apply(left, right, |l, r| Some(ConstantBool(l != r))),
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
                (ConstantInt(l64, lty), ConstantInt(r64, rty)) => {
                    f(l64, r64).and_then(|value| {
                        unify_int_type(lty,
                                       rty,
                                       if is_negative(lty) == is_negative(rty) {
                                           Plus
                                       } else {
                                           Minus
                                       })
                            .map(|ty| ConstantInt(value, ty))
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
                (ConstantBool(l), ConstantBool(r)) => Some(ConstantBool(f(l as u64, r as u64) != 0)),
                (ConstantByte(l8), ConstantByte(r8)) => Some(ConstantByte(f(l8 as u64, r8 as u64) as u8)),
                (ConstantInt(l, lty), ConstantInt(r, rty)) => {
                    unify_int_type(lty, rty, Plus).map(|ty| ConstantInt(f(l, r), ty))
                }
                _ => None,
            }
        })
    }

    fn cmp(&mut self, left: &Expr, right: &Expr, ordering: Ordering, b: bool) -> Option<Constant> {
        self.binop_apply(left,
                         right,
                         |l, r| l.partial_cmp(&r).map(|o| ConstantBool(b == (o == ordering))))
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
            if let ConstantBool(lbool) = left {
                if lbool == b {
                    Some(left)
                } else {
                    self.expr(right).and_then(|right| {
                        if let ConstantBool(_) = right {
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
