use rustc::lint::Context;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::PathResolution;
use rustc::middle::def::Def::*;
use syntax::ast::*;
use syntax::ptr::P;
use std::rc::Rc;
use std::ops::Deref;
use self::ConstantVariant::*;
use self::FloatWidth::*;

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum FloatWidth {
    Fw32,
    Fw64,
    FwAny
}

impl From<FloatTy> for FloatWidth {
    fn from(ty: FloatTy) -> FloatWidth {
        match ty {
            TyF32 => Fw32,
            TyF64 => Fw64,
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Constant {
    pub constant: ConstantVariant,
    pub needed_resolution: bool
}

impl Constant {
    pub fn new(variant: ConstantVariant) -> Constant {
        Constant { constant: variant, needed_resolution: false }
    }

    pub fn new_resolved(variant: ConstantVariant) -> Constant {
        Constant { constant: variant, needed_resolution: true }
    }

    // convert this constant to a f64, if possible
    pub fn as_float(&self) -> Option<f64> {
        match &self.constant {
            &ConstantByte(b) => Some(b as f64),
            &ConstantFloat(ref s, _) => s.parse().ok(),
            &ConstantInt(i, ty) => Some(if is_negative(ty) {
                -(i as f64) } else { i as f64 }),
            _ => None
        }
    }
}

/// a Lit_-like enum to fold constant `Expr`s into
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum ConstantVariant {
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
    ConstantRepeat(Box<ConstantVariant>, usize),
    /// a tuple of constants
    ConstantTuple(Vec<Constant>),
}

impl ConstantVariant {
    /// convert to u64 if possible
    ///
    /// # panics
    ///
    /// if the constant could not be converted to u64 losslessly
    fn as_u64(&self) -> u64 {
        if let &ConstantInt(val, _) = self {
            val // TODO we may want to check the sign if any
        } else {
            panic!("Could not convert a {:?} to u64");
        }
    }
}

/// simple constant folding: Insert an expression, get a constant or none.
pub fn constant(cx: &Context, e: &Expr) -> Option<Constant> {
    match &e.node {
        &ExprParen(ref inner) => constant(cx, inner),
        &ExprPath(_, _) => fetch_path(cx, e),
        &ExprBlock(ref block) => constant_block(cx, block),
        &ExprIf(ref cond, ref then, ref otherwise) =>
            constant_if(cx, &*cond, &*then, &*otherwise),
        &ExprLit(ref lit) => Some(lit_to_constant(&lit.node)),
        &ExprVec(ref vec) => constant_vec(cx, &vec[..]),
        &ExprTup(ref tup) => constant_tup(cx, &tup[..]),
        &ExprRepeat(ref value, ref number) =>
            constant_binop_apply(cx, value, number,|v, n|
                Some(ConstantRepeat(Box::new(v), n.as_u64() as usize))),
        &ExprUnary(op, ref operand) => constant(cx, operand).and_then(
            |o| match op {
                UnNot =>
                    if let ConstantBool(b) = o.constant {
                        Some(Constant{
                            needed_resolution: o.needed_resolution,
                            constant: ConstantBool(!b),
                        })
                    } else { None },
                UnNeg => constant_negate(o),
                UnUniq | UnDeref => Some(o),
            }),
        &ExprBinary(op, ref left, ref right) =>
            constant_binop(cx, op, left, right),
        //TODO: add other expressions
        _ => None,
    }
}

fn lit_to_constant(lit: &Lit_) -> Constant {
    match lit {
        &LitStr(ref is, style) =>
            Constant::new(ConstantStr(is.to_string(), style)),
        &LitBinary(ref blob) => Constant::new(ConstantBinary(blob.clone())),
        &LitByte(b) => Constant::new(ConstantByte(b)),
        &LitChar(c) => Constant::new(ConstantChar(c)),
        &LitInt(value, ty) => Constant::new(ConstantInt(value, ty)),
        &LitFloat(ref is, ty) => {
            Constant::new(ConstantFloat(is.to_string(), ty.into()))
        },
        &LitFloatUnsuffixed(ref is) => {
            Constant::new(ConstantFloat(is.to_string(), FwAny))
        },
        &LitBool(b) => Constant::new(ConstantBool(b)),
    }
}

/// create `Some(ConstantVec(..))` of all constants, unless there is any
/// non-constant part
fn constant_vec<E: Deref<Target=Expr> + Sized>(cx: &Context, vec: &[E]) -> Option<Constant> {
    let mut parts = Vec::new();
    let mut resolved = false;
    for opt_part in vec {
        match constant(cx, opt_part) {
            Some(p) => {
                resolved |= (&p).needed_resolution;
                parts.push(p)
            },
            None => { return None; },
        }
    }
    Some(Constant {
        constant: ConstantVec(parts),
        needed_resolution: resolved
    })
}

fn constant_tup<E: Deref<Target=Expr> + Sized>(cx: &Context, tup: &[E]) -> Option<Constant> {
    let mut parts = Vec::new();
    let mut resolved = false;
    for opt_part in tup {
        match constant(cx, opt_part) {
            Some(p) => {
                resolved |= (&p).needed_resolution;
                parts.push(p)
            },
            None => { return None; },
        }
    }
    Some(Constant {
        constant: ConstantTuple(parts),
        needed_resolution: resolved
    })
}

/// lookup a possibly constant expression from a ExprPath
fn fetch_path(cx: &Context, e: &Expr) -> Option<Constant> {
    if let Some(&PathResolution { base_def: DefConst(id), ..}) =
            cx.tcx.def_map.borrow().get(&e.id) {
        lookup_const_by_id(cx.tcx, id, None).and_then(
            |l| constant(cx, l).map(|c| Constant::new_resolved(c.constant)))
    } else { None }
}

/// A block can only yield a constant if it only has one constant expression
fn constant_block(cx: &Context, block: &Block) -> Option<Constant> {
    if block.stmts.is_empty() {
        block.expr.as_ref().and_then(|b| constant(cx, &*b))
    } else { None }
}

fn constant_if(cx: &Context, cond: &Expr, then: &Block, otherwise:
        &Option<P<Expr>>) -> Option<Constant> {
    if let Some(Constant{ constant: ConstantBool(b), needed_resolution: res }) =
            constant(cx, cond) {
        if b {
            constant_block(cx, then)
        } else {
            otherwise.as_ref().and_then(|expr| constant(cx, &*expr))
        }.map(|part|
            Constant {
                constant: part.constant,
                needed_resolution: res || part.needed_resolution,
            })
    } else { None }
}

fn constant_negate(o: Constant) -> Option<Constant> {
    Some(Constant{
        needed_resolution: o.needed_resolution,
        constant: match o.constant {
            ConstantInt(value, ty) =>
                ConstantInt(value, match ty {
                    SignedIntLit(ity, sign) =>
                        SignedIntLit(ity, neg_sign(sign)),
                    UnsuffixedIntLit(sign) => UnsuffixedIntLit(neg_sign(sign)),
                    _ => { return None; },
                }),
            ConstantFloat(is, ty) =>
                ConstantFloat(neg_float_str(is), ty),
            _ => { return None; },
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
        format!("-{}", &*s)
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
        (SignedIntLit(lty, _), SignedIntLit(rty, _)) => if lty == rty {
            Some(SignedIntLit(lty, s)) } else { None },
        (UnsignedIntLit(lty), UnsignedIntLit(rty)) =>
            if s == Plus && lty == rty {
                Some(UnsignedIntLit(lty))
            } else { None },
        (UnsuffixedIntLit(_), UnsuffixedIntLit(_)) => Some(UnsuffixedIntLit(s)),
        (SignedIntLit(lty, _), UnsuffixedIntLit(_)) => Some(SignedIntLit(lty, s)),
        (UnsignedIntLit(lty), UnsuffixedIntLit(rs)) => if rs == Plus {
            Some(UnsignedIntLit(lty)) } else { None },
        (UnsuffixedIntLit(_), SignedIntLit(rty, _)) => Some(SignedIntLit(rty, s)),
        (UnsuffixedIntLit(ls), UnsignedIntLit(rty)) => if ls == Plus {
            Some(UnsignedIntLit(rty)) } else { None },
        _ => None,
    }
}

fn constant_binop(cx: &Context, op: BinOp, left: &Expr, right: &Expr)
        -> Option<Constant> {
    match op.node {
        BiAdd => constant_binop_apply(cx, left, right, |l, r|
            match (l, r) {
                (ConstantByte(l8), ConstantByte(r8)) =>
                    l8.checked_add(r8).map(ConstantByte),
                (ConstantInt(l64, lty), ConstantInt(r64, rty)) => {
                    let (ln, rn) = (is_negative(lty), is_negative(rty));
                    if ln == rn {
                        unify_int_type(lty, rty, if ln { Minus } else { Plus })
                            .and_then(|ty| l64.checked_add(r64).map(
                                |v| ConstantInt(v, ty)))
                    } else {
                        if ln {
                            add_neg_int(r64, rty, l64, lty)
                        } else {
                            add_neg_int(l64, lty, r64, rty)
                        }
                    }
                },
                // TODO: float
                _ => None
            }),
        BiSub => constant_binop_apply(cx, left, right, |l, r|
            match (l, r) {
                (ConstantByte(l8), ConstantByte(r8)) => if r8 > l8 {
                    None } else { Some(ConstantByte(l8 - r8)) },
                (ConstantInt(l64, lty), ConstantInt(r64, rty)) => {
                    let (ln, rn) = (is_negative(lty), is_negative(rty));
                    match (ln, rn) {
                        (false, false) => sub_int(l64, lty, r64, rty, r64 > l64),
                        (true, true) => sub_int(l64, lty, r64, rty, l64 > r64),
                        (true, false) => unify_int_type(lty, rty, Minus)
                            .and_then(|ty| l64.checked_add(r64).map(
                                |v| ConstantInt(v, ty))),
                        (false, true) => unify_int_type(lty, rty, Plus)
                            .and_then(|ty| l64.checked_add(r64).map(
                                |v| ConstantInt(v, ty))),
                    }
                },
                _ => None,
            }),
        //BiMul,
        //BiDiv,
        //BiRem,
        BiAnd => constant_short_circuit(cx, left, right, false),
        BiOr => constant_short_circuit(cx, left, right, true),
        //BiBitXor,
        //BiBitAnd,
        //BiBitOr,
        //BiShl,
        //BiShr,
        //BiEq,
        //BiLt,
        //BiLe,
        //BiNe,
        //BiGe,
        //BiGt,
        _ => None,
    }
}

fn add_neg_int(pos: u64, pty: LitIntType, neg: u64, nty: LitIntType) ->
        Option<ConstantVariant> {
    if neg > pos {
        unify_int_type(nty, pty, Minus).map(|ty| ConstantInt(neg - pos, ty))
    } else {
        unify_int_type(nty, pty, Plus).map(|ty| ConstantInt(pos - neg, ty))
    }
}

fn sub_int(l: u64, lty: LitIntType, r: u64, rty: LitIntType, neg: bool) ->
        Option<ConstantVariant> {
     unify_int_type(lty, rty, if neg { Minus } else { Plus }).and_then(
        |ty| l.checked_sub(r).map(|v| ConstantInt(v, ty)))
}

fn constant_binop_apply<F>(cx: &Context, left: &Expr, right: &Expr, op: F)
        -> Option<Constant>
where F: Fn(ConstantVariant, ConstantVariant) -> Option<ConstantVariant> {
    if let (Some(Constant { constant: lc, needed_resolution: ln }),
            Some(Constant { constant: rc, needed_resolution: rn })) =
            (constant(cx, left), constant(cx, right)) {
        op(lc, rc).map(|c|
            Constant {
                needed_resolution: ln || rn,
                constant: c,
            })
    } else { None }
}

fn constant_short_circuit(cx: &Context, left: &Expr, right: &Expr, b: bool) ->
        Option<Constant> {
    constant(cx, left).and_then(|left|
        if let &ConstantBool(lbool) = &left.constant {
            if lbool == b {
                Some(left)
            } else {
                constant(cx, right).and_then(|right|
                    if let ConstantBool(_) = right.constant {
                        Some(Constant {
                            constant: right.constant,
                            needed_resolution: left.needed_resolution ||
                                               right.needed_resolution,
                        })
                    } else { None }
                )
            }
        } else { None }
    )
}
