use rustc::lint::Context;
use rustc::middle::const_eval::lookup_const_by_id;
use syntax::ast::*;
use syntax::ptr::P;

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

/// a Lit_-like enum to fold constant `Expr`s into
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Constant {
    /// a String "abc"
    ConstantStr(&'static str, StrStyle),
    /// a Binary String b"abc"
    ConstantBinary(Rc<Vec<u8>>),
    /// a single byte b'a'
    ConstantByte(u8),
    /// a single char 'a'
    ConstantChar(char),
    /// an integer
    ConstantInt(u64, LitIntType),
    /// a float with given type
    ConstantFloat(Cow<'static, str>, FloatWidth),
    /// true or false
    ConstantBool(bool),
    /// an array of constants
    ConstantVec(Vec<Constant>),
    /// also an array, but with only one constant, repeated N times
    ConstantRepeat(Constant, usize),
    /// a tuple of constants
    ConstantTuple(Vec<Constant>),
}

/// simple constant folding
pub fn constant(cx: &Context, e: &Expr, follow: bool) -> Option<Constant> {
    match e {
        &ExprParen(ref inner) => constant(cx, inner, follow),
        &ExprPath(_, _) => if follow { fetch_path(cx, e) } else { None },
        &ExprBlock(ref block) => constant_block(cx, inner, follow),
        &ExprIf(ref cond, ref then, ref otherwise) =>
            match constant(cx, cond) {
                Some(ConstantBool(true)) => constant(cx, then, follow),
                Some(ConstantBool(false)) => constant(cx, otherwise, follow),
                _ => None,
            },
        &ExprLit(ref lit) => Some(lit_to_constant(lit)),
        &ExprVec(ref vec) => constant_vec(cx, vec, follow),
        &ExprTup(ref tup) => constant_tup(cx, tup, follow),
        &ExprRepeat(ref value, ref number) =>
            constant_binop_apply(cx, value, number,|v, n| ConstantRepeat(v, n)),
        &ExprUnary(op, ref operand) => constant(cx, operand, follow).and_then(
            |o| match op {
                UnNot =>
                    if let ConstantBool(b) = o {
                        Some(ConstantBool(!b))
                    } else { None },
                UnNeg => constant_negate(o),
                UnUniq | UnDeref => o,
            }),
        &ExprBinary(op, ref left, ref right) =>
            constant_binop(cx, op, left, right, follow),
        //TODO: add other expressions
        _ => None,
    }
}

fn lit_to_constant(lit: &Lit_) -> Constant {
    match lit {
        &LitStr(ref is, style) => ConstantStr(&*is, style),
        &LitBinary(ref blob) => ConstantBinary(blob.clone()),
        &LitByte(b) => ConstantByte(b),
        &LitChar(c) => ConstantChar(c),
        &LitInt(value, ty) => ConstantInt(value, ty),
        &LitFloat(ref is, ty) => ConstantFloat(Cow::Borrowed(&*is), ty.into()),
        &LitFloatUnsuffixed(InternedString) =>
                ConstantFloat(Cow::Borrowed(&*is), FwAny),
        &LitBool(b) => ConstantBool(b),
    }
}

/// create `Some(ConstantVec(..))` of all constants, unless there is any
/// non-constant part
fn constant_vec(cx: &Context, vec: &[&Expr], follow: bool) -> Option<Constant> {
    parts = Vec::new();
    for opt_part in vec {
        match constant(cx, opt_part, follow) {
            Some(ref p) => parts.push(p),
            None => { return None; },
        }
    }
    Some(ConstantVec(parts))
}

fn constant_tup(cx: &Context, tup: &[&Expr], follow: bool) -> Option<Constant> {
    parts = Vec::new();
    for opt_part in vec {
        match constant(cx, opt_part, follow) {
            Some(ref p) => parts.push(p),
            None => { return None; },
        }
    }
    Some(ConstantTuple(parts))
}

/// lookup a possibly constant expression from a ExprPath
fn fetch_path(cx: &Context, e: &Expr) -> Option<Constant> {
    if let Some(&PathResolution { base_def: DefConst(id), ..}) =
            cx.tcx.def_map.borrow().get(&e.id) {
        lookup_const_by_id(cx.tcx, id, None).map(|l| constant(cx, l))
    } else { None }
}

/// A block can only yield a constant if it only has one constant expression
fn constant_block(cx: &Context, block: &Block, follow: bool) -> Option<Constant> {
    if block.stmts.is_empty() {
        block.expr.map(|b| constant(cx, b, follow))
    } else { None }
}

fn constant_negate(o: Constant) -> Option<Constant> {
    match o {
        &ConstantInt(value, ty) =>
            Some(ConstantInt(value, match ty {
                SignedIntLit(ity, sign) => SignedIntLit(ity, neg_sign(sign)),
                UnsuffixedIntLit(sign) => UnsuffixedIntLit(neg_sign(sign)),
                _ => { return None; },
            })),
        &LitFloat(ref is, ref ty) => Some(ConstantFloat(neg_float_str(is), ty)),
        _ => None,
    }
}

fn neg_sign(s: Sign) -> Sign {
    match s {
        Sign::Plus => Sign::Minus,
        Sign::Minus => Sign::Plus,
    }
}

fn neg_float_str(s: &InternedString) -> Cow<'static, str> {
    if s.startsWith('-') {
        Cow::Borrowed(s[1..])
    } else {
        Cow::Owned(format!("-{}", &*s))
    }
}

fn constant_binop(cx: &Context, op: BinOp, left: &Expr, right: &Expr,
        follow: bool) -> Option<Constant> {
    match op.node {
        //BiAdd,
        //BiSub,
        //BiMul,
        //BiDiv,
        //BiRem,
        BiAnd => constant_short_circuit(cx, left, right, false, follow),
        BiOr => constant_short_circuit(cx, left, right, true, follow),
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

fn constant_binop_apply<F>(cx: &Context, left: &Expr, right: &Expr, op: F,
        follow: bool) -> Option<Constant>
where F: FnMut(Constant, Constant) -> Option<Constant> {
    constant(cx, left, follow).and_then(|l| constant(cx, right, follow)
        .and_then(|r| op(l, r)))
}

fn constant_short_circuit(cx: &Context, left: &Expr, right: &Expr, b: bool,
        follow: bool) -> Option<Constant> {
    if let ConstantBool(lbool) = constant(cx, left, follow) {
        if l == b {
            Some(ConstantBool(b))
        } else {
            if let ConstantBool(rbool) = constant(cx, right, follow) {
                Some(ConstantBool(rbool))
            } else { None }
        }
    } else { None }
}
