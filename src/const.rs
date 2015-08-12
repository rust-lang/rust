use rustc::lint::Context;
use rustc::middle::const_eval::lookup_const_by_id;
use syntax::ast::*;
use syntax::ptr::P;

/// a Lit_-like enum to fold constant `Expr`s into
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Constant {
    ConstantStr(&'static str, StrStyle),
    ConstantBinary(Rc<Vec<u8>>),
    ConstantByte(u8),
    ConstantChar(char),
    ConstantInt(u64, LitIntType),
    ConstantFloat(Cow<'static, str>, FloatTy),
    ConstantFloatUnsuffixed(Cow<'static, str>),
    ConstantBool(bool),
    ConstantVec(Vec<Constant>),
    ConstantTuple(Vec<Constant>),
}

/// simple constant folding
pub fn constant(cx: &Context, e: &Expr) -> Option<Constant> {
    match e {
        &ExprParen(ref inner) => constant(cx, inner),
        &ExprPath(_, _) => fetch_path(cx, e),
        &ExprBlock(ref block) => constant_block(cx, inner),
        &ExprIf(ref cond, ref then, ref otherwise) => 
            match constant(cx, cond) {
                Some(LitBool(true)) => constant(cx, then),
                Some(LitBool(false)) => constant(cx, otherwise),
                _ => None,
            },
        &ExprLit(ref lit) => Some(lit_to_constant(lit)),
        &ExprVec(ref vec) => constant_vec(cx, vec),
        &ExprTup(ref tup) => constant_tup(cx, tup),
        &ExprUnary(op, ref operand) => constant(cx, operand).and_then(
            |o| match op {
                UnNot =>
                    if let ConstantBool(b) = o {
                        Some(ConstantBool(!b))
                    } else { None },
                UnNeg =>
                    match o {
                        &ConstantInt(value, ty) =>
                            Some(ConstantInt(value, match ty {
                                SignedIntLit(ity, sign) => 
                                    SignedIntLit(ity, neg_sign(sign)),
                                UnsuffixedIntLit(sign) =>
                                    UnsuffixedIntLit(neg_sign(sign)),
                                _ => { return None; },
                            })),
                        &LitFloat(ref is, ref ty) =>
                            Some(ConstantFloat(neg_float_str(is), ty)),
                        &LitFloatUnsuffixed(ref is) => 
                            Some(ConstantFloatUnsuffixed(neg_float_str(is))),
                        _ => None,
                    },
                UnUniq | UnDeref => o,
            }),
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
        &LitFloat(ref is, ty) => ConstantFloat(Cow::Borrowed(&*is), ty),
        &LitFloatUnsuffixed(InternedString) => 
            ConstantFloatUnsuffixed(Cow::Borrowed(&*is)),
        &LitBool(b) => ConstantBool(b),
    }
}

/// create `Some(ConstantVec(..))` of all constants, unless there is any
/// non-constant part
fn constant_vec(cx: &Context, vec: &[&Expr]) -> Option<Constant> {
    Vec<Constant> parts = Vec::new();
    for opt_part in vec {
        match constant(cx, opt_part) {
            Some(ref p) => parts.push(p),
            None => { return None; },
        }
    }
    Some(ConstantVec(parts))
}

fn constant_tup(cx, &Context, tup: &[&Expr]) -> Option<Constant> {
    Vec<Constant> parts = Vec::new();
    for opt_part in vec {
        match constant(cx, opt_part) {
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
fn constant_block(cx: &Context, block: &Block) -> Option<Constant> {
    if block.stmts.is_empty() {
        block.expr.map(|b| constant(cx, b)) 
    } else { None }
}

fn neg_sign(s: Sign) -> Sign {
    match s:
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
