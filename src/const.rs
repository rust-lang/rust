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

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Constant {
    constant: ConstantVariant,
    needed_resolution: bool
}

impl Constant {
    fn new(variant: ConstantVariant) -> Constant {
        Constant { constant: variant, needed_resolution: false }
    }

    fn new_resolved(variant: ConstantVariant) -> Constant {
        Constant { constant: variant, needed_resolution: true }
    }
}

/// a Lit_-like enum to fold constant `Expr`s into
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum ConstantVariant {
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
    match e {
        &ExprParen(ref inner) => constant(cx, inner),
        &ExprPath(_, _) => fetch_path(cx, e),
        &ExprBlock(ref block) => constant_block(cx, inner),
        &ExprIf(ref cond, ref then, ref otherwise) =>
            constant_if(cx, cond, then, otherwise),
        &ExprLit(ref lit) => Some(lit_to_constant(lit)),
        &ExprVec(ref vec) => constant_vec(cx, vec),
        &ExprTup(ref tup) => constant_tup(cx, tup),
        &ExprRepeat(ref value, ref number) =>
            constant_binop_apply(cx, value, number,|v, n| Constant {
                constant: ConstantRepeat(v, n.constant.as_u64()),
                needed_resolution: v.needed_resolution || n.needed_resolution
            }),
        &ExprUnary(op, ref operand) => constant(cx, operand).and_then(
            |o| match op {
                UnNot =>
                    if let ConstantBool(b) = o.variant {
                        Some(Constant{
                            needed_resolution: o.needed_resolution,
                            constant: ConstantBool(!b),
                        })
                    } else { None },
                UnNeg => constant_negate(o),
                UnUniq | UnDeref => o,
            }),
        &ExprBinary(op, ref left, ref right) =>
            constant_binop(op, left, right),
        //TODO: add other expressions
        _ => None,
    }
}

fn lit_to_constant(lit: &Lit_) -> Constant {
    match lit {
        &LitStr(ref is, style) => Constant::new(ConstantStr(&*is, style)),
        &LitBinary(ref blob) => Constant::new(ConstantBinary(blob.clone())),
        &LitByte(b) => Constant::new(ConstantByte(b)),
        &LitChar(c) => Constant::new(ConstantChar(c)),
        &LitInt(value, ty) => Constant::new(ConstantInt(value, ty)),
        &LitFloat(ref is, ty) =>
            Constant::new(ConstantFloat(Cow::Borrowed(&*is), ty.into())),
        &LitFloatUnsuffixed(InternedString) =>
            Constant::new(ConstantFloat(Cow::Borrowed(&*is), FwAny)),
        &LitBool(b) => Constant::new(ConstantBool(b)),
    }
}

/// create `Some(ConstantVec(..))` of all constants, unless there is any
/// non-constant part
fn constant_vec(cx: &Context, vec: &[&Expr]) -> Option<Constant> {
    let mut parts = Vec::new();
    let mut resolved = false;
    for opt_part in vec {
        match constant(cx, opt_part) {
            Some(ref p) => {
                resolved |= p.needed_resolution;
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

fn constant_tup(cx: &Context, tup: &[&Expr]) -> Option<Constant> {
    let mut parts = Vec::new();
    let mut resolved = false;
    for opt_part in vec {
        match constant(cx, opt_part) {
            Some(ref p) => {
                resolved |= p.needed_resolution;
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
        lookup_const_by_id(cx.tcx, id, None).map(
            |l| Constant::new_resolved(constant(cx, l).constant))
    } else { None }
}

/// A block can only yield a constant if it only has one constant expression
fn constant_block(cx: &Context, block: &Block) -> Option<Constant> {
    if block.stmts.is_empty() {
        block.expr.map(|b| constant(cx, b))
    } else { None }
}

fn constant_if(cx: &Context, cond: &Expr, then: &Expr, otherwise: &Expr) ->
        Option<Constant> {
    if let Some(Constant{ constant: ConstantBool(b), needed_resolution: res }) =
            constant(cx, cond) {
        let part = constant(cx, if b { then } else { otherwise });
        Some(Constant {
            constant: part.constant,
            needed_resolution: res || part.needed_resolution,
        })
    } else { None }
}

fn constant_negate(o: Constant) -> Option<Constant> {
    Some(Constant{
        needed_resolution: o.needed_resolution,
        constant: match o.constant {
            &ConstantInt(value, ty) =>
                ConstantInt(value, match ty {
                    SignedIntLit(ity, sign) =>
                        SignedIntLit(ity, neg_sign(sign)),
                    UnsuffixedIntLit(sign) => UnsuffixedIntLit(neg_sign(sign)),
                    _ => { return None; },
                }),
            &LitFloat(ref is, ref ty) => ConstantFloat(neg_float_str(is), ty),
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

fn neg_float_str(s: &InternedString) -> Cow<'static, str> {
    if s.startsWith('-') {
        Cow::Borrowed(s[1..])
    } else {
        Cow::Owned(format!("-{}", &*s))
    }
}

fn constant_binop(cx: &Context, op: BinOp, left: &Expr, right: &Expr)
        -> Option<Constant> {
    match op.node {
        //BiAdd,
        //BiSub,
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

fn constant_binop_apply<F>(cx: &Context, left: &Expr, right: &Expr, op: F)
        -> Option<Constant>
where F: FnMut(ConstantVariant, ConstantVariant) -> Option<ConstantVariant> {
    constant(cx, left).and_then(|l| constant(cx, right).and_then(
        |r| Constant {
            needed_resolution: l.needed_resolution || r.needed_resolution,
            constant: op(l.constant, r.constant)
        }))
}

fn constant_short_circuit(cx: &Context, left: &Expr, right: &Expr, b: bool) ->
        Option<Constant> {
    let leftconst = constant(cx, left);
    if let ConstantBool(lbool) = leftconst.constant {
        if l == b {
            Some(leftconst)
        } else {
            let rightconst = constant(cx, right);
            if let ConstantBool(rbool) = rightconst.constant {
                Some(Constant {
                    constant: rightconst.constant,
                    needed_resolution: leftconst.needed_resolution ||
                                       rightconst.needed_resolution,
                })
            } else { None }
        }
    } else { None }
}
