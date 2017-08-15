#![allow(cast_possible_truncation)]

use rustc::lint::LateContext;
use rustc::hir::def::Def;
use rustc_const_eval::lookup_const_by_id;
use rustc_const_math::ConstInt;
use rustc::hir::*;
use rustc::ty::{self, TyCtxt, Ty};
use rustc::ty::subst::{Substs, Subst};
use std::cmp::Ordering::{self, Equal};
use std::cmp::PartialOrd;
use std::hash::{Hash, Hasher};
use std::mem;
use std::rc::Rc;
use syntax::ast::{FloatTy, LitKind, StrStyle};
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

impl PartialEq for Constant {
    fn eq(&self, other: &Constant) -> bool {
        match (self, other) {
            (&Constant::Str(ref ls, ref l_sty), &Constant::Str(ref rs, ref r_sty)) => ls == rs && l_sty == r_sty,
            (&Constant::Binary(ref l), &Constant::Binary(ref r)) => l == r,
            (&Constant::Char(l), &Constant::Char(r)) => l == r,
            (&Constant::Int(l), &Constant::Int(r)) => {
                l.is_negative() == r.is_negative() && l.to_u128_unchecked() == r.to_u128_unchecked()
            },
            (&Constant::Float(ref ls, _), &Constant::Float(ref rs, _)) => {
                // we want `Fw32 == FwAny` and `FwAny == Fw64`, by transitivity we must have
                // `Fw32 == Fw64` so don’t compare them
                match (ls.parse::<f64>(), rs.parse::<f64>()) {
                    // mem::transmute is required to catch non-matching 0.0, -0.0, and NaNs
                    (Ok(l), Ok(r)) => unsafe { mem::transmute::<f64, u64>(l) == mem::transmute::<f64, u64>(r) },
                    _ => false,
                }
            },
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
    where
        H: Hasher,
    {
        match *self {
            Constant::Str(ref s, ref k) => {
                s.hash(state);
                k.hash(state);
            },
            Constant::Binary(ref b) => {
                b.hash(state);
            },
            Constant::Char(c) => {
                c.hash(state);
            },
            Constant::Int(i) => {
                i.to_u128_unchecked().hash(state);
                i.is_negative().hash(state);
            },
            Constant::Float(ref f, _) => {
                // don’t use the width here because of PartialEq implementation
                if let Ok(f) = f.parse::<f64>() {
                    unsafe { mem::transmute::<f64, u64>(f) }.hash(state);
                }
            },
            Constant::Bool(b) => {
                b.hash(state);
            },
            Constant::Vec(ref v) |
            Constant::Tuple(ref v) => {
                v.hash(state);
            },
            Constant::Repeat(ref c, l) => {
                c.hash(state);
                l.hash(state);
            },
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
            },
            (&Constant::Char(ref l), &Constant::Char(ref r)) => Some(l.cmp(r)),
            (&Constant::Int(l), &Constant::Int(r)) => Some(l.cmp(&r)),
            (&Constant::Float(ref ls, _), &Constant::Float(ref rs, _)) => {
                match (ls.parse::<f64>(), rs.parse::<f64>()) {
                    (Ok(ref l), Ok(ref r)) => {
                        match (l.partial_cmp(r), l.is_sign_positive() == r.is_sign_positive()) {
                            // Check for comparison of -0.0 and 0.0
                            (Some(Ordering::Equal), false) => None,
                            (x, _) => x,
                        }
                    },
                    _ => None,
                }
            },
            (&Constant::Bool(ref l), &Constant::Bool(ref r)) => Some(l.cmp(r)),
            (&Constant::Tuple(ref l), &Constant::Tuple(ref r)) |
            (&Constant::Vec(ref l), &Constant::Vec(ref r)) => l.partial_cmp(r),
            (&Constant::Repeat(ref lv, ref ls), &Constant::Repeat(ref rv, ref rs)) => {
                match lv.partial_cmp(rv) {
                    Some(Equal) => Some(ls.cmp(rs)),
                    x => x,
                }
            },
            _ => None, //TODO: Are there any useful inter-type orderings?
        }
    }
}

/// parse a `LitKind` to a `Constant`
#[allow(cast_possible_wrap)]
pub fn lit_to_constant<'a, 'tcx>(lit: &LitKind, tcx: TyCtxt<'a, 'tcx, 'tcx>, mut ty: Ty<'tcx>) -> Constant {
    use syntax::ast::*;
    use syntax::ast::LitIntType::*;
    use rustc::ty::util::IntTypeExt;

    if let ty::TyAdt(adt, _) = ty.sty {
        if adt.is_enum() {
            ty = adt.repr.discr_type().to_ty(tcx)
        }
    }
    match *lit {
        LitKind::Str(ref is, style) => Constant::Str(is.to_string(), style),
        LitKind::Byte(b) => Constant::Int(ConstInt::U8(b)),
        LitKind::ByteStr(ref s) => Constant::Binary(s.clone()),
        LitKind::Char(c) => Constant::Char(c),
        LitKind::Int(n, hint) => {
            match (&ty.sty, hint) {
                (&ty::TyInt(ity), _) |
                (_, Signed(ity)) => {
                    Constant::Int(ConstInt::new_signed_truncating(n as i128, ity, tcx.sess.target.int_type))
                },
                (&ty::TyUint(uty), _) |
                (_, Unsigned(uty)) => {
                    Constant::Int(ConstInt::new_unsigned_truncating(n as u128, uty, tcx.sess.target.uint_type))
                },
                _ => bug!(),
            }
        },
        LitKind::Float(ref is, ty) => Constant::Float(is.to_string(), ty.into()),
        LitKind::FloatUnsuffixed(ref is) => Constant::Float(is.to_string(), FloatWidth::Any),
        LitKind::Bool(b) => Constant::Bool(b),
    }
}

fn constant_not(o: &Constant) -> Option<Constant> {
    use self::Constant::*;
    match *o {
        Bool(b) => Some(Bool(!b)),
        Int(value) => (!value).ok().map(Int),
        _ => None,
    }
}

fn constant_negate(o: Constant) -> Option<Constant> {
    use self::Constant::*;
    match o {
        Int(value) => (-value).ok().map(Int),
        Float(is, ty) => Some(Float(neg_float_str(&is), ty)),
        _ => None,
    }
}

fn neg_float_str(s: &str) -> String {
    if s.starts_with('-') {
        s[1..].to_owned()
    } else {
        format!("-{}", s)
    }
}

pub fn constant(lcx: &LateContext, e: &Expr) -> Option<(Constant, bool)> {
    let mut cx = ConstEvalLateContext {
        tcx: lcx.tcx,
        tables: lcx.tables,
        param_env: lcx.param_env,
        needed_resolution: false,
        substs: lcx.tcx.intern_substs(&[]),
    };
    cx.expr(e).map(|cst| (cst, cx.needed_resolution))
}

pub fn constant_simple(lcx: &LateContext, e: &Expr) -> Option<Constant> {
    constant(lcx, e).and_then(|(cst, res)| if res { None } else { Some(cst) })
}

struct ConstEvalLateContext<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    needed_resolution: bool,
    substs: &'tcx Substs<'tcx>,
}

impl<'c, 'cc> ConstEvalLateContext<'c, 'cc> {
    /// simple constant folding: Insert an expression, get a constant or none.
    fn expr(&mut self, e: &Expr) -> Option<Constant> {
        match e.node {
            ExprPath(ref qpath) => self.fetch_path(qpath, e.hir_id),
            ExprBlock(ref block) => self.block(block),
            ExprIf(ref cond, ref then, ref otherwise) => self.ifthenelse(cond, then, otherwise),
            ExprLit(ref lit) => Some(lit_to_constant(&lit.node, self.tcx, self.tables.expr_ty(e))),
            ExprArray(ref vec) => self.multi(vec).map(Constant::Vec),
            ExprTup(ref tup) => self.multi(tup).map(Constant::Tuple),
            ExprRepeat(ref value, _) => {
                let n = match self.tables.expr_ty(e).sty {
                    ty::TyArray(_, n) => n,
                    _ => span_bug!(e.span, "typeck error"),
                };
                self.expr(value).map(|v| Constant::Repeat(Box::new(v), n))
            },
            ExprUnary(op, ref operand) => {
                self.expr(operand).and_then(|o| match op {
                    UnNot => constant_not(&o),
                    UnNeg => constant_negate(o),
                    UnDeref => Some(o),
                })
            },
            ExprBinary(op, ref left, ref right) => self.binop(op, left, right),
            // TODO: add other expressions
            _ => None,
        }
    }

    /// create `Some(Vec![..])` of all constants, unless there is any
    /// non-constant part
    fn multi(&mut self, vec: &[Expr]) -> Option<Vec<Constant>> {
        vec.iter()
            .map(|elem| self.expr(elem))
            .collect::<Option<_>>()
    }

    /// lookup a possibly constant expression from a ExprPath
    fn fetch_path(&mut self, qpath: &QPath, id: HirId) -> Option<Constant> {
        let def = self.tables.qpath_def(qpath, id);
        match def {
            Def::Const(def_id) |
            Def::AssociatedConst(def_id) => {
                let substs = self.tables.node_substs(id);
                let substs = if self.substs.is_empty() {
                    substs
                } else {
                    substs.subst(self.tcx, self.substs)
                };
                let param_env = self.param_env.and((def_id, substs));
                if let Some((def_id, substs)) = lookup_const_by_id(self.tcx, param_env) {
                    let mut cx = ConstEvalLateContext {
                        tcx: self.tcx,
                        tables: self.tcx.typeck_tables_of(def_id),
                        needed_resolution: false,
                        substs: substs,
                        param_env: param_env.param_env,
                    };
                    let body = if let Some(id) = self.tcx.hir.as_local_node_id(def_id) {
                        self.tcx.mir_const_qualif(def_id);
                        self.tcx.hir.body(self.tcx.hir.body_owned_by(id))
                    } else {
                        self.tcx.sess.cstore.item_body(self.tcx, def_id)
                    };
                    let ret = cx.expr(&body.value);
                    if ret.is_some() {
                        self.needed_resolution = true;
                    }
                    return ret;
                }
            },
            _ => {},
        }
        None
    }

    /// A block can only yield a constant if it only has one constant expression
    fn block(&mut self, block: &Block) -> Option<Constant> {
        if block.stmts.is_empty() {
            block.expr.as_ref().and_then(|b| self.expr(b))
        } else {
            None
        }
    }

    fn ifthenelse(&mut self, cond: &Expr, then: &P<Expr>, otherwise: &Option<P<Expr>>) -> Option<Constant> {
        if let Some(Constant::Bool(b)) = self.expr(cond) {
            if b {
                self.expr(&**then)
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
}
