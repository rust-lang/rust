// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]
#![allow(unsigned_negation)]

pub use self::const_val::*;

use self::ErrKind::*;

use metadata::csearch;
use middle::{astencode, def};
use middle::pat_util::def_to_path;
use middle::ty::{self, Ty};
use middle::astconv_util::ast_ty_to_prim_ty;

use syntax::ast::{self, Expr};
use syntax::codemap::Span;
use syntax::feature_gate;
use syntax::parse::token::InternedString;
use syntax::ptr::P;
use syntax::{ast_map, ast_util, codemap};

use std::borrow::{Cow, IntoCow};
use std::num::wrapping::OverflowingOps;
use std::num::ToPrimitive;
use std::cmp::Ordering;
use std::collections::hash_map::Entry::Vacant;
use std::{i8, i16, i32, i64};
use std::rc::Rc;

fn lookup_const<'a>(tcx: &'a ty::ctxt, e: &Expr) -> Option<&'a Expr> {
    let opt_def = tcx.def_map.borrow().get(&e.id).map(|d| d.full_def());
    match opt_def {
        Some(def::DefConst(def_id)) => {
            lookup_const_by_id(tcx, def_id)
        }
        Some(def::DefVariant(enum_def, variant_def, _)) => {
            lookup_variant_by_id(tcx, enum_def, variant_def)
        }
        _ => None
    }
}

fn lookup_variant_by_id<'a>(tcx: &'a ty::ctxt,
                            enum_def: ast::DefId,
                            variant_def: ast::DefId)
                            -> Option<&'a Expr> {
    fn variant_expr<'a>(variants: &'a [P<ast::Variant>], id: ast::NodeId)
                        -> Option<&'a Expr> {
        for variant in variants {
            if variant.node.id == id {
                return variant.node.disr_expr.as_ref().map(|e| &**e);
            }
        }
        None
    }

    if ast_util::is_local(enum_def) {
        match tcx.map.find(enum_def.node) {
            None => None,
            Some(ast_map::NodeItem(it)) => match it.node {
                ast::ItemEnum(ast::EnumDef { ref variants }, _) => {
                    variant_expr(&variants[..], variant_def.node)
                }
                _ => None
            },
            Some(_) => None
        }
    } else {
        match tcx.extern_const_variants.borrow().get(&variant_def) {
            Some(&ast::DUMMY_NODE_ID) => return None,
            Some(&expr_id) => {
                return Some(tcx.map.expect_expr(expr_id));
            }
            None => {}
        }
        let expr_id = match csearch::maybe_get_item_ast(tcx, enum_def,
            Box::new(|a, b, c, d| astencode::decode_inlined_item(a, b, c, d))) {
            csearch::FoundAst::Found(&ast::IIItem(ref item)) => match item.node {
                ast::ItemEnum(ast::EnumDef { ref variants }, _) => {
                    // NOTE this doesn't do the right thing, it compares inlined
                    // NodeId's to the original variant_def's NodeId, but they
                    // come from different crates, so they will likely never match.
                    variant_expr(&variants[..], variant_def.node).map(|e| e.id)
                }
                _ => None
            },
            _ => None
        };
        tcx.extern_const_variants.borrow_mut().insert(variant_def,
                                                      expr_id.unwrap_or(ast::DUMMY_NODE_ID));
        expr_id.map(|id| tcx.map.expect_expr(id))
    }
}

pub fn lookup_const_by_id<'a>(tcx: &'a ty::ctxt, def_id: ast::DefId)
                          -> Option<&'a Expr> {
    if ast_util::is_local(def_id) {
        match tcx.map.find(def_id.node) {
            None => None,
            Some(ast_map::NodeItem(it)) => match it.node {
                ast::ItemConst(_, ref const_expr) => {
                    Some(&**const_expr)
                }
                _ => None
            },
            Some(_) => None
        }
    } else {
        match tcx.extern_const_statics.borrow().get(&def_id) {
            Some(&ast::DUMMY_NODE_ID) => return None,
            Some(&expr_id) => {
                return Some(tcx.map.expect_expr(expr_id));
            }
            None => {}
        }
        let expr_id = match csearch::maybe_get_item_ast(tcx, def_id,
            Box::new(|a, b, c, d| astencode::decode_inlined_item(a, b, c, d))) {
            csearch::FoundAst::Found(&ast::IIItem(ref item)) => match item.node {
                ast::ItemConst(_, ref const_expr) => Some(const_expr.id),
                _ => None
            },
            _ => None
        };
        tcx.extern_const_statics.borrow_mut().insert(def_id,
                                                     expr_id.unwrap_or(ast::DUMMY_NODE_ID));
        expr_id.map(|id| tcx.map.expect_expr(id))
    }
}

#[derive(Clone, PartialEq)]
pub enum const_val {
    const_float(f64),
    const_int(i64),
    const_uint(u64),
    const_str(InternedString),
    const_binary(Rc<Vec<u8>>),
    const_bool(bool),
    Struct(ast::NodeId),
    Tuple(ast::NodeId)
}

pub fn const_expr_to_pat(tcx: &ty::ctxt, expr: &Expr, span: Span) -> P<ast::Pat> {
    let pat = match expr.node {
        ast::ExprTup(ref exprs) =>
            ast::PatTup(exprs.iter().map(|expr| const_expr_to_pat(tcx, &**expr, span)).collect()),

        ast::ExprCall(ref callee, ref args) => {
            let def = *tcx.def_map.borrow().get(&callee.id).unwrap();
            if let Vacant(entry) = tcx.def_map.borrow_mut().entry(expr.id) {
               entry.insert(def);
            }
            let path = match def.full_def() {
                def::DefStruct(def_id) => def_to_path(tcx, def_id),
                def::DefVariant(_, variant_did, _) => def_to_path(tcx, variant_did),
                _ => unreachable!()
            };
            let pats = args.iter().map(|expr| const_expr_to_pat(tcx, &**expr, span)).collect();
            ast::PatEnum(path, Some(pats))
        }

        ast::ExprStruct(ref path, ref fields, None) => {
            let field_pats = fields.iter().map(|field| codemap::Spanned {
                span: codemap::DUMMY_SP,
                node: ast::FieldPat {
                    ident: field.ident.node,
                    pat: const_expr_to_pat(tcx, &*field.expr, span),
                    is_shorthand: false,
                },
            }).collect();
            ast::PatStruct(path.clone(), field_pats, false)
        }

        ast::ExprVec(ref exprs) => {
            let pats = exprs.iter().map(|expr| const_expr_to_pat(tcx, &**expr, span)).collect();
            ast::PatVec(pats, None, vec![])
        }

        ast::ExprPath(_, ref path) => {
            let opt_def = tcx.def_map.borrow().get(&expr.id).map(|d| d.full_def());
            match opt_def {
                Some(def::DefStruct(..)) =>
                    ast::PatStruct(path.clone(), vec![], false),
                Some(def::DefVariant(..)) =>
                    ast::PatEnum(path.clone(), None),
                _ => {
                    match lookup_const(tcx, expr) {
                        Some(actual) => return const_expr_to_pat(tcx, actual, span),
                        _ => unreachable!()
                    }
                }
            }
        }

        _ => ast::PatLit(P(expr.clone()))
    };
    P(ast::Pat { id: expr.id, node: pat, span: span })
}

pub fn eval_const_expr(tcx: &ty::ctxt, e: &Expr) -> const_val {
    match eval_const_expr_partial(tcx, e, None) {
        Ok(r) => r,
        Err(s) => tcx.sess.span_fatal(s.span, &s.description())
    }
}


#[derive(Clone)]
pub struct ConstEvalErr {
    pub span: Span,
    pub kind: ErrKind,
}

#[derive(Clone)]
pub enum ErrKind {
    CannotCast,
    CannotCastTo(&'static str),
    InvalidOpForBools(ast::BinOp_),
    InvalidOpForFloats(ast::BinOp_),
    InvalidOpForIntUint(ast::BinOp_),
    InvalidOpForUintInt(ast::BinOp_),
    NegateOnString,
    NegateOnBoolean,
    NegateOnBinary,
    NegateOnStruct,
    NegateOnTuple,
    NotOnFloat,
    NotOnString,
    NotOnBinary,
    NotOnStruct,
    NotOnTuple,

    NegateWithOverflow(i64),
    AddiWithOverflow(i64, i64),
    SubiWithOverflow(i64, i64),
    MuliWithOverflow(i64, i64),
    AdduWithOverflow(u64, u64),
    SubuWithOverflow(u64, u64),
    MuluWithOverflow(u64, u64),
    DivideByZero,
    DivideWithOverflow,
    ModuloByZero,
    ModuloWithOverflow,
    ShiftLeftWithOverflow,
    ShiftRightWithOverflow,
    MissingStructField,
    NonConstPath,
    ExpectedConstTuple,
    ExpectedConstStruct,
    TupleIndexOutOfBounds,

    MiscBinaryOp,
    MiscCatchAll,
}

impl ConstEvalErr {
    pub fn description(&self) -> Cow<str> {
        use self::ErrKind::*;

        match self.kind {
            CannotCast => "can't cast this type".into_cow(),
            CannotCastTo(s) => format!("can't cast this type to {}", s).into_cow(),
            InvalidOpForBools(_) =>  "can't do this op on bools".into_cow(),
            InvalidOpForFloats(_) => "can't do this op on floats".into_cow(),
            InvalidOpForIntUint(..) => "can't do this op on an isize and usize".into_cow(),
            InvalidOpForUintInt(..) => "can't do this op on a usize and isize".into_cow(),
            NegateOnString => "negate on string".into_cow(),
            NegateOnBoolean => "negate on boolean".into_cow(),
            NegateOnBinary => "negate on binary literal".into_cow(),
            NegateOnStruct => "negate on struct".into_cow(),
            NegateOnTuple => "negate on tuple".into_cow(),
            NotOnFloat => "not on float or string".into_cow(),
            NotOnString => "not on float or string".into_cow(),
            NotOnBinary => "not on binary literal".into_cow(),
            NotOnStruct => "not on struct".into_cow(),
            NotOnTuple => "not on tuple".into_cow(),

            NegateWithOverflow(..) => "attempted to negate with overflow".into_cow(),
            AddiWithOverflow(..) => "attempted to add with overflow".into_cow(),
            SubiWithOverflow(..) => "attempted to sub with overflow".into_cow(),
            MuliWithOverflow(..) => "attempted to mul with overflow".into_cow(),
            AdduWithOverflow(..) => "attempted to add with overflow".into_cow(),
            SubuWithOverflow(..) => "attempted to sub with overflow".into_cow(),
            MuluWithOverflow(..) => "attempted to mul with overflow".into_cow(),
            DivideByZero         => "attempted to divide by zero".into_cow(),
            DivideWithOverflow   => "attempted to divide with overflow".into_cow(),
            ModuloByZero         => "attempted remainder with a divisor of zero".into_cow(),
            ModuloWithOverflow   => "attempted remainder with overflow".into_cow(),
            ShiftLeftWithOverflow => "attempted left shift with overflow".into_cow(),
            ShiftRightWithOverflow => "attempted right shift with overflow".into_cow(),
            MissingStructField  => "nonexistent struct field".into_cow(),
            NonConstPath        => "non-constant path in constant expr".into_cow(),
            ExpectedConstTuple => "expected constant tuple".into_cow(),
            ExpectedConstStruct => "expected constant struct".into_cow(),
            TupleIndexOutOfBounds => "tuple index out of bounds".into_cow(),

            MiscBinaryOp => "bad operands for binary".into_cow(),
            MiscCatchAll => "unsupported constant expr".into_cow(),
        }
    }
}

pub type EvalResult = Result<const_val, ConstEvalErr>;
pub type CastResult = Result<const_val, ErrKind>;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum IntTy { I8, I16, I32, I64 }
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum UintTy { U8, U16, U32, U64 }

impl IntTy {
    pub fn from(tcx: &ty::ctxt, t: ast::IntTy) -> IntTy {
        let t = if let ast::TyIs = t {
            tcx.sess.target.int_type
        } else {
            t
        };
        match t {
            ast::TyIs => unreachable!(),
            ast::TyI8  => IntTy::I8,
            ast::TyI16 => IntTy::I16,
            ast::TyI32 => IntTy::I32,
            ast::TyI64 => IntTy::I64,
        }
    }
}

impl UintTy {
    pub fn from(tcx: &ty::ctxt, t: ast::UintTy) -> UintTy {
        let t = if let ast::TyUs = t {
            tcx.sess.target.uint_type
        } else {
            t
        };
        match t {
            ast::TyUs => unreachable!(),
            ast::TyU8  => UintTy::U8,
            ast::TyU16 => UintTy::U16,
            ast::TyU32 => UintTy::U32,
            ast::TyU64 => UintTy::U64,
        }
    }
}

macro_rules! signal {
    ($e:expr, $exn:expr) => {
        return Err(ConstEvalErr { span: $e.span, kind: $exn })
    }
}

// The const_{int,uint}_checked_{neg,add,sub,mul,div,shl,shr} family
// of functions catch and signal overflow errors during constant
// evaluation.
//
// They all take the operator's arguments (`a` and `b` if binary), the
// overall expression (`e`) and, if available, whole expression's
// concrete type (`opt_ety`).
//
// If the whole expression's concrete type is None, then this is a
// constant evaluation happening before type check (e.g. in the check
// to confirm that a pattern range's left-side is not greater than its
// right-side). We do not do arithmetic modulo the type's bitwidth in
// such a case; we just do 64-bit arithmetic and assume that later
// passes will do it again with the type information, and thus do the
// overflow checks then.

pub fn const_int_checked_neg<'a>(
    a: i64, e: &'a Expr, opt_ety: Option<IntTy>) -> EvalResult {

    let (min,max) = match opt_ety {
        // (-i8::MIN is itself not an i8, etc, but this is an easy way
        // to allow literals to pass the check. Of course that does
        // not work for i64::MIN.)
        Some(IntTy::I8) =>  (-(i8::MAX as i64), -(i8::MIN as i64)),
        Some(IntTy::I16) => (-(i16::MAX as i64), -(i16::MIN as i64)),
        Some(IntTy::I32) => (-(i32::MAX as i64), -(i32::MIN as i64)),
        None | Some(IntTy::I64) => (-i64::MAX, -(i64::MIN+1)),
    };

    let oflo = a < min || a > max;
    if oflo {
        signal!(e, NegateWithOverflow(a));
    } else {
        Ok(const_int(-a))
    }
}

pub fn const_uint_checked_neg<'a>(
    a: u64, _e: &'a Expr, _opt_ety: Option<UintTy>) -> EvalResult {
    // This always succeeds, and by definition, returns `(!a)+1`.
    Ok(const_uint(-a))
}

macro_rules! overflow_checking_body {
    ($a:ident, $b:ident, $ety:ident, $overflowing_op:ident,
     lhs: $to_8_lhs:ident $to_16_lhs:ident $to_32_lhs:ident,
     rhs: $to_8_rhs:ident $to_16_rhs:ident $to_32_rhs:ident $to_64_rhs:ident,
     $EnumTy:ident $T8: ident $T16: ident $T32: ident $T64: ident,
     $result_type: ident) => { {
        let (a,b,opt_ety) = ($a,$b,$ety);
        match opt_ety {
            Some($EnumTy::$T8) => match (a.$to_8_lhs(), b.$to_8_rhs()) {
                (Some(a), Some(b)) => {
                    let (a, oflo) = a.$overflowing_op(b);
                    (a as $result_type, oflo)
                }
                (None, _) | (_, None) => (0, true)
            },
            Some($EnumTy::$T16) => match (a.$to_16_lhs(), b.$to_16_rhs()) {
                (Some(a), Some(b)) => {
                    let (a, oflo) = a.$overflowing_op(b);
                    (a as $result_type, oflo)
                }
                (None, _) | (_, None) => (0, true)
            },
            Some($EnumTy::$T32) => match (a.$to_32_lhs(), b.$to_32_rhs()) {
                (Some(a), Some(b)) => {
                    let (a, oflo) = a.$overflowing_op(b);
                    (a as $result_type, oflo)
                }
                (None, _) | (_, None) => (0, true)
            },
            None | Some($EnumTy::$T64) => match b.$to_64_rhs() {
                Some(b) => a.$overflowing_op(b),
                None => (0, true),
            }
        }
    } }
}

macro_rules! int_arith_body {
    ($a:ident, $b:ident, $ety:ident, $overflowing_op:ident) => {
        overflow_checking_body!(
            $a, $b, $ety, $overflowing_op,
            lhs: to_i8 to_i16 to_i32,
            rhs: to_i8 to_i16 to_i32 to_i64, IntTy I8 I16 I32 I64, i64)
    }
}

macro_rules! uint_arith_body {
    ($a:ident, $b:ident, $ety:ident, $overflowing_op:ident) => {
        overflow_checking_body!(
            $a, $b, $ety, $overflowing_op,
            lhs: to_u8 to_u16 to_u32,
            rhs: to_u8 to_u16 to_u32 to_u64, UintTy U8 U16 U32 U64, u64)
    }
}

macro_rules! int_shift_body {
    ($a:ident, $b:ident, $ety:ident, $overflowing_op:ident) => {
        overflow_checking_body!(
            $a, $b, $ety, $overflowing_op,
            lhs: to_i8 to_i16 to_i32,
            rhs: to_u32 to_u32 to_u32 to_u32, IntTy I8 I16 I32 I64, i64)
    }
}

macro_rules! uint_shift_body {
    ($a:ident, $b:ident, $ety:ident, $overflowing_op:ident) => {
        overflow_checking_body!(
            $a, $b, $ety, $overflowing_op,
            lhs: to_u8 to_u16 to_u32,
            rhs: to_u32 to_u32 to_u32 to_u32, UintTy U8 U16 U32 U64, u64)
    }
}

macro_rules! pub_fn_checked_op {
    {$fn_name:ident ($a:ident : $a_ty:ty, $b:ident : $b_ty:ty,.. $WhichTy:ident) {
        $ret_oflo_body:ident $overflowing_op:ident
            $const_ty:ident $signal_exn:expr
    }} => {
        pub fn $fn_name<'a>($a: $a_ty,
                            $b: $b_ty,
                            e: &'a Expr,
                            opt_ety: Option<$WhichTy>) -> EvalResult {
            let (ret, oflo) = $ret_oflo_body!($a, $b, opt_ety, $overflowing_op);
            if !oflo { Ok($const_ty(ret)) } else { signal!(e, $signal_exn) }
        }
    }
}

pub_fn_checked_op!{ const_int_checked_add(a: i64, b: i64,.. IntTy) {
           int_arith_body overflowing_add const_int AddiWithOverflow(a, b)
}}

pub_fn_checked_op!{ const_int_checked_sub(a: i64, b: i64,.. IntTy) {
           int_arith_body overflowing_sub const_int SubiWithOverflow(a, b)
}}

pub_fn_checked_op!{ const_int_checked_mul(a: i64, b: i64,.. IntTy) {
           int_arith_body overflowing_mul const_int MuliWithOverflow(a, b)
}}

pub fn const_int_checked_div<'a>(
    a: i64, b: i64, e: &'a Expr, opt_ety: Option<IntTy>) -> EvalResult {
    if b == 0 { signal!(e, DivideByZero); }
    let (ret, oflo) = int_arith_body!(a, b, opt_ety, overflowing_div);
    if !oflo { Ok(const_int(ret)) } else { signal!(e, DivideWithOverflow) }
}

pub fn const_int_checked_rem<'a>(
    a: i64, b: i64, e: &'a Expr, opt_ety: Option<IntTy>) -> EvalResult {
    if b == 0 { signal!(e, ModuloByZero); }
    let (ret, oflo) = int_arith_body!(a, b, opt_ety, overflowing_rem);
    if !oflo { Ok(const_int(ret)) } else { signal!(e, ModuloWithOverflow) }
}

pub_fn_checked_op!{ const_int_checked_shl(a: i64, b: i64,.. IntTy) {
           int_shift_body overflowing_shl const_int ShiftLeftWithOverflow
}}

pub_fn_checked_op!{ const_int_checked_shl_via_uint(a: i64, b: u64,.. IntTy) {
           int_shift_body overflowing_shl const_int ShiftLeftWithOverflow
}}

pub_fn_checked_op!{ const_int_checked_shr(a: i64, b: i64,.. IntTy) {
           int_shift_body overflowing_shr const_int ShiftRightWithOverflow
}}

pub_fn_checked_op!{ const_int_checked_shr_via_uint(a: i64, b: u64,.. IntTy) {
           int_shift_body overflowing_shr const_int ShiftRightWithOverflow
}}

pub_fn_checked_op!{ const_uint_checked_add(a: u64, b: u64,.. UintTy) {
           uint_arith_body overflowing_add const_uint AdduWithOverflow(a, b)
}}

pub_fn_checked_op!{ const_uint_checked_sub(a: u64, b: u64,.. UintTy) {
           uint_arith_body overflowing_sub const_uint SubuWithOverflow(a, b)
}}

pub_fn_checked_op!{ const_uint_checked_mul(a: u64, b: u64,.. UintTy) {
           uint_arith_body overflowing_mul const_uint MuluWithOverflow(a, b)
}}

pub fn const_uint_checked_div<'a>(
    a: u64, b: u64, e: &'a Expr, opt_ety: Option<UintTy>) -> EvalResult {
    if b == 0 { signal!(e, DivideByZero); }
    let (ret, oflo) = uint_arith_body!(a, b, opt_ety, overflowing_div);
    if !oflo { Ok(const_uint(ret)) } else { signal!(e, DivideWithOverflow) }
}

pub fn const_uint_checked_rem<'a>(
    a: u64, b: u64, e: &'a Expr, opt_ety: Option<UintTy>) -> EvalResult {
    if b == 0 { signal!(e, ModuloByZero); }
    let (ret, oflo) = uint_arith_body!(a, b, opt_ety, overflowing_rem);
    if !oflo { Ok(const_uint(ret)) } else { signal!(e, ModuloWithOverflow) }
}

pub_fn_checked_op!{ const_uint_checked_shl(a: u64, b: u64,.. UintTy) {
           uint_shift_body overflowing_shl const_uint ShiftLeftWithOverflow
}}

pub_fn_checked_op!{ const_uint_checked_shl_via_int(a: u64, b: i64,.. UintTy) {
           uint_shift_body overflowing_shl const_uint ShiftLeftWithOverflow
}}

pub_fn_checked_op!{ const_uint_checked_shr(a: u64, b: u64,.. UintTy) {
           uint_shift_body overflowing_shr const_uint ShiftRightWithOverflow
}}

pub_fn_checked_op!{ const_uint_checked_shr_via_int(a: u64, b: i64,.. UintTy) {
           uint_shift_body overflowing_shr const_uint ShiftRightWithOverflow
}}

pub fn eval_const_expr_partial<'tcx>(tcx: &ty::ctxt<'tcx>,
                                     e: &Expr,
                                     ty_hint: Option<Ty<'tcx>>) -> EvalResult {
    fn fromb(b: bool) -> const_val { const_int(b as i64) }

    let ety = ty_hint.or_else(|| ty::expr_ty_opt(tcx, e));

    // If type of expression itself is int or uint, normalize in these
    // bindings so that isize/usize is mapped to a type with an
    // inherently known bitwidth.
    let expr_int_type = ety.and_then(|ty| {
        if let ty::ty_int(t) = ty.sty {
            Some(IntTy::from(tcx, t)) } else { None }
    });
    let expr_uint_type = ety.and_then(|ty| {
        if let ty::ty_uint(t) = ty.sty {
            Some(UintTy::from(tcx, t)) } else { None }
    });

    let result = match e.node {
      ast::ExprUnary(ast::UnNeg, ref inner) => {
        match try!(eval_const_expr_partial(tcx, &**inner, ety)) {
          const_float(f) => const_float(-f),
          const_int(n) =>  try!(const_int_checked_neg(n, e, expr_int_type)),
          const_uint(i) => {
              if !tcx.sess.features.borrow().negate_unsigned {
                  feature_gate::emit_feature_err(
                      &tcx.sess.parse_sess.span_diagnostic,
                      "negate_unsigned",
                      e.span,
                      "unary negation of unsigned integers may be removed in the future");
              }
              try!(const_uint_checked_neg(i, e, expr_uint_type))
          }
          const_str(_) => signal!(e, NegateOnString),
          const_bool(_) => signal!(e, NegateOnBoolean),
          const_binary(_) => signal!(e, NegateOnBinary),
          const_val::Tuple(_) => signal!(e, NegateOnTuple),
          const_val::Struct(..) => signal!(e, NegateOnStruct),
        }
      }
      ast::ExprUnary(ast::UnNot, ref inner) => {
        match try!(eval_const_expr_partial(tcx, &**inner, ety)) {
          const_int(i) => const_int(!i),
          const_uint(i) => const_uint(!i),
          const_bool(b) => const_bool(!b),
          const_str(_) => signal!(e, NotOnString),
          const_float(_) => signal!(e, NotOnFloat),
          const_binary(_) => signal!(e, NotOnBinary),
          const_val::Tuple(_) => signal!(e, NotOnTuple),
          const_val::Struct(..) => signal!(e, NotOnStruct),
        }
      }
      ast::ExprBinary(op, ref a, ref b) => {
        let b_ty = match op.node {
            ast::BiShl | ast::BiShr => Some(tcx.types.usize),
            _ => ety
        };
        match (try!(eval_const_expr_partial(tcx, &**a, ety)),
               try!(eval_const_expr_partial(tcx, &**b, b_ty))) {
          (const_float(a), const_float(b)) => {
            match op.node {
              ast::BiAdd => const_float(a + b),
              ast::BiSub => const_float(a - b),
              ast::BiMul => const_float(a * b),
              ast::BiDiv => const_float(a / b),
              ast::BiRem => const_float(a % b),
              ast::BiEq => fromb(a == b),
              ast::BiLt => fromb(a < b),
              ast::BiLe => fromb(a <= b),
              ast::BiNe => fromb(a != b),
              ast::BiGe => fromb(a >= b),
              ast::BiGt => fromb(a > b),
              _ => signal!(e, InvalidOpForFloats(op.node))
            }
          }
          (const_int(a), const_int(b)) => {
            match op.node {
              ast::BiAdd => try!(const_int_checked_add(a,b,e,expr_int_type)),
              ast::BiSub => try!(const_int_checked_sub(a,b,e,expr_int_type)),
              ast::BiMul => try!(const_int_checked_mul(a,b,e,expr_int_type)),
              ast::BiDiv => try!(const_int_checked_div(a,b,e,expr_int_type)),
              ast::BiRem => try!(const_int_checked_rem(a,b,e,expr_int_type)),
              ast::BiAnd | ast::BiBitAnd => const_int(a & b),
              ast::BiOr | ast::BiBitOr => const_int(a | b),
              ast::BiBitXor => const_int(a ^ b),
              ast::BiShl => try!(const_int_checked_shl(a,b,e,expr_int_type)),
              ast::BiShr => try!(const_int_checked_shr(a,b,e,expr_int_type)),
              ast::BiEq => fromb(a == b),
              ast::BiLt => fromb(a < b),
              ast::BiLe => fromb(a <= b),
              ast::BiNe => fromb(a != b),
              ast::BiGe => fromb(a >= b),
              ast::BiGt => fromb(a > b)
            }
          }
          (const_uint(a), const_uint(b)) => {
            match op.node {
              ast::BiAdd => try!(const_uint_checked_add(a,b,e,expr_uint_type)),
              ast::BiSub => try!(const_uint_checked_sub(a,b,e,expr_uint_type)),
              ast::BiMul => try!(const_uint_checked_mul(a,b,e,expr_uint_type)),
              ast::BiDiv => try!(const_uint_checked_div(a,b,e,expr_uint_type)),
              ast::BiRem => try!(const_uint_checked_rem(a,b,e,expr_uint_type)),
              ast::BiAnd | ast::BiBitAnd => const_uint(a & b),
              ast::BiOr | ast::BiBitOr => const_uint(a | b),
              ast::BiBitXor => const_uint(a ^ b),
              ast::BiShl => try!(const_uint_checked_shl(a,b,e,expr_uint_type)),
              ast::BiShr => try!(const_uint_checked_shr(a,b,e,expr_uint_type)),
              ast::BiEq => fromb(a == b),
              ast::BiLt => fromb(a < b),
              ast::BiLe => fromb(a <= b),
              ast::BiNe => fromb(a != b),
              ast::BiGe => fromb(a >= b),
              ast::BiGt => fromb(a > b),
            }
          }
          // shifts can have any integral type as their rhs
          (const_int(a), const_uint(b)) => {
            match op.node {
              ast::BiShl => try!(const_int_checked_shl_via_uint(a,b,e,expr_int_type)),
              ast::BiShr => try!(const_int_checked_shr_via_uint(a,b,e,expr_int_type)),
              _ => signal!(e, InvalidOpForIntUint(op.node)),
            }
          }
          (const_uint(a), const_int(b)) => {
            match op.node {
              ast::BiShl => try!(const_uint_checked_shl_via_int(a,b,e,expr_uint_type)),
              ast::BiShr => try!(const_uint_checked_shr_via_int(a,b,e,expr_uint_type)),
              _ => signal!(e, InvalidOpForUintInt(op.node)),
            }
          }
          (const_bool(a), const_bool(b)) => {
            const_bool(match op.node {
              ast::BiAnd => a && b,
              ast::BiOr => a || b,
              ast::BiBitXor => a ^ b,
              ast::BiBitAnd => a & b,
              ast::BiBitOr => a | b,
              ast::BiEq => a == b,
              ast::BiNe => a != b,
              _ => signal!(e, InvalidOpForBools(op.node)),
             })
          }

          _ => signal!(e, MiscBinaryOp),
        }
      }
      ast::ExprCast(ref base, ref target_ty) => {
        // This tends to get called w/o the type actually having been
        // populated in the ctxt, which was causing things to blow up
        // (#5900). Fall back to doing a limited lookup to get past it.
        let ety = ety.or_else(|| ast_ty_to_prim_ty(tcx, &**target_ty))
                .unwrap_or_else(|| {
                    tcx.sess.span_fatal(target_ty.span,
                                        "target type not found for const cast")
                });

        // Prefer known type to noop, but always have a type hint.
        //
        // FIXME (#23833): the type-hint can cause problems,
        // e.g. `(i8::MAX + 1_i8) as u32` feeds in `u32` as result
        // type to the sum, and thus no overflow is signaled.
        let base_hint = ty::expr_ty_opt(tcx, &**base).unwrap_or(ety);
        let val = try!(eval_const_expr_partial(tcx, &**base, Some(base_hint)));
        match cast_const(tcx, val, ety) {
            Ok(val) => val,
            Err(kind) => return Err(ConstEvalErr { span: e.span, kind: kind }),
        }
      }
      ast::ExprPath(..) => {
          let opt_def = tcx.def_map.borrow().get(&e.id).map(|d| d.full_def());
          let (const_expr, const_ty) = match opt_def {
              Some(def::DefConst(def_id)) => {
                  if ast_util::is_local(def_id) {
                      match tcx.map.find(def_id.node) {
                          Some(ast_map::NodeItem(it)) => match it.node {
                              ast::ItemConst(ref ty, ref expr) => {
                                  (Some(&**expr), Some(&**ty))
                              }
                              _ => (None, None)
                          },
                          _ => (None, None)
                      }
                  } else {
                      (lookup_const_by_id(tcx, def_id), None)
                  }
              }
              Some(def::DefVariant(enum_def, variant_def, _)) => {
                  (lookup_variant_by_id(tcx, enum_def, variant_def), None)
              }
              _ => (None, None)
          };
          let const_expr = match const_expr {
              Some(actual_e) => actual_e,
              None => signal!(e, NonConstPath)
          };
          let ety = ety.or_else(|| const_ty.and_then(|ty| ast_ty_to_prim_ty(tcx, ty)));
          try!(eval_const_expr_partial(tcx, const_expr, ety))
      }
      ast::ExprLit(ref lit) => {
          lit_to_const(&**lit, ety)
      }
      ast::ExprParen(ref e) => try!(eval_const_expr_partial(tcx, &**e, ety)),
      ast::ExprBlock(ref block) => {
        match block.expr {
            Some(ref expr) => try!(eval_const_expr_partial(tcx, &**expr, ety)),
            None => const_int(0)
        }
      }
      ast::ExprTup(_) => {
        const_val::Tuple(e.id)
      }
      ast::ExprStruct(..) => {
        const_val::Struct(e.id)
      }
      ast::ExprTupField(ref base, index) => {
        if let Ok(c) = eval_const_expr_partial(tcx, base, None) {
            if let const_val::Tuple(tup_id) = c {
                if let ast::ExprTup(ref fields) = tcx.map.expect_expr(tup_id).node {
                    if index.node < fields.len() {
                        return eval_const_expr_partial(tcx, &fields[index.node], None)
                    } else {
                        signal!(e, TupleIndexOutOfBounds);
                    }
                } else {
                    unreachable!()
                }
            } else {
                signal!(base, ExpectedConstTuple);
            }
        } else {
            signal!(base, NonConstPath)
        }
      }
      ast::ExprField(ref base, field_name) => {
        // Get the base expression if it is a struct and it is constant
        if let Ok(c) = eval_const_expr_partial(tcx, base, None) {
            if let const_val::Struct(struct_id) = c {
                if let ast::ExprStruct(_, ref fields, _) = tcx.map.expect_expr(struct_id).node {
                    // Check that the given field exists and evaluate it
                    if let Some(f) = fields.iter().find(|f| f.ident.node.as_str()
                                                         == field_name.node.as_str()) {
                        return eval_const_expr_partial(tcx, &*f.expr, None)
                    } else {
                        signal!(e, MissingStructField);
                    }
                } else {
                    unreachable!()
                }
            } else {
                signal!(base, ExpectedConstStruct);
            }
        } else {
            signal!(base, NonConstPath);
        }
      }
      _ => signal!(e, MiscCatchAll)
    };

    Ok(result)
}

fn cast_const<'tcx>(tcx: &ty::ctxt<'tcx>, val: const_val, ty: Ty) -> CastResult {
    macro_rules! convert_val {
        ($intermediate_ty:ty, $const_type:ident, $target_ty:ty) => {
            match val {
                const_bool(b) => Ok($const_type(b as $intermediate_ty as $target_ty)),
                const_uint(u) => Ok($const_type(u as $intermediate_ty as $target_ty)),
                const_int(i) => Ok($const_type(i as $intermediate_ty as $target_ty)),
                const_float(f) => Ok($const_type(f as $intermediate_ty as $target_ty)),
                _ => Err(ErrKind::CannotCastTo(stringify!($const_type))),
            }
        }
    }

    // Issue #23890: If isize/usize, then dispatch to appropriate target representation type
    match (&ty.sty, tcx.sess.target.int_type, tcx.sess.target.uint_type) {
        (&ty::ty_int(ast::TyIs), ast::TyI32, _) => return convert_val!(i32, const_int, i64),
        (&ty::ty_int(ast::TyIs), ast::TyI64, _) => return convert_val!(i64, const_int, i64),
        (&ty::ty_int(ast::TyIs), _, _) => panic!("unexpected target.int_type"),

        (&ty::ty_uint(ast::TyUs), _, ast::TyU32) => return convert_val!(u32, const_uint, u64),
        (&ty::ty_uint(ast::TyUs), _, ast::TyU64) => return convert_val!(u64, const_uint, u64),
        (&ty::ty_uint(ast::TyUs), _, _) => panic!("unexpected target.uint_type"),

        _ => {}
    }

    match ty.sty {
        ty::ty_int(ast::TyIs) => unreachable!(),
        ty::ty_uint(ast::TyUs) => unreachable!(),

        ty::ty_int(ast::TyI8) => convert_val!(i8, const_int, i64),
        ty::ty_int(ast::TyI16) => convert_val!(i16, const_int, i64),
        ty::ty_int(ast::TyI32) => convert_val!(i32, const_int, i64),
        ty::ty_int(ast::TyI64) => convert_val!(i64, const_int, i64),

        ty::ty_uint(ast::TyU8) => convert_val!(u8, const_uint, u64),
        ty::ty_uint(ast::TyU16) => convert_val!(u16, const_uint, u64),
        ty::ty_uint(ast::TyU32) => convert_val!(u32, const_uint, u64),
        ty::ty_uint(ast::TyU64) => convert_val!(u64, const_uint, u64),

        ty::ty_float(ast::TyF32) => convert_val!(f32, const_float, f64),
        ty::ty_float(ast::TyF64) => convert_val!(f64, const_float, f64),
        _ => Err(ErrKind::CannotCast),
    }
}

fn lit_to_const(lit: &ast::Lit, ty_hint: Option<Ty>) -> const_val {
    match lit.node {
        ast::LitStr(ref s, _) => const_str((*s).clone()),
        ast::LitBinary(ref data) => {
            const_binary(data.clone())
        }
        ast::LitByte(n) => const_uint(n as u64),
        ast::LitChar(n) => const_uint(n as u64),
        ast::LitInt(n, ast::SignedIntLit(_, ast::Plus)) => const_int(n as i64),
        ast::LitInt(n, ast::UnsuffixedIntLit(ast::Plus)) => {
            match ty_hint.map(|ty| &ty.sty) {
                Some(&ty::ty_uint(_)) => const_uint(n),
                _ => const_int(n as i64)
            }
        }
        ast::LitInt(n, ast::SignedIntLit(_, ast::Minus)) |
        ast::LitInt(n, ast::UnsuffixedIntLit(ast::Minus)) => const_int(-(n as i64)),
        ast::LitInt(n, ast::UnsignedIntLit(_)) => const_uint(n),
        ast::LitFloat(ref n, _) |
        ast::LitFloatUnsuffixed(ref n) => {
            const_float(n.parse::<f64>().unwrap() as f64)
        }
        ast::LitBool(b) => const_bool(b)
    }
}

pub fn compare_const_vals(a: &const_val, b: &const_val) -> Option<Ordering> {
    Some(match (a, b) {
        (&const_int(a), &const_int(b)) => a.cmp(&b),
        (&const_uint(a), &const_uint(b)) => a.cmp(&b),
        (&const_float(a), &const_float(b)) => {
            // This is pretty bad but it is the existing behavior.
            if a == b {
                Ordering::Equal
            } else if a < b {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (&const_str(ref a), &const_str(ref b)) => a.cmp(b),
        (&const_bool(a), &const_bool(b)) => a.cmp(&b),
        (&const_binary(ref a), &const_binary(ref b)) => a.cmp(b),
        _ => return None
    })
}

pub fn compare_lit_exprs<'tcx>(tcx: &ty::ctxt<'tcx>,
                               a: &Expr,
                               b: &Expr,
                               ty_hint: Option<Ty<'tcx>>)
                               -> Option<Ordering> {
    let a = match eval_const_expr_partial(tcx, a, ty_hint) {
        Ok(a) => a,
        Err(e) => {
            tcx.sess.span_err(a.span, &e.description());
            return None;
        }
    };
    let b = match eval_const_expr_partial(tcx, b, ty_hint) {
        Ok(b) => b,
        Err(e) => {
            tcx.sess.span_err(b.span, &e.description());
            return None;
        }
    };
    compare_const_vals(&a, &b)
}
