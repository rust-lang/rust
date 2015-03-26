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

use metadata::csearch;
use middle::{astencode, def};
use middle::pat_util::def_to_path;
use middle::ty::{self, Ty};
use middle::astconv_util::ast_ty_to_prim_ty;

use syntax::ast::{self, Expr};
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::ptr::P;
use syntax::{ast_map, ast_util, codemap};

use std::borrow::{Cow, IntoCow};
use std::num::wrapping::OverflowingOps;
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

macro_rules! signal {
    ($e:expr, $ctor:ident) => {
        return Err(ConstEvalErr { span: $e.span, kind: ErrKind::$ctor })
    };

    ($e:expr, $ctor:ident($($arg:expr),*)) => {
        return Err(ConstEvalErr { span: $e.span, kind: ErrKind::$ctor($($arg),*) })
    }
}

fn checked_add_int(e: &Expr, a: i64, b: i64) -> Result<const_val, ConstEvalErr> {
    let (ret, oflo) = a.overflowing_add(b);
    if !oflo { Ok(const_int(ret)) } else { signal!(e, AddiWithOverflow(a, b)) }
}
fn checked_sub_int(e: &Expr, a: i64, b: i64) -> Result<const_val, ConstEvalErr> {
    let (ret, oflo) = a.overflowing_sub(b);
    if !oflo { Ok(const_int(ret)) } else { signal!(e, SubiWithOverflow(a, b)) }
}
fn checked_mul_int(e: &Expr, a: i64, b: i64) -> Result<const_val, ConstEvalErr> {
    let (ret, oflo) = a.overflowing_mul(b);
    if !oflo { Ok(const_int(ret)) } else { signal!(e, MuliWithOverflow(a, b)) }
}

fn checked_add_uint(e: &Expr, a: u64, b: u64) -> Result<const_val, ConstEvalErr> {
    let (ret, oflo) = a.overflowing_add(b);
    if !oflo { Ok(const_uint(ret)) } else { signal!(e, AdduWithOverflow(a, b)) }
}
fn checked_sub_uint(e: &Expr, a: u64, b: u64) -> Result<const_val, ConstEvalErr> {
    let (ret, oflo) = a.overflowing_sub(b);
    if !oflo { Ok(const_uint(ret)) } else { signal!(e, SubuWithOverflow(a, b)) }
}
fn checked_mul_uint(e: &Expr, a: u64, b: u64) -> Result<const_val, ConstEvalErr> {
    let (ret, oflo) = a.overflowing_mul(b);
    if !oflo { Ok(const_uint(ret)) } else { signal!(e, MuluWithOverflow(a, b)) }
}


pub fn eval_const_expr_partial<'tcx>(tcx: &ty::ctxt<'tcx>,
                                     e: &Expr,
                                     ty_hint: Option<Ty<'tcx>>)
                                     -> Result<const_val, ConstEvalErr> {
    fn fromb(b: bool) -> const_val { const_int(b as i64) }

    let ety = ty_hint.or_else(|| ty::expr_ty_opt(tcx, e));

    let result = match e.node {
      ast::ExprUnary(ast::UnNeg, ref inner) => {
        match try!(eval_const_expr_partial(tcx, &**inner, ety)) {
          const_float(f) => const_float(-f),
          const_int(i) => const_int(-i),
          const_uint(i) => const_uint(-i),
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
            let is_a_min_value = || {
                let int_ty = match ty::expr_ty_opt(tcx, e).map(|ty| &ty.sty) {
                    Some(&ty::ty_int(int_ty)) => int_ty,
                    _ => return false
                };
                let int_ty = if let ast::TyIs = int_ty {
                    tcx.sess.target.int_type
                } else {
                    int_ty
                };
                match int_ty {
                    ast::TyI8 => (a as i8) == i8::MIN,
                    ast::TyI16 =>  (a as i16) == i16::MIN,
                    ast::TyI32 =>  (a as i32) == i32::MIN,
                    ast::TyI64 =>  (a as i64) == i64::MIN,
                    ast::TyIs => unreachable!()
                }
            };
            match op.node {
              ast::BiAdd => try!(checked_add_int(e, a, b)),
              ast::BiSub => try!(checked_sub_int(e, a, b)),
              ast::BiMul => try!(checked_mul_int(e, a, b)),
              ast::BiDiv => {
                  if b == 0 {
                      signal!(e, DivideByZero);
                  } else if b == -1 && is_a_min_value() {
                      signal!(e, DivideWithOverflow);
                  } else {
                      const_int(a / b)
                  }
              }
              ast::BiRem => {
                  if b == 0 {
                      signal!(e, ModuloByZero)
                  } else if b == -1 && is_a_min_value() {
                      signal!(e, ModuloWithOverflow)
                  } else {
                      const_int(a % b)
                  }
              }
              ast::BiAnd | ast::BiBitAnd => const_int(a & b),
              ast::BiOr | ast::BiBitOr => const_int(a | b),
              ast::BiBitXor => const_int(a ^ b),
              ast::BiShl => const_int(a << b as usize),
              ast::BiShr => const_int(a >> b as usize),
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
              ast::BiAdd => try!(checked_add_uint(e, a, b)),
              ast::BiSub => try!(checked_sub_uint(e, a, b)),
              ast::BiMul => try!(checked_mul_uint(e, a, b)),
              ast::BiDiv if b == 0 => signal!(e, DivideByZero),
              ast::BiDiv => const_uint(a / b),
              ast::BiRem if b == 0 => signal!(e, ModuloByZero),
              ast::BiRem => const_uint(a % b),
              ast::BiAnd | ast::BiBitAnd => const_uint(a & b),
              ast::BiOr | ast::BiBitOr => const_uint(a | b),
              ast::BiBitXor => const_uint(a ^ b),
              ast::BiShl => const_uint(a << b as usize),
              ast::BiShr => const_uint(a >> b as usize),
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
              ast::BiShl => const_int(a << b as usize),
              ast::BiShr => const_int(a >> b as usize),
              _ => signal!(e, InvalidOpForIntUint(op.node)),
            }
          }
          (const_uint(a), const_int(b)) => {
            match op.node {
              ast::BiShl => const_uint(a << b as usize),
              ast::BiShr => const_uint(a >> b as usize),
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
        let base_hint = ty::expr_ty_opt(tcx, &**base).unwrap_or(ety);
        let val = try!(eval_const_expr_partial(tcx, &**base, Some(base_hint)));
        match cast_const(val, ety) {
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

fn cast_const(val: const_val, ty: Ty) -> Result<const_val, ErrKind> {
    macro_rules! define_casts {
        ($($ty_pat:pat => (
            $intermediate_ty:ty,
            $const_type:ident,
            $target_ty:ty
        )),*) => (match ty.sty {
            $($ty_pat => {
                match val {
                    const_bool(b) => Ok($const_type(b as $intermediate_ty as $target_ty)),
                    const_uint(u) => Ok($const_type(u as $intermediate_ty as $target_ty)),
                    const_int(i) => Ok($const_type(i as $intermediate_ty as $target_ty)),
                    const_float(f) => Ok($const_type(f as $intermediate_ty as $target_ty)),
                    _ => Err(ErrKind::CannotCastTo(stringify!($const_type))),
                }
            },)*
            _ => Err(ErrKind::CannotCast),
        })
    }

    define_casts!{
        ty::ty_int(ast::TyIs) => (isize, const_int, i64),
        ty::ty_int(ast::TyI8) => (i8, const_int, i64),
        ty::ty_int(ast::TyI16) => (i16, const_int, i64),
        ty::ty_int(ast::TyI32) => (i32, const_int, i64),
        ty::ty_int(ast::TyI64) => (i64, const_int, i64),
        ty::ty_uint(ast::TyUs) => (usize, const_uint, u64),
        ty::ty_uint(ast::TyU8) => (u8, const_uint, u64),
        ty::ty_uint(ast::TyU16) => (u16, const_uint, u64),
        ty::ty_uint(ast::TyU32) => (u32, const_uint, u64),
        ty::ty_uint(ast::TyU64) => (u64, const_uint, u64),
        ty::ty_float(ast::TyF32) => (f32, const_float, f64),
        ty::ty_float(ast::TyF64) => (f64, const_float, f64)
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
