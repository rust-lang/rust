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
use middle::astconv_util::{ast_ty_to_prim_ty};

use syntax::ast::{self, Expr};
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::ptr::P;
use syntax::{ast_map, ast_util, codemap};

use std::cmp::Ordering;
use std::collections::hash_map::Entry::Vacant;
use std::{i8, i16, i32, i64};
use std::rc::Rc;

fn lookup_const<'a>(tcx: &'a ty::ctxt, e: &Expr) -> Option<&'a Expr> {
    let opt_def = tcx.def_map.borrow().get(&e.id).cloned();
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
            box |a, b, c, d| astencode::decode_inlined_item(a, b, c, d)) {
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
            box |a, b, c, d| astencode::decode_inlined_item(a, b, c, d)) {
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

// FIXME (#33): this doesn't handle big integer/float literals correctly
// (nor does the rest of our literal handling).
#[derive(Clone, PartialEq)]
pub enum const_val {
    const_float(f64),
    const_int(i64),
    const_uint(u64),
    const_str(InternedString),
    const_binary(Rc<Vec<u8> >),
    const_bool(bool)
}

pub fn const_expr_to_pat(tcx: &ty::ctxt, expr: &Expr, span: Span) -> P<ast::Pat> {
    let pat = match expr.node {
        ast::ExprTup(ref exprs) =>
            ast::PatTup(exprs.iter().map(|expr| const_expr_to_pat(tcx, &**expr, span)).collect()),

        ast::ExprCall(ref callee, ref args) => {
            let def = tcx.def_map.borrow()[callee.id].clone();
            if let Vacant(entry) = tcx.def_map.borrow_mut().entry(expr.id) {
               entry.insert(def);
            }
            let path = match def {
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

        ast::ExprPath(ref path) => {
            let opt_def = tcx.def_map.borrow().get(&expr.id).cloned();
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

        ast::ExprQPath(_) => {
            match lookup_const(tcx, expr) {
                Some(actual) => return const_expr_to_pat(tcx, actual, span),
                _ => unreachable!()
            }
        }

        _ => ast::PatLit(P(expr.clone()))
    };
    P(ast::Pat { id: expr.id, node: pat, span: span })
}

pub fn eval_const_expr(tcx: &ty::ctxt, e: &Expr) -> const_val {
    match eval_const_expr_partial(tcx, e, None) {
        Ok(r) => r,
        Err(s) => tcx.sess.span_fatal(e.span, &s[..])
    }
}

pub fn eval_const_expr_partial<'tcx>(tcx: &ty::ctxt<'tcx>,
                                     e: &Expr,
                                     ty_hint: Option<Ty<'tcx>>)
                                     -> Result<const_val, String> {
    fn fromb(b: bool) -> Result<const_val, String> { Ok(const_int(b as i64)) }

    let ety = ty_hint.or_else(|| ty::expr_ty_opt(tcx, e));

    match e.node {
      ast::ExprUnary(ast::UnNeg, ref inner) => {
        match eval_const_expr_partial(tcx, &**inner, ety) {
          Ok(const_float(f)) => Ok(const_float(-f)),
          Ok(const_int(i)) => Ok(const_int(-i)),
          Ok(const_uint(i)) => Ok(const_uint(-i)),
          Ok(const_str(_)) => Err("negate on string".to_string()),
          Ok(const_bool(_)) => Err("negate on boolean".to_string()),
          ref err => ((*err).clone())
        }
      }
      ast::ExprUnary(ast::UnNot, ref inner) => {
        match eval_const_expr_partial(tcx, &**inner, ety) {
          Ok(const_int(i)) => Ok(const_int(!i)),
          Ok(const_uint(i)) => Ok(const_uint(!i)),
          Ok(const_bool(b)) => Ok(const_bool(!b)),
          _ => Err("not on float or string".to_string())
        }
      }
      ast::ExprBinary(op, ref a, ref b) => {
        let b_ty = match op.node {
            ast::BiShl | ast::BiShr => Some(tcx.types.uint),
            _ => ety
        };
        match (eval_const_expr_partial(tcx, &**a, ety),
               eval_const_expr_partial(tcx, &**b, b_ty)) {
          (Ok(const_float(a)), Ok(const_float(b))) => {
            match op.node {
              ast::BiAdd => Ok(const_float(a + b)),
              ast::BiSub => Ok(const_float(a - b)),
              ast::BiMul => Ok(const_float(a * b)),
              ast::BiDiv => Ok(const_float(a / b)),
              ast::BiRem => Ok(const_float(a % b)),
              ast::BiEq => fromb(a == b),
              ast::BiLt => fromb(a < b),
              ast::BiLe => fromb(a <= b),
              ast::BiNe => fromb(a != b),
              ast::BiGe => fromb(a >= b),
              ast::BiGt => fromb(a > b),
              _ => Err("can't do this op on floats".to_string())
            }
          }
          (Ok(const_int(a)), Ok(const_int(b))) => {
            let is_a_min_value = |&:| {
                let int_ty = match ty::expr_ty_opt(tcx, e).map(|ty| &ty.sty) {
                    Some(&ty::ty_int(int_ty)) => int_ty,
                    _ => return false
                };
                let int_ty = if let ast::TyIs(_) = int_ty {
                    tcx.sess.target.int_type
                } else {
                    int_ty
                };
                match int_ty {
                    ast::TyI8 => (a as i8) == i8::MIN,
                    ast::TyI16 =>  (a as i16) == i16::MIN,
                    ast::TyI32 =>  (a as i32) == i32::MIN,
                    ast::TyI64 =>  (a as i64) == i64::MIN,
                    ast::TyIs(_) => unreachable!()
                }
            };
            match op.node {
              ast::BiAdd => Ok(const_int(a + b)),
              ast::BiSub => Ok(const_int(a - b)),
              ast::BiMul => Ok(const_int(a * b)),
              ast::BiDiv => {
                  if b == 0 {
                      Err("attempted to divide by zero".to_string())
                  } else if b == -1 && is_a_min_value() {
                      Err("attempted to divide with overflow".to_string())
                  } else {
                      Ok(const_int(a / b))
                  }
              }
              ast::BiRem => {
                  if b == 0 {
                      Err("attempted remainder with a divisor of zero".to_string())
                  } else if b == -1 && is_a_min_value() {
                      Err("attempted remainder with overflow".to_string())
                  } else {
                      Ok(const_int(a % b))
                  }
              }
              ast::BiAnd | ast::BiBitAnd => Ok(const_int(a & b)),
              ast::BiOr | ast::BiBitOr => Ok(const_int(a | b)),
              ast::BiBitXor => Ok(const_int(a ^ b)),
              ast::BiShl => Ok(const_int(a << b as uint)),
              ast::BiShr => Ok(const_int(a >> b as uint)),
              ast::BiEq => fromb(a == b),
              ast::BiLt => fromb(a < b),
              ast::BiLe => fromb(a <= b),
              ast::BiNe => fromb(a != b),
              ast::BiGe => fromb(a >= b),
              ast::BiGt => fromb(a > b)
            }
          }
          (Ok(const_uint(a)), Ok(const_uint(b))) => {
            match op.node {
              ast::BiAdd => Ok(const_uint(a + b)),
              ast::BiSub => Ok(const_uint(a - b)),
              ast::BiMul => Ok(const_uint(a * b)),
              ast::BiDiv if b == 0 => {
                  Err("attempted to divide by zero".to_string())
              }
              ast::BiDiv => Ok(const_uint(a / b)),
              ast::BiRem if b == 0 => {
                  Err("attempted remainder with a divisor of \
                       zero".to_string())
              }
              ast::BiRem => Ok(const_uint(a % b)),
              ast::BiAnd | ast::BiBitAnd => Ok(const_uint(a & b)),
              ast::BiOr | ast::BiBitOr => Ok(const_uint(a | b)),
              ast::BiBitXor => Ok(const_uint(a ^ b)),
              ast::BiShl => Ok(const_uint(a << b as uint)),
              ast::BiShr => Ok(const_uint(a >> b as uint)),
              ast::BiEq => fromb(a == b),
              ast::BiLt => fromb(a < b),
              ast::BiLe => fromb(a <= b),
              ast::BiNe => fromb(a != b),
              ast::BiGe => fromb(a >= b),
              ast::BiGt => fromb(a > b),
            }
          }
          // shifts can have any integral type as their rhs
          (Ok(const_int(a)), Ok(const_uint(b))) => {
            match op.node {
              ast::BiShl => Ok(const_int(a << b as uint)),
              ast::BiShr => Ok(const_int(a >> b as uint)),
              _ => Err("can't do this op on an int and uint".to_string())
            }
          }
          (Ok(const_uint(a)), Ok(const_int(b))) => {
            match op.node {
              ast::BiShl => Ok(const_uint(a << b as uint)),
              ast::BiShr => Ok(const_uint(a >> b as uint)),
              _ => Err("can't do this op on a uint and int".to_string())
            }
          }
          (Ok(const_bool(a)), Ok(const_bool(b))) => {
            Ok(const_bool(match op.node {
              ast::BiAnd => a && b,
              ast::BiOr => a || b,
              ast::BiBitXor => a ^ b,
              ast::BiBitAnd => a & b,
              ast::BiBitOr => a | b,
              ast::BiEq => a == b,
              ast::BiNe => a != b,
              _ => return Err("can't do this op on bools".to_string())
             }))
          }
          _ => Err("bad operands for binary".to_string())
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
        cast_const(val, ety)
      }
      ast::ExprPath(_) | ast::ExprQPath(_) => {
          let opt_def = tcx.def_map.borrow().get(&e.id).cloned();
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
              None => return Err("non-constant path in constant expr".to_string())
          };
          let ety = ety.or_else(|| const_ty.and_then(|ty| ast_ty_to_prim_ty(tcx, ty)));
          eval_const_expr_partial(tcx, const_expr, ety)
      }
      ast::ExprLit(ref lit) => {
          Ok(lit_to_const(&**lit, ety))
      }
      ast::ExprParen(ref e)     => eval_const_expr_partial(tcx, &**e, ety),
      ast::ExprBlock(ref block) => {
        match block.expr {
            Some(ref expr) => eval_const_expr_partial(tcx, &**expr, ety),
            None => Ok(const_int(0i64))
        }
      }
      ast::ExprTupField(ref base, index) => {
        // Get the base tuple if it is constant
        if let Some(&ast::ExprTup(ref fields)) = lookup_const(tcx, &**base).map(|s| &s.node) {
            // Check that the given index is within bounds and evaluate its value
            if fields.len() > index.node {
                return eval_const_expr_partial(tcx, &*fields[index.node], None)
            } else {
                return Err("tuple index out of bounds".to_string())
            }
        }

        Err("non-constant struct in constant expr".to_string())
      }
      ast::ExprField(ref base, field_name) => {
        // Get the base expression if it is a struct and it is constant
        if let Some(&ast::ExprStruct(_, ref fields, _)) = lookup_const(tcx, &**base)
                                                            .map(|s| &s.node) {
            // Check that the given field exists and evaluate it
            if let Some(f) = fields.iter().find(|f|
                                           f.ident.node.as_str() == field_name.node.as_str()) {
                return eval_const_expr_partial(tcx, &*f.expr, None)
            } else {
                return Err("nonexistent struct field".to_string())
            }
        }

        Err("non-constant struct in constant expr".to_string())
      }
      _ => Err("unsupported constant expr".to_string())
    }
}

fn cast_const(val: const_val, ty: Ty) -> Result<const_val, String> {
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
                    _ => Err(concat!("can't cast this type to ",
                                     stringify!($const_type)).to_string())
                }
            },)*
            _ => Err("can't cast this type".to_string())
        })
    }

    define_casts!{
        ty::ty_int(ast::TyIs(_)) => (int, const_int, i64),
        ty::ty_int(ast::TyI8) => (i8, const_int, i64),
        ty::ty_int(ast::TyI16) => (i16, const_int, i64),
        ty::ty_int(ast::TyI32) => (i32, const_int, i64),
        ty::ty_int(ast::TyI64) => (i64, const_int, i64),
        ty::ty_uint(ast::TyUs(_)) => (uint, const_uint, u64),
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
        Err(s) => {
            tcx.sess.span_err(a.span, &s[..]);
            return None;
        }
    };
    let b = match eval_const_expr_partial(tcx, b, ty_hint) {
        Ok(b) => b,
        Err(s) => {
            tcx.sess.span_err(b.span, &s[..]);
            return None;
        }
    };
    compare_const_vals(&a, &b)
}
