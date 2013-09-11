// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use metadata::csearch;
use middle::astencode;
use middle::ty;
use middle;

use syntax::{ast, ast_map, ast_util};
use syntax::visit;
use syntax::visit::Visitor;
use syntax::ast::*;

use std::float;
use std::hashmap::{HashMap, HashSet};

//
// This pass classifies expressions by their constant-ness.
//
// Constant-ness comes in 3 flavours:
//
//   - Integer-constants: can be evaluated by the frontend all the way down
//     to their actual value. They are used in a few places (enum
//     discriminants, switch arms) and are a subset of
//     general-constants. They cover all the integer and integer-ish
//     literals (nil, bool, int, uint, char, iNN, uNN) and all integer
//     operators and copies applied to them.
//
//   - General-constants: can be evaluated by LLVM but not necessarily by
//     the frontend; usually due to reliance on target-specific stuff such
//     as "where in memory the value goes" or "what floating point mode the
//     target uses". This _includes_ integer-constants, plus the following
//     constructors:
//
//        fixed-size vectors and strings: [] and ""/_
//        vector and string slices: &[] and &""
//        tuples: (,)
//        records: {...}
//        enums: foo(...)
//        floating point literals and operators
//        & and * pointers
//        copies of general constants
//
//        (in theory, probably not at first: if/match on integer-const
//         conditions / descriminants)
//
//   - Non-constants: everything else.
//

pub enum constness {
    integral_const,
    general_const,
    non_const
}

pub fn join(a: constness, b: constness) -> constness {
    match (a, b) {
      (integral_const, integral_const) => integral_const,
      (integral_const, general_const)
      | (general_const, integral_const)
      | (general_const, general_const) => general_const,
      _ => non_const
    }
}

pub fn join_all<It: Iterator<constness>>(mut cs: It) -> constness {
    cs.fold(integral_const, |a, b| join(a, b))
}

pub fn classify(e: &Expr,
                tcx: ty::ctxt)
             -> constness {
    let did = ast_util::local_def(e.id);
    match tcx.ccache.find(&did) {
      Some(&x) => x,
      None => {
        let cn =
            match e.node {
              ast::ExprLit(lit) => {
                match lit.node {
                  ast::lit_str(*) |
                  ast::lit_float(*) => general_const,
                  _ => integral_const
                }
              }

              ast::ExprUnary(_, _, inner) |
              ast::ExprParen(inner) => {
                classify(inner, tcx)
              }

              ast::ExprBinary(_, _, a, b) => {
                join(classify(a, tcx),
                     classify(b, tcx))
              }

              ast::ExprTup(ref es) |
              ast::ExprVec(ref es, ast::MutImmutable) => {
                join_all(es.iter().map(|e| classify(*e, tcx)))
              }

              ast::ExprVstore(e, vstore) => {
                  match vstore {
                      ast::ExprVstoreSlice => classify(e, tcx),
                      ast::ExprVstoreUniq |
                      ast::ExprVstoreBox |
                      ast::ExprVstoreMutBox |
                      ast::ExprVstoreMutSlice => non_const
                  }
              }

              ast::ExprStruct(_, ref fs, None) => {
                let cs = do fs.iter().map |f| {
                    classify(f.expr, tcx)
                };
                join_all(cs)
              }

              ast::ExprCast(base, _) => {
                let ty = ty::expr_ty(tcx, e);
                let base = classify(base, tcx);
                if ty::type_is_integral(ty) {
                    join(integral_const, base)
                } else if ty::type_is_fp(ty) {
                    join(general_const, base)
                } else {
                    non_const
                }
              }

              ast::ExprField(base, _, _) => {
                classify(base, tcx)
              }

              ast::ExprIndex(_, base, idx) => {
                join(classify(base, tcx),
                     classify(idx, tcx))
              }

              ast::ExprAddrOf(ast::MutImmutable, base) => {
                classify(base, tcx)
              }

              // FIXME: (#3728) we can probably do something CCI-ish
              // surrounding nonlocal constants. But we don't yet.
              ast::ExprPath(_) => {
                lookup_constness(tcx, e)
              }

              ast::ExprRepeat(*) => general_const,

              _ => non_const
            };
        tcx.ccache.insert(did, cn);
        cn
      }
    }
}

pub fn lookup_const(tcx: ty::ctxt, e: &Expr) -> Option<@Expr> {
    match tcx.def_map.find(&e.id) {
        Some(&ast::DefStatic(def_id, false)) => lookup_const_by_id(tcx, def_id),
        Some(&ast::DefVariant(enum_def, variant_def, _)) => lookup_variant_by_id(tcx,
                                                                               enum_def,
                                                                               variant_def),
        _ => None
    }
}

pub fn lookup_variant_by_id(tcx: ty::ctxt,
                            enum_def: ast::DefId,
                            variant_def: ast::DefId)
                       -> Option<@Expr> {
    fn variant_expr(variants: &[ast::variant], id: ast::NodeId) -> Option<@Expr> {
        for variant in variants.iter() {
            if variant.node.id == id {
                return variant.node.disr_expr;
            }
        }
        None
    }

    if ast_util::is_local(enum_def) {
        match tcx.items.find(&enum_def.node) {
            None => None,
            Some(&ast_map::node_item(it, _)) => match it.node {
                item_enum(ast::enum_def { variants: ref variants }, _) => {
                    variant_expr(*variants, variant_def.node)
                }
                _ => None
            },
            Some(_) => None
        }
    } else {
        let maps = astencode::Maps {
            root_map: @mut HashMap::new(),
            method_map: @mut HashMap::new(),
            vtable_map: @mut HashMap::new(),
            write_guard_map: @mut HashSet::new(),
            capture_map: @mut HashMap::new()
        };
        match csearch::maybe_get_item_ast(tcx, enum_def,
            |a, b, c, d| astencode::decode_inlined_item(a,
                                                        b,
                                                        maps,
                                                        /*bad*/ c.clone(),
                                                        d)) {
            csearch::found(ast::ii_item(item)) => match item.node {
                item_enum(ast::enum_def { variants: ref variants }, _) => {
                    variant_expr(*variants, variant_def.node)
                }
                _ => None
            },
            _ => None
        }
    }
}

pub fn lookup_const_by_id(tcx: ty::ctxt,
                          def_id: ast::DefId)
                       -> Option<@Expr> {
    if ast_util::is_local(def_id) {
        match tcx.items.find(&def_id.node) {
            None => None,
            Some(&ast_map::node_item(it, _)) => match it.node {
                item_static(_, ast::MutImmutable, const_expr) => Some(const_expr),
                _ => None
            },
            Some(_) => None
        }
    } else {
        let maps = astencode::Maps {
            root_map: @mut HashMap::new(),
            method_map: @mut HashMap::new(),
            vtable_map: @mut HashMap::new(),
            write_guard_map: @mut HashSet::new(),
            capture_map: @mut HashMap::new()
        };
        match csearch::maybe_get_item_ast(tcx, def_id,
            |a, b, c, d| astencode::decode_inlined_item(a, b, maps, c, d)) {
            csearch::found(ast::ii_item(item)) => match item.node {
                item_static(_, ast::MutImmutable, const_expr) => Some(const_expr),
                _ => None
            },
            _ => None
        }
    }
}

pub fn lookup_constness(tcx: ty::ctxt, e: &Expr) -> constness {
    match lookup_const(tcx, e) {
        Some(rhs) => {
            let ty = ty::expr_ty(tcx, rhs);
            if ty::type_is_integral(ty) {
                integral_const
            } else {
                general_const
            }
        }
        None => non_const
    }
}

struct ConstEvalVisitor { tcx: ty::ctxt }

impl Visitor<()> for ConstEvalVisitor {
    fn visit_expr_post(&mut self, e:@Expr, _:()) {
        classify(e, self.tcx);
    }
}

pub fn process_crate(crate: &ast::Crate,
                     tcx: ty::ctxt) {
    let mut v = ConstEvalVisitor { tcx: tcx };
    visit::walk_crate(&mut v, crate, ());
    tcx.sess.abort_if_errors();
}


// FIXME (#33): this doesn't handle big integer/float literals correctly
// (nor does the rest of our literal handling).
#[deriving(Clone, Eq)]
pub enum const_val {
    const_float(f64),
    const_int(i64),
    const_uint(u64),
    const_str(@str),
    const_bool(bool)
}

pub fn eval_const_expr(tcx: middle::ty::ctxt, e: &Expr) -> const_val {
    match eval_const_expr_partial(&tcx, e) {
        Ok(r) => r,
        Err(s) => tcx.sess.span_fatal(e.span, s)
    }
}

pub fn eval_const_expr_partial<T: ty::ExprTyProvider>(tcx: &T, e: &Expr)
                            -> Result<const_val, ~str> {
    use middle::ty;
    fn fromb(b: bool) -> Result<const_val, ~str> { Ok(const_int(b as i64)) }
    match e.node {
      ExprUnary(_, UnNeg, inner) => {
        match eval_const_expr_partial(tcx, inner) {
          Ok(const_float(f)) => Ok(const_float(-f)),
          Ok(const_int(i)) => Ok(const_int(-i)),
          Ok(const_uint(i)) => Ok(const_uint(-i)),
          Ok(const_str(_)) => Err(~"Negate on string"),
          Ok(const_bool(_)) => Err(~"Negate on boolean"),
          ref err => ((*err).clone())
        }
      }
      ExprUnary(_, UnNot, inner) => {
        match eval_const_expr_partial(tcx, inner) {
          Ok(const_int(i)) => Ok(const_int(!i)),
          Ok(const_uint(i)) => Ok(const_uint(!i)),
          Ok(const_bool(b)) => Ok(const_bool(!b)),
          _ => Err(~"Not on float or string")
        }
      }
      ExprBinary(_, op, a, b) => {
        match (eval_const_expr_partial(tcx, a),
               eval_const_expr_partial(tcx, b)) {
          (Ok(const_float(a)), Ok(const_float(b))) => {
            match op {
              BiAdd => Ok(const_float(a + b)),
              BiSub => Ok(const_float(a - b)),
              BiMul => Ok(const_float(a * b)),
              BiDiv => Ok(const_float(a / b)),
              BiRem => Ok(const_float(a % b)),
              BiEq => fromb(a == b),
              BiLt => fromb(a < b),
              BiLe => fromb(a <= b),
              BiNe => fromb(a != b),
              BiGe => fromb(a >= b),
              BiGt => fromb(a > b),
              _ => Err(~"Can't do this op on floats")
            }
          }
          (Ok(const_int(a)), Ok(const_int(b))) => {
            match op {
              BiAdd => Ok(const_int(a + b)),
              BiSub => Ok(const_int(a - b)),
              BiMul => Ok(const_int(a * b)),
              BiDiv if b == 0 => Err(~"attempted to divide by zero"),
              BiDiv => Ok(const_int(a / b)),
              BiRem if b == 0 => Err(~"attempted remainder with a divisor of zero"),
              BiRem => Ok(const_int(a % b)),
              BiAnd | BiBitAnd => Ok(const_int(a & b)),
              BiOr | BiBitOr => Ok(const_int(a | b)),
              BiBitXor => Ok(const_int(a ^ b)),
              BiShl => Ok(const_int(a << b)),
              BiShr => Ok(const_int(a >> b)),
              BiEq => fromb(a == b),
              BiLt => fromb(a < b),
              BiLe => fromb(a <= b),
              BiNe => fromb(a != b),
              BiGe => fromb(a >= b),
              BiGt => fromb(a > b)
            }
          }
          (Ok(const_uint(a)), Ok(const_uint(b))) => {
            match op {
              BiAdd => Ok(const_uint(a + b)),
              BiSub => Ok(const_uint(a - b)),
              BiMul => Ok(const_uint(a * b)),
              BiDiv if b == 0 => Err(~"attempted to divide by zero"),
              BiDiv => Ok(const_uint(a / b)),
              BiRem if b == 0 => Err(~"attempted remainder with a divisor of zero"),
              BiRem => Ok(const_uint(a % b)),
              BiAnd | BiBitAnd => Ok(const_uint(a & b)),
              BiOr | BiBitOr => Ok(const_uint(a | b)),
              BiBitXor => Ok(const_uint(a ^ b)),
              BiShl => Ok(const_uint(a << b)),
              BiShr => Ok(const_uint(a >> b)),
              BiEq => fromb(a == b),
              BiLt => fromb(a < b),
              BiLe => fromb(a <= b),
              BiNe => fromb(a != b),
              BiGe => fromb(a >= b),
              BiGt => fromb(a > b),
            }
          }
          // shifts can have any integral type as their rhs
          (Ok(const_int(a)), Ok(const_uint(b))) => {
            match op {
              BiShl => Ok(const_int(a << b)),
              BiShr => Ok(const_int(a >> b)),
              _ => Err(~"Can't do this op on an int and uint")
            }
          }
          (Ok(const_uint(a)), Ok(const_int(b))) => {
            match op {
              BiShl => Ok(const_uint(a << b)),
              BiShr => Ok(const_uint(a >> b)),
              _ => Err(~"Can't do this op on a uint and int")
            }
          }
          (Ok(const_bool(a)), Ok(const_bool(b))) => {
            Ok(const_bool(match op {
              BiAnd => a && b,
              BiOr => a || b,
              BiBitXor => a ^ b,
              BiBitAnd => a & b,
              BiBitOr => a | b,
              BiEq => a == b,
              BiNe => a != b,
              _ => return Err(~"Can't do this op on bools")
             }))
          }
          _ => Err(~"Bad operands for binary")
        }
      }
      ExprCast(base, _) => {
        let ety = tcx.expr_ty(e);
        let base = eval_const_expr_partial(tcx, base);
        match base {
            Err(_) => base,
            Ok(val) => {
                match ty::get(ety).sty {
                    ty::ty_float(_) => {
                        match val {
                            const_uint(u) => Ok(const_float(u as f64)),
                            const_int(i) => Ok(const_float(i as f64)),
                            const_float(f) => Ok(const_float(f)),
                            _ => Err(~"Can't cast float to str"),
                        }
                    }
                    ty::ty_uint(_) => {
                        match val {
                            const_uint(u) => Ok(const_uint(u)),
                            const_int(i) => Ok(const_uint(i as u64)),
                            const_float(f) => Ok(const_uint(f as u64)),
                            _ => Err(~"Can't cast str to uint"),
                        }
                    }
                    ty::ty_int(_) | ty::ty_bool => {
                        match val {
                            const_uint(u) => Ok(const_int(u as i64)),
                            const_int(i) => Ok(const_int(i)),
                            const_float(f) => Ok(const_int(f as i64)),
                            _ => Err(~"Can't cast str to int"),
                        }
                    }
                    _ => Err(~"Can't cast this type")
                }
            }
        }
      }
      ExprPath(_) => {
          match lookup_const(tcx.ty_ctxt(), e) {
              Some(actual_e) => eval_const_expr_partial(&tcx.ty_ctxt(), actual_e),
              None => Err(~"Non-constant path in constant expr")
          }
      }
      ExprLit(lit) => Ok(lit_to_const(lit)),
      // If we have a vstore, just keep going; it has to be a string
      ExprVstore(e, _) => eval_const_expr_partial(tcx, e),
      ExprParen(e)     => eval_const_expr_partial(tcx, e),
      _ => Err(~"Unsupported constant expr")
    }
}

pub fn lit_to_const(lit: &lit) -> const_val {
    match lit.node {
      lit_str(s) => const_str(s),
      lit_char(n) => const_uint(n as u64),
      lit_int(n, _) => const_int(n),
      lit_uint(n, _) => const_uint(n),
      lit_int_unsuffixed(n) => const_int(n),
      lit_float(n, _) => const_float(float::from_str(n).unwrap() as f64),
      lit_float_unsuffixed(n) =>
        const_float(float::from_str(n).unwrap() as f64),
      lit_nil => const_int(0i64),
      lit_bool(b) => const_bool(b)
    }
}

fn compare_vals<T : Eq + Ord>(a: T, b: T) -> Option<int> {
    Some(if a == b { 0 } else if a < b { -1 } else { 1 })
}
pub fn compare_const_vals(a: &const_val, b: &const_val) -> Option<int> {
    match (a, b) {
        (&const_int(a), &const_int(b)) => compare_vals(a, b),
        (&const_uint(a), &const_uint(b)) => compare_vals(a, b),
        (&const_float(a), &const_float(b)) => compare_vals(a, b),
        (&const_str(a), &const_str(b)) => compare_vals(a, b),
        (&const_bool(a), &const_bool(b)) => compare_vals(a, b),
        _ => None
    }
}

pub fn compare_lit_exprs(tcx: middle::ty::ctxt, a: &Expr, b: &Expr) -> Option<int> {
    compare_const_vals(&eval_const_expr(tcx, a), &eval_const_expr(tcx, b))
}

pub fn lit_expr_eq(tcx: middle::ty::ctxt, a: &Expr, b: &Expr) -> Option<bool> {
    compare_lit_exprs(tcx, a, b).map_move(|val| val == 0)
}

pub fn lit_eq(a: &lit, b: &lit) -> Option<bool> {
    compare_const_vals(&lit_to_const(a), &lit_to_const(b)).map_move(|val| val == 0)
}
