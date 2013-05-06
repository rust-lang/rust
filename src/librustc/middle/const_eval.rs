// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use syntax::{ast, ast_map, ast_util, visit};
use syntax::ast::*;

use core::hashmap::{HashMap, HashSet};

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

pub fn join_all(cs: &[constness]) -> constness {
    vec::foldl(integral_const, cs, |a, b| join(a, *b))
}

pub fn classify(e: @expr,
                tcx: ty::ctxt)
             -> constness {
    let did = ast_util::local_def(e.id);
    match tcx.ccache.find(&did) {
      Some(&x) => x,
      None => {
        let cn =
            match e.node {
              ast::expr_lit(lit) => {
                match lit.node {
                  ast::lit_str(*) |
                  ast::lit_float(*) => general_const,
                  _ => integral_const
                }
              }

              ast::expr_copy(inner) |
              ast::expr_unary(_, inner) |
              ast::expr_paren(inner) => {
                classify(inner, tcx)
              }

              ast::expr_binary(_, a, b) => {
                join(classify(a, tcx),
                     classify(b, tcx))
              }

              ast::expr_tup(ref es) |
              ast::expr_vec(ref es, ast::m_imm) => {
                join_all(vec::map(*es, |e| classify(*e, tcx)))
              }

              ast::expr_vstore(e, vstore) => {
                  match vstore {
                      ast::expr_vstore_slice => classify(e, tcx),
                      ast::expr_vstore_uniq |
                      ast::expr_vstore_box |
                      ast::expr_vstore_mut_box |
                      ast::expr_vstore_mut_slice => non_const
                  }
              }

              ast::expr_struct(_, ref fs, None) => {
                let cs = do vec::map((*fs)) |f| {
                    if f.node.mutbl == ast::m_imm {
                        classify(f.node.expr, tcx)
                    } else {
                        non_const
                    }
                };
                join_all(cs)
              }

              ast::expr_cast(base, _) => {
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

              ast::expr_field(base, _, _) => {
                classify(base, tcx)
              }

              ast::expr_index(base, idx) => {
                join(classify(base, tcx),
                     classify(idx, tcx))
              }

              ast::expr_addr_of(ast::m_imm, base) => {
                classify(base, tcx)
              }

              // FIXME: (#3728) we can probably do something CCI-ish
              // surrounding nonlocal constants. But we don't yet.
              ast::expr_path(_) => {
                lookup_constness(tcx, e)
              }

              _ => non_const
            };
        tcx.ccache.insert(did, cn);
        cn
      }
    }
}

pub fn lookup_const(tcx: ty::ctxt, e: @expr) -> Option<@expr> {
    match tcx.def_map.find(&e.id) {
        Some(&ast::def_const(def_id)) => lookup_const_by_id(tcx, def_id),
        _ => None
    }
}

pub fn lookup_const_by_id(tcx: ty::ctxt,
                          def_id: ast::def_id)
                       -> Option<@expr> {
    if ast_util::is_local(def_id) {
        match tcx.items.find(&def_id.node) {
            None => None,
            Some(&ast_map::node_item(it, _)) => match it.node {
                item_const(_, const_expr) => Some(const_expr),
                _ => None
            },
            Some(_) => None
        }
    } else {
        let maps = astencode::Maps {
            mutbl_map: @mut HashSet::new(),
            root_map: @mut HashMap::new(),
            last_use_map: @mut HashMap::new(),
            method_map: @mut HashMap::new(),
            vtable_map: @mut HashMap::new(),
            write_guard_map: @mut HashSet::new(),
            moves_map: @mut HashSet::new(),
            capture_map: @mut HashMap::new()
        };
        match csearch::maybe_get_item_ast(tcx, def_id,
            |a, b, c, d| astencode::decode_inlined_item(a, b, maps, /*bar*/ copy c, d)) {
            csearch::found(ast::ii_item(item)) => match item.node {
                item_const(_, const_expr) => Some(const_expr),
                _ => None
            },
            _ => None
        }
    }
}

pub fn lookup_constness(tcx: ty::ctxt, e: @expr) -> constness {
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

pub fn process_crate(crate: @ast::crate,
                     tcx: ty::ctxt) {
    let v = visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_expr_post: |e| { classify(e, tcx); },
        .. *visit::default_simple_visitor()
    });
    visit::visit_crate(crate, (), v);
    tcx.sess.abort_if_errors();
}


// FIXME (#33): this doesn't handle big integer/float literals correctly
// (nor does the rest of our literal handling).
#[deriving(Eq)]
pub enum const_val {
    const_float(f64),
    const_int(i64),
    const_uint(u64),
    const_str(~str),
    const_bool(bool)
}

pub fn eval_const_expr(tcx: middle::ty::ctxt, e: @expr) -> const_val {
    match eval_const_expr_partial(tcx, e) {
        Ok(ref r) => (/*bad*/copy *r),
        Err(ref s) => fail!(/*bad*/copy *s)
    }
}

pub fn eval_const_expr_partial(tcx: middle::ty::ctxt, e: @expr)
                            -> Result<const_val, ~str> {
    use middle::ty;
    fn fromb(b: bool) -> Result<const_val, ~str> { Ok(const_int(b as i64)) }
    match e.node {
      expr_unary(neg, inner) => {
        match eval_const_expr_partial(tcx, inner) {
          Ok(const_float(f)) => Ok(const_float(-f)),
          Ok(const_int(i)) => Ok(const_int(-i)),
          Ok(const_uint(i)) => Ok(const_uint(-i)),
          Ok(const_str(_)) => Err(~"Negate on string"),
          Ok(const_bool(_)) => Err(~"Negate on boolean"),
          ref err => (/*bad*/copy *err)
        }
      }
      expr_unary(not, inner) => {
        match eval_const_expr_partial(tcx, inner) {
          Ok(const_int(i)) => Ok(const_int(!i)),
          Ok(const_uint(i)) => Ok(const_uint(!i)),
          Ok(const_bool(b)) => Ok(const_bool(!b)),
          _ => Err(~"Not on float or string")
        }
      }
      expr_binary(op, a, b) => {
        match (eval_const_expr_partial(tcx, a),
               eval_const_expr_partial(tcx, b)) {
          (Ok(const_float(a)), Ok(const_float(b))) => {
            match op {
              add => Ok(const_float(a + b)),
              subtract => Ok(const_float(a - b)),
              mul => Ok(const_float(a * b)),
              div => Ok(const_float(a / b)),
              rem => Ok(const_float(a % b)),
              eq => fromb(a == b),
              lt => fromb(a < b),
              le => fromb(a <= b),
              ne => fromb(a != b),
              ge => fromb(a >= b),
              gt => fromb(a > b),
              _ => Err(~"Can't do this op on floats")
            }
          }
          (Ok(const_int(a)), Ok(const_int(b))) => {
            match op {
              add => Ok(const_int(a + b)),
              subtract => Ok(const_int(a - b)),
              mul => Ok(const_int(a * b)),
              div if b == 0 => Err(~"attempted to divide by zero"),
              div => Ok(const_int(a / b)),
              rem if b == 0 => Err(~"attempted remainder with a divisor of zero"),
              rem => Ok(const_int(a % b)),
              and | bitand => Ok(const_int(a & b)),
              or | bitor => Ok(const_int(a | b)),
              bitxor => Ok(const_int(a ^ b)),
              shl => Ok(const_int(a << b)),
              shr => Ok(const_int(a >> b)),
              eq => fromb(a == b),
              lt => fromb(a < b),
              le => fromb(a <= b),
              ne => fromb(a != b),
              ge => fromb(a >= b),
              gt => fromb(a > b)
            }
          }
          (Ok(const_uint(a)), Ok(const_uint(b))) => {
            match op {
              add => Ok(const_uint(a + b)),
              subtract => Ok(const_uint(a - b)),
              mul => Ok(const_uint(a * b)),
              div if b == 0 => Err(~"attempted to divide by zero"),
              div => Ok(const_uint(a / b)),
              rem if b == 0 => Err(~"attempted remainder with a divisor of zero"),
              rem => Ok(const_uint(a % b)),
              and | bitand => Ok(const_uint(a & b)),
              or | bitor => Ok(const_uint(a | b)),
              bitxor => Ok(const_uint(a ^ b)),
              shl => Ok(const_uint(a << b)),
              shr => Ok(const_uint(a >> b)),
              eq => fromb(a == b),
              lt => fromb(a < b),
              le => fromb(a <= b),
              ne => fromb(a != b),
              ge => fromb(a >= b),
              gt => fromb(a > b),
            }
          }
          // shifts can have any integral type as their rhs
          (Ok(const_int(a)), Ok(const_uint(b))) => {
            match op {
              shl => Ok(const_int(a << b)),
              shr => Ok(const_int(a >> b)),
              _ => Err(~"Can't do this op on an int and uint")
            }
          }
          (Ok(const_uint(a)), Ok(const_int(b))) => {
            match op {
              shl => Ok(const_uint(a << b)),
              shr => Ok(const_uint(a >> b)),
              _ => Err(~"Can't do this op on a uint and int")
            }
          }
          (Ok(const_bool(a)), Ok(const_bool(b))) => {
            Ok(const_bool(match op {
              and => a && b,
              or => a || b,
              bitxor => a ^ b,
              bitand => a & b,
              bitor => a | b,
              eq => a == b,
              ne => a != b,
              _ => return Err(~"Can't do this op on bools")
             }))
          }
          _ => Err(~"Bad operands for binary")
        }
      }
      expr_cast(base, _) => {
        let ety = ty::expr_ty(tcx, e);
        let base = eval_const_expr_partial(tcx, base);
        match /*bad*/copy base {
            Err(_) => base,
            Ok(val) => {
                match ty::get(ety).sty {
                    ty::ty_float(_) => match val {
                        const_uint(u) => Ok(const_float(u as f64)),
                        const_int(i) => Ok(const_float(i as f64)),
                        const_float(_) => base,
                        _ => Err(~"Can't cast float to str"),
                    },
                    ty::ty_uint(_) => match val {
                        const_uint(_) => base,
                        const_int(i) => Ok(const_uint(i as u64)),
                        const_float(f) => Ok(const_uint(f as u64)),
                        _ => Err(~"Can't cast str to uint"),
                    },
                    ty::ty_int(_) | ty::ty_bool => match val {
                        const_uint(u) => Ok(const_int(u as i64)),
                        const_int(_) => base,
                        const_float(f) => Ok(const_int(f as i64)),
                        _ => Err(~"Can't cast str to int"),
                    },
                    _ => Err(~"Can't cast this type")
                }
            }
        }
      }
      expr_path(_) => {
          match lookup_const(tcx, e) {
              Some(actual_e) => eval_const_expr_partial(tcx, actual_e),
              None => Err(~"Non-constant path in constant expr")
          }
      }
      expr_lit(lit) => Ok(lit_to_const(lit)),
      // If we have a vstore, just keep going; it has to be a string
      expr_vstore(e, _) => eval_const_expr_partial(tcx, e),
      expr_paren(e)     => eval_const_expr_partial(tcx, e),
      _ => Err(~"Unsupported constant expr")
    }
}

pub fn lit_to_const(lit: @lit) -> const_val {
    match lit.node {
      lit_str(s) => const_str(/*bad*/copy *s),
      lit_int(n, _) => const_int(n),
      lit_uint(n, _) => const_uint(n),
      lit_int_unsuffixed(n) => const_int(n),
      lit_float(n, _) => const_float(float::from_str(*n).get() as f64),
      lit_float_unsuffixed(n) =>
        const_float(float::from_str(*n).get() as f64),
      lit_nil => const_int(0i64),
      lit_bool(b) => const_bool(b)
    }
}

pub fn compare_const_vals(a: &const_val, b: &const_val) -> int {
  match (a, b) {
    (&const_int(a), &const_int(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (&const_uint(a), &const_uint(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (&const_float(a), &const_float(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (&const_str(ref a), &const_str(ref b)) => {
        if (*a) == (*b) {
            0
        } else if (*a) < (*b) {
            -1
        } else {
            1
        }
    }
    (&const_bool(a), &const_bool(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    _ => fail!(~"compare_const_vals: ill-typed comparison")
  }
}

pub fn compare_lit_exprs(tcx: middle::ty::ctxt, a: @expr, b: @expr) -> int {
  compare_const_vals(&eval_const_expr(tcx, a), &eval_const_expr(tcx, b))
}

pub fn lit_expr_eq(tcx: middle::ty::ctxt, a: @expr, b: @expr) -> bool {
    compare_lit_exprs(tcx, a, b) == 0
}

pub fn lit_eq(a: @lit, b: @lit) -> bool {
    compare_const_vals(&lit_to_const(a), &lit_to_const(b)) == 0
}
