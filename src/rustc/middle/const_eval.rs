import syntax::{ast,ast_util,visit};
import ast::*;

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
//        fixed-size vectors and strings: []/_ and ""/_
//        vector and string slices: &[] and &""
//        tuples: (,)
//        records: {...}
//        enums: foo(...)
//        floating point literals and operators
//        & and * pointers
//        copies of general constants
//
//        (in theory, probably not at first: if/alt on integer-const
//         conditions / descriminants)
//
//   - Non-constants: everything else.
//

enum constness {
    integral_const,
    general_const,
    non_const
}

fn join(a: constness, b: constness) -> constness {
    match (a,b) {
      (integral_const, integral_const) => integral_const,
      (integral_const, general_const)
      | (general_const, integral_const)
      | (general_const, general_const) => general_const,
      _ => non_const
    }
}

fn join_all(cs: &[constness]) -> constness {
    vec::foldl(integral_const, cs, join)
}

fn classify(e: @expr,
            def_map: resolve3::DefMap,
            tcx: ty::ctxt) -> constness {
    let did = ast_util::local_def(e.id);
    match tcx.ccache.find(did) {
      some(x) => x,
      none => {
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
              ast::expr_unary(_, inner) => {
                classify(inner, def_map, tcx)
              }

              ast::expr_binary(_, a, b) => {
                join(classify(a, def_map, tcx),
                     classify(b, def_map, tcx))
              }

              ast::expr_tup(es) |
              ast::expr_vec(es, ast::m_imm) => {
                join_all(vec::map(es, |e| classify(e, def_map, tcx)))
              }

              ast::expr_vstore(e, vstore) => {
                match vstore {
                  ast::vstore_fixed(_) |
                  ast::vstore_slice(_) => classify(e, def_map, tcx),
                  ast::vstore_uniq |
                  ast::vstore_box => non_const
                }
              }

              ast::expr_struct(_, fs, none) |
              ast::expr_rec(fs, none) => {
                let cs = do vec::map(fs) |f| {
                    if f.node.mutbl == ast::m_imm {
                        classify(f.node.expr, def_map, tcx)
                    } else {
                        non_const
                    }
                };
                join_all(cs)
              }

              ast::expr_cast(base, _) => {
                let ty = ty::expr_ty(tcx, e);
                let base = classify(base, def_map, tcx);
                if ty::type_is_integral(ty) {
                    join(integral_const, base)
                } else if ty::type_is_fp(ty) {
                    join(general_const, base)
                } else {
                    non_const
                }
              }

              ast::expr_field(base, _, _) => {
                classify(base, def_map, tcx)
              }

              ast::expr_index(base, idx) => {
                join(classify(base, def_map, tcx),
                     classify(idx, def_map, tcx))
              }

              ast::expr_addr_of(ast::m_imm, base) => {
                classify(base, def_map, tcx)
              }

              // FIXME: #1272, we can probably do something CCI-ish
              // surrounding nonlocal constants. But we don't yet.
              ast::expr_path(_) => {
                match def_map.find(e.id) {
                  some(ast::def_const(def_id)) => {
                    if ast_util::is_local(def_id) {
                        let ty = ty::expr_ty(tcx, e);
                        if ty::type_is_integral(ty) {
                            integral_const
                        } else {
                            general_const
                        }
                    } else {
                        non_const
                    }
                  }
                  some(_) => {
                    non_const
                  }
                  none => {
                    tcx.sess.span_bug(e.span,
                                      ~"unknown path when \
                                        classifying constants");
                  }
                }
              }

              _ => non_const
            };
        tcx.ccache.insert(did, cn);
        cn
      }
    }
}

fn process_crate(crate: @ast::crate,
                 def_map: resolve3::DefMap,
                 tcx: ty::ctxt) {
    let v = visit::mk_simple_visitor(@{
        visit_expr_post: |e| { classify(e, def_map, tcx); }
        with *visit::default_simple_visitor()
    });
    visit::visit_crate(*crate, (), v);
    tcx.sess.abort_if_errors();
}


// FIXME (#33): this doesn't handle big integer/float literals correctly
// (nor does the rest of our literal handling).
enum const_val {
    const_float(f64),
    const_int(i64),
    const_uint(u64),
    const_str(~str),
    const_bool(bool)
}

// FIXME: issue #1417
fn eval_const_expr(tcx: middle::ty::ctxt, e: @expr) -> const_val {
    import middle::ty;
    fn fromb(b: bool) -> const_val { const_int(b as i64) }
    match e.node {
      expr_unary(neg, inner) => {
        match eval_const_expr(tcx, inner) {
          const_float(f) => const_float(-f),
          const_int(i) => const_int(-i),
          const_uint(i) => const_uint(-i),
          const_str(_) => fail ~"Negate on string",
          const_bool(_) => fail ~"Negate on boolean"
        }
      }
      expr_unary(not, inner) => {
        match eval_const_expr(tcx, inner) {
          const_int(i) => const_int(!i),
          const_uint(i) => const_uint(!i),
          const_bool(b) => const_bool(!b),
          _ => fail ~"Not on float or string"
        }
      }
      expr_binary(op, a, b) => {
        match (eval_const_expr(tcx, a), eval_const_expr(tcx, b)) {
          (const_float(a), const_float(b)) => {
            match op {
              add => const_float(a + b),
              subtract => const_float(a - b),
              mul => const_float(a * b),
              div => const_float(a / b),
              rem => const_float(a % b),
              eq => fromb(a == b),
              lt => fromb(a < b),
              le => fromb(a <= b),
              ne => fromb(a != b),
              ge => fromb(a >= b),
              gt => fromb(a > b),
              _ => fail ~"Can't do this op on floats"
            }
          }
          (const_int(a), const_int(b)) => {
            match op {
              add => const_int(a + b),
              subtract => const_int(a - b),
              mul => const_int(a * b),
              div => const_int(a / b),
              rem => const_int(a % b),
              and | bitand => const_int(a & b),
              or | bitor => const_int(a | b),
              bitxor => const_int(a ^ b),
              shl => const_int(a << b),
              shr => const_int(a >> b),
              eq => fromb(a == b),
              lt => fromb(a < b),
              le => fromb(a <= b),
              ne => fromb(a != b),
              ge => fromb(a >= b),
              gt => fromb(a > b)
            }
          }
          (const_uint(a), const_uint(b)) => {
            match op {
              add => const_uint(a + b),
              subtract => const_uint(a - b),
              mul => const_uint(a * b),
              div => const_uint(a / b),
              rem => const_uint(a % b),
              and | bitand => const_uint(a & b),
              or | bitor => const_uint(a | b),
              bitxor => const_uint(a ^ b),
              shl => const_uint(a << b),
              shr => const_uint(a >> b),
              eq => fromb(a == b),
              lt => fromb(a < b),
              le => fromb(a <= b),
              ne => fromb(a != b),
              ge => fromb(a >= b),
              gt => fromb(a > b),
            }
          }
          // shifts can have any integral type as their rhs
          (const_int(a), const_uint(b)) => {
            match op {
              shl => const_int(a << b),
              shr => const_int(a >> b),
              _ => fail ~"Can't do this op on an int and uint"
            }
          }
          (const_uint(a), const_int(b)) => {
            match op {
              shl => const_uint(a << b),
              shr => const_uint(a >> b),
              _ => fail ~"Can't do this op on a uint and int"
            }
          }
          (const_bool(a), const_bool(b)) => {
            const_bool(match op {
              and => a && b,
              or => a || b,
              bitxor => a ^ b,
              bitand => a & b,
              bitor => a | b,
              eq => a == b,
              ne => a != b,
              _ => fail ~"Can't do this op on bools"
             })
          }
          _ => fail ~"Bad operands for binary"
        }
      }
      expr_cast(base, _) => {
        let ety = ty::expr_ty(tcx, e);
        let base = eval_const_expr(tcx, base);
        match ty::get(ety).struct {
          ty::ty_float(_) => {
            match base {
              const_uint(u) => const_float(u as f64),
              const_int(i) => const_float(i as f64),
              const_float(_) => base,
              _ => fail ~"Can't cast float to str"
            }
          }
          ty::ty_uint(_) => {
            match base {
              const_uint(_) => base,
              const_int(i) => const_uint(i as u64),
              const_float(f) => const_uint(f as u64),
              _ => fail ~"Can't cast str to uint"
            }
          }
          ty::ty_int(_) | ty::ty_bool => {
            match base {
              const_uint(u) => const_int(u as i64),
              const_int(_) => base,
              const_float(f) => const_int(f as i64),
              _ => fail ~"Can't cast str to int"
            }
          }
          _ => fail ~"Can't cast this type"
        }
      }
      expr_lit(lit) => lit_to_const(lit),
      // If we have a vstore, just keep going; it has to be a string
      expr_vstore(e, _) => eval_const_expr(tcx, e),
      _ => fail ~"Unsupported constant expr"
    }
}

fn lit_to_const(lit: @lit) -> const_val {
    match lit.node {
      lit_str(s) => const_str(*s),
      lit_int(n, _) => const_int(n),
      lit_uint(n, _) => const_uint(n),
      lit_int_unsuffixed(n) => const_int(n),
      lit_float(n, _) => const_float(option::get(float::from_str(*n)) as f64),
      lit_nil => const_int(0i64),
      lit_bool(b) => const_bool(b)
    }
}

fn compare_const_vals(a: const_val, b: const_val) -> int {
  match (a, b) {
    (const_int(a), const_int(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (const_uint(a), const_uint(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (const_float(a), const_float(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (const_str(a), const_str(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (const_bool(a), const_bool(b)) => {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    _ => fail ~"compare_const_vals: ill-typed comparison"
  }
}

fn compare_lit_exprs(tcx: middle::ty::ctxt, a: @expr, b: @expr) -> int {
  compare_const_vals(eval_const_expr(tcx, a), eval_const_expr(tcx, b))
}

fn lit_expr_eq(tcx: middle::ty::ctxt, a: @expr, b: @expr) -> bool {
    compare_lit_exprs(tcx, a, b) == 0
}

fn lit_eq(a: @lit, b: @lit) -> bool {
    compare_const_vals(lit_to_const(a), lit_to_const(b)) == 0
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
