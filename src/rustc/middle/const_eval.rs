import syntax::ast::*;

// FIXME this doesn't handle big integer/float literals correctly (nor does
// the rest of our literal handling - issue #33)
enum const_val {
    const_float(f64),
    const_int(i64),
    const_uint(u64),
    const_str(str),
}

// FIXME: issue #1417
fn eval_const_expr(tcx: middle::ty::ctxt, e: @expr) -> const_val {
    import middle::ty;
    fn fromb(b: bool) -> const_val { const_int(b as i64) }
    alt check e.node {
      expr_unary(neg, inner) {
        alt check eval_const_expr(tcx, inner) {
          const_float(f) { const_float(-f) }
          const_int(i) { const_int(-i) }
          const_uint(i) { const_uint(-i) }
        }
      }
      expr_unary(not, inner) {
        alt check eval_const_expr(tcx, inner) {
          const_int(i) { const_int(!i) }
          const_uint(i) { const_uint(!i) }
        }
      }
      expr_binary(op, a, b) {
        alt check (eval_const_expr(tcx, a), eval_const_expr(tcx, b)) {
          (const_float(a), const_float(b)) {
            alt check op {
              add { const_float(a + b) } subtract { const_float(a - b) }
              mul { const_float(a * b) } div { const_float(a / b) }
              rem { const_float(a % b) } eq { fromb(a == b) }
              lt { fromb(a < b) } le { fromb(a <= b) } ne { fromb(a != b) }
              ge { fromb(a >= b) } gt { fromb(a > b) }
            }
          }
          (const_int(a), const_int(b)) {
            alt check op {
              add { const_int(a + b) } subtract { const_int(a - b) }
              mul { const_int(a * b) } div { const_int(a / b) }
              rem { const_int(a % b) } and | bitand { const_int(a & b) }
              or | bitor { const_int(a | b) } bitxor { const_int(a ^ b) }
              shl { const_int(a << b) } shr { const_int(a >> b) }
              eq { fromb(a == b) } lt { fromb(a < b) }
              le { fromb(a <= b) } ne { fromb(a != b) }
              ge { fromb(a >= b) } gt { fromb(a > b) }
            }
          }
          (const_uint(a), const_uint(b)) {
            alt check op {
              add { const_uint(a + b) } subtract { const_uint(a - b) }
              mul { const_uint(a * b) } div { const_uint(a / b) }
              rem { const_uint(a % b) } and | bitand { const_uint(a & b) }
              or | bitor { const_uint(a | b) } bitxor { const_uint(a ^ b) }
              shl { const_uint(a << b) } shr { const_uint(a >> b) }
              eq { fromb(a == b) } lt { fromb(a < b) }
              le { fromb(a <= b) } ne { fromb(a != b) }
              ge { fromb(a >= b) } gt { fromb(a > b) }
            }
          }
          // shifts can have any integral type as their rhs
          (const_int(a), const_uint(b)) {
            alt check op {
              shl { const_int(a << b) } shr { const_int(a >> b) }
            }
          }
          (const_uint(a), const_int(b)) {
            alt check op {
              shl { const_uint(a << b) } shr { const_uint(a >> b) }
            }
          }
        }
      }
      expr_cast(base, _) {
        let ety = ty::expr_ty(tcx, e);
        let base = eval_const_expr(tcx, base);
        alt check ty::get(ety).struct {
          ty::ty_float(_) {
            alt check base {
              const_uint(u) { const_float(u as f64) }
              const_int(i) { const_float(i as f64) }
              const_float(_) { base }
            }
          }
          ty::ty_uint(_) {
            alt check base {
              const_uint(_) { base }
              const_int(i) { const_uint(i as u64) }
              const_float(f) { const_uint(f as u64) }
            }
          }
          ty::ty_int(_) | ty::ty_bool {
            alt check base {
              const_uint(u) { const_int(u as i64) }
              const_int(_) { base }
              const_float(f) { const_int(f as i64) }
            }
          }
        }
      }
      expr_lit(lit) { lit_to_const(lit) }
    }
}

fn lit_to_const(lit: @lit) -> const_val {
    alt lit.node {
      lit_str(s) { const_str(*s) }
      lit_int(n, _) { const_int(n) }
      lit_uint(n, _) { const_uint(n) }
      lit_int_unsuffixed(n, _) { const_int(n) }
      lit_float(n, _) { const_float(option::get(float::from_str(*n)) as f64) }
      lit_nil { const_int(0i64) }
      lit_bool(b) { const_int(b as i64) }
    }
}

fn compare_const_vals(a: const_val, b: const_val) -> int {
  alt (a, b) {
    (const_int(a), const_int(b)) {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (const_uint(a), const_uint(b)) {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (const_float(a), const_float(b)) {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    (const_str(a), const_str(b)) {
        if a == b {
            0
        } else if a < b {
            -1
        } else {
            1
        }
    }
    _ {
        fail "compare_const_vals: ill-typed comparison";
    }
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
