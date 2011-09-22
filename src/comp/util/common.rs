import std::{str, map, uint, int, option};
import std::map::hashmap;
import std::option::{none, some};
import syntax::ast;
import ast::{ty, pat, lit, path};
import syntax::codemap::{codemap, span};
import syntax::visit;
import std::io::{stdout, str_writer, string_writer};
import syntax::print;
import print::pprust::{print_block, print_item, print_expr, print_path,
                       print_decl, print_fn, print_type, print_literal};
import print::pp::mk_printer;

type flag = hashmap<str, ()>;

fn def_eq(a: ast::def_id, b: ast::def_id) -> bool {
    ret a.crate == b.crate && a.node == b.node;
}

fn hash_def(d: ast::def_id) -> uint {
    let h = 5381u;
    h = (h << 5u) + h ^ (d.crate as uint);
    h = (h << 5u) + h ^ (d.node as uint);
    ret h;
}

fn new_def_hash<@V>() -> std::map::hashmap<ast::def_id, V> {
    let hasher: std::map::hashfn<ast::def_id> = hash_def;
    let eqer: std::map::eqfn<ast::def_id> = def_eq;
    ret std::map::mk_hashmap::<ast::def_id, V>(hasher, eqer);
}

fn field_expr(f: ast::field) -> @ast::expr { ret f.node.expr; }

fn field_exprs(fields: [ast::field]) -> [@ast::expr] {
    let es = [];
    for f: ast::field in fields { es += [f.node.expr]; }
    ret es;
}

fn log_expr(e: ast::expr) { log print::pprust::expr_to_str(@e); }

fn log_expr_err(e: ast::expr) { log_err print::pprust::expr_to_str(@e); }

fn log_ty_err(t: @ty) { log_err print::pprust::ty_to_str(t); }

fn log_pat_err(p: @pat) { log_err print::pprust::pat_to_str(p); }

fn log_block(b: ast::blk) { log print::pprust::block_to_str(b); }

fn log_block_err(b: ast::blk) { log_err print::pprust::block_to_str(b); }

fn log_item_err(i: @ast::item) { log_err print::pprust::item_to_str(i); }

fn log_fn(f: ast::_fn, name: ast::ident, params: [ast::ty_param]) {
    log print::pprust::fun_to_str(f, name, params);
}

fn log_fn_err(f: ast::_fn, name: ast::ident, params: [ast::ty_param]) {
    log_err print::pprust::fun_to_str(f, name, params);
}

fn log_stmt(st: ast::stmt) { log print::pprust::stmt_to_str(st); }

fn log_stmt_err(st: ast::stmt) { log_err print::pprust::stmt_to_str(st); }

fn has_nonlocal_exits(b: ast::blk) -> bool {
    let has_exits = @mutable false;
    fn visit_expr(flag: @mutable bool, e: @ast::expr) {
        alt e.node {
          ast::expr_break. { *flag = true; }
          ast::expr_cont. { *flag = true; }
          _ { }
        }
    }
    let v =
        visit::mk_simple_visitor(@{visit_expr: bind visit_expr(has_exits, _)
                                      with *visit::default_simple_visitor()});
    visit::visit_block(b, (), v);
    ret *has_exits;
}

fn local_rhs_span(l: @ast::local, def: span) -> span {
    alt l.node.init { some(i) { ret i.expr.span; } _ { ret def; } }
}

fn lit_is_numeric(l: @ast::lit) -> bool {
    alt l.node {
      ast::lit_int(_) | ast::lit_char(_) | ast::lit_uint(_) |
      ast::lit_mach_int(_, _) | ast::lit_float(_) | ast::lit_mach_float(_,_) {
        true
      }
      _ { false }
    }
}

fn lit_type_eq(l: @ast::lit, m: @ast::lit) -> bool {
    alt l.node {
      ast::lit_str(_) {
        alt m.node { ast::lit_str(_) { true } _ { false } }
      }
      ast::lit_char(_) {
        alt m.node { ast::lit_char(_) { true } _ { false } }
      }
      ast::lit_int(_) {
        alt m.node { ast::lit_int(_) { true } _ { false } }
      }
      ast::lit_uint(_) {
        alt m.node { ast::lit_uint(_) { true } _ { false } }
      }
      ast::lit_mach_int(_, _) {
        alt m.node { ast::lit_mach_int(_, _) { true } _ { false } }
      }
      ast::lit_float(_) {
        alt m.node { ast::lit_float(_) { true } _ { false } }
      }
      ast::lit_mach_float(_, _) {
        alt m.node { ast::lit_mach_float(_, _) { true } _ { false } }
      }
      ast::lit_nil. {
        alt m.node { ast::lit_nil. { true } _ { false } }
      }
      ast::lit_bool(_) {
        alt m.node { ast::lit_bool(_) { true } _ { false } }
      }
    }
}

fn lit_in_range(l: @ast::lit, m1: @ast::lit, m2: @ast::lit) -> bool {
    alt lits_to_range(m1, m2) {
      irange(i1, i2) {
        alt l.node {
          ast::lit_int(i3) | ast::lit_mach_int(_, i3) {
            i3 >= *min(i1, i2) && i3 <= *max(i1, i2)
          }
          _ { fail }
        }
      }
      urange(u1, u2) {
        alt l.node {
          ast::lit_uint(u3) {
            u3 >= *min(u1, u2) && u3 <= *max(u1, u2)
          }
          _ { fail }
        }
      }
      crange(c1, c2) {
        alt l.node {
          ast::lit_char(c3) {
            (c3 as uint) >= *min(c1 as uint, c2 as uint) &&
            (c3 as uint) <= *max(c1 as uint, c2 as uint)
          }
          _ { fail }
        }
      }
      frange(f1, f2) {
        alt l.node {
          ast::lit_float(f3) | ast::lit_mach_float(_, f3) {
            str_to_float(f3) >= *min(f1, f2) &&
            str_to_float(f3) <= *max(f1, f2)
          }
          _ { fail }
        }
      }
    }
}

fn min<T>(x: T, y: T) -> @T {
    ret @(if x > y { y } else { x });
}

fn max<T>(x: T, y: T) -> @T {
    ret @(if x > y { x } else { y });
}

fn ranges_overlap<T>(a1: T, a2: T, b1: T, b2: T) -> bool {
    let min1 = *min(a1, a2);
    let max1 = *max(a1, a2);
    let min2 = *min(b1, b2);
    let max2 = *max(b1, b2);
    ret (min1 >= min2 && max1 <= max2) || (min1 <= min2 && max1 >= min2) ||
        (min1 >= min2 && min1 <= max2) || (max1 >= min2 && max1 <= max2);
}

fn lit_ranges_overlap(a1: @ast::lit, a2: @ast::lit,
                      b1: @ast::lit, b2: @ast::lit) -> bool {
    alt lits_to_range(a1, a2) {
      irange(i1, i2) {
        alt lits_to_range(b1, b2) {
          irange(i3, i4) { ranges_overlap(i1, i2, i3, i4) }
          _ { fail }
        }
      }
      urange(u1, u2) {
        alt lits_to_range(b1, b2) {
          urange(u3, u4) { ranges_overlap(u1, u2, u3, u4) }
          _ { fail }
        }
      }
      crange(c1, c2) {
        alt lits_to_range(b1, b2) {
          crange(c3, c4) { ranges_overlap(c1, c2, c3, c4) }
          _ { fail }
        }
      }
      frange(f1, f2) {
        alt lits_to_range(b1, b2) {
          frange(f3, f4) { ranges_overlap(f1, f2, f3, f4) }
          _ { fail }
        }
      }
    }
}

tag range {
    irange(int, int);
    urange(uint, uint);
    crange(char, char);
    frange(float, float);
}

fn lits_to_range(l: @ast::lit, r: @ast::lit) -> range {
    alt l.node {
      ast::lit_int(i1) | ast::lit_mach_int(_, i1) {
        alt r.node { ast::lit_int(i2) { irange(i1, i2) } _ { fail } }
      }
      ast::lit_uint(u1) {
        alt r.node { ast::lit_uint(u2) { urange(u1, u2) } _ { fail } }
      }
      ast::lit_char(c1) {
        alt r.node { ast::lit_char(c2) { crange(c1, c2) } _ { fail } }
      }
      ast::lit_float(f1) | ast::lit_mach_float(_, f1) {
        alt r.node { ast::lit_float(f2) | ast::lit_mach_float(_, f2) {
          frange(str_to_float(f1), str_to_float(f2))
        }
        _ { fail } }
      }
      _ { fail }
    }
}

fn lit_eq(l: @ast::lit, m: @ast::lit) -> bool {
    alt l.node {
      ast::lit_str(s) {
        alt m.node { ast::lit_str(t) { ret s == t } _ { ret false; } }
      }
      ast::lit_char(c) {
        alt m.node { ast::lit_char(d) { ret c == d; } _ { ret false; } }
      }
      ast::lit_int(i) {
        alt m.node { ast::lit_int(j) { ret i == j; } _ { ret false; } }
      }
      ast::lit_uint(i) {
        alt m.node { ast::lit_uint(j) { ret i == j; } _ { ret false; } }
      }
      ast::lit_mach_int(_, i) {
        alt m.node {
          ast::lit_mach_int(_, j) { ret i == j; }
          _ { ret false; }
        }
      }
      ast::lit_float(s) {
        alt m.node { ast::lit_float(t) { ret s == t; } _ { ret false; } }
      }
      ast::lit_mach_float(_, s) {
        alt m.node {
          ast::lit_mach_float(_, t) { ret s == t; }
          _ { ret false; }
        }
      }
      ast::lit_nil. {
        alt m.node { ast::lit_nil. { ret true; } _ { ret false; } }
      }
      ast::lit_bool(b) {
        alt m.node { ast::lit_bool(c) { ret b == c; } _ { ret false; } }
      }
    }
}

tag call_kind { kind_call; kind_spawn; kind_bind; kind_for_each; }

fn call_kind_str(c: call_kind) -> str {
    alt c {
      kind_call. { "Call" }
      kind_spawn. { "Spawn" }
      kind_bind. { "Bind" }
      kind_for_each. { "For-Each" }
    }
}

fn is_main_name(path: [ast::ident]) -> bool {
    str::eq(option::get(std::vec::last(path)), "main")
}

// FIXME mode this to std::float when editing the stdlib no longer
// requires a snapshot
fn float_to_str(num: float, digits: uint) -> str {
    let accum = if num < 0.0 { num = -num; "-" } else { "" };
    let trunc = num as uint;
    let frac = num - (trunc as float);
    accum += uint::str(trunc);
    if frac == 0.0 || digits == 0u { ret accum; }
    accum += ".";
    while digits > 0u && frac > 0.0 {
        frac *= 10.0;
        let digit = frac as uint;
        accum += uint::str(digit);
        frac -= digit as float;
        digits -= 1u;
    }
    ret accum;
}

fn str_to_float(num: str) -> float {
    let digits = str::split(num, '.' as u8);
    let total = int::from_str(digits[0]) as float;

    fn dec_val(c: char) -> int { ret (c as int) - ('0' as int); }

    let right = digits[1];
    let len = str::char_len(digits[1]);
    let i = 1u;
    while (i < len) {
        total += dec_val(str::pop_char(right)) as float /
                 (int::pow(10, i) as float);
        i += 1u;
    }
    ret total;
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
