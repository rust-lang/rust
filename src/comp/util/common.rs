import std::str;
import std::map;
import std::map::hashmap;
import std::uint;
import std::int;
import std::option;
import std::option::none;
import std::option::some;
import syntax::ast;
import ast::ty;
import ast::pat;
import syntax::codemap::codemap;
import syntax::codemap::span;
import ast::lit;
import ast::path;
import syntax::visit;
import std::io::stdout;
import std::io::str_writer;
import std::io::string_writer;
import syntax::print;
import print::pprust::print_block;
import print::pprust::print_item;
import print::pprust::print_expr;
import print::pprust::print_path;
import print::pprust::print_decl;
import print::pprust::print_fn;
import print::pprust::print_type;
import print::pprust::print_literal;
import print::pp::mk_printer;

type flag = hashmap[str, ()];

fn def_eq(a: &ast::def_id, b: &ast::def_id) -> bool {
    ret a.crate == b.crate && a.node == b.node;
}

fn hash_def(d: &ast::def_id) -> uint {
    let h = 5381u;
    h = (h << 5u) + h ^ (d.crate as uint);
    h = (h << 5u) + h ^ (d.node as uint);
    ret h;
}

fn new_def_hash[@V]() -> std::map::hashmap[ast::def_id, V] {
    let hasher: std::map::hashfn[ast::def_id] = hash_def;
    let eqer: std::map::eqfn[ast::def_id] = def_eq;
    ret std::map::mk_hashmap[ast::def_id, V](hasher, eqer);
}

fn field_expr(f: &ast::field) -> @ast::expr { ret f.node.expr; }

fn field_exprs(fields: &[ast::field]) -> [@ast::expr] {
    let es = ~[];
    for f: ast::field in fields { es += ~[f.node.expr]; }
    ret es;
}

fn log_expr(e: &ast::expr) { log print::pprust::expr_to_str(@e); }

fn log_expr_err(e: &ast::expr) { log_err print::pprust::expr_to_str(@e); }

fn log_ty_err(t: &@ty) { log_err print::pprust::ty_to_str(t); }

fn log_pat_err(p: &@pat) { log_err print::pprust::pat_to_str(p); }

fn log_block(b: &ast::blk) { log print::pprust::block_to_str(b); }

fn log_block_err(b: &ast::blk) { log_err print::pprust::block_to_str(b); }

fn log_item_err(i: &@ast::item) { log_err print::pprust::item_to_str(i); }

fn log_fn(f: &ast::_fn, name: str, params: &[ast::ty_param]) {
    log print::pprust::fun_to_str(f, name, params);
}

fn log_fn_err(f: &ast::_fn, name: str, params: &[ast::ty_param]) {
    log_err print::pprust::fun_to_str(f, name, params);
}

fn log_stmt(st: &ast::stmt) { log print::pprust::stmt_to_str(st); }

fn log_stmt_err(st: &ast::stmt) { log_err print::pprust::stmt_to_str(st); }

fn has_nonlocal_exits(b: &ast::blk) -> bool {
    let has_exits = @mutable false;
    fn visit_expr(flag: @mutable bool, e: &@ast::expr) {
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

fn local_rhs_span(l: &@ast::local, def: &span) -> span {
    alt l.node.init { some(i) { ret i.expr.span; } _ { ret def; } }
}

fn lit_eq(l: &@ast::lit, m: &@ast::lit) -> bool {
    alt l.node {
      ast::lit_str(s, kind_s) {
        alt m.node {
          ast::lit_str(t, kind_t) { ret s == t && kind_s == kind_t; }
          _ { ret false; }
        }
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

fn is_main_name(path: &[str]) -> bool {
    str::eq(option::get(std::ivec::last(path)), "main")
}

// FIXME mode this to std::float when editing the stdlib no longer
// requires a snapshot
fn float_to_str(num: float, digits: uint) -> str {
    let accum = if num < 0.0 { num = -num; "-" }
                else { "" };
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
