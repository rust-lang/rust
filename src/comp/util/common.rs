import math::{max, min};
import std::map::hashmap;
import option::{some};
import syntax::ast;
import ast::{ty, pat};
import syntax::codemap::{span};
import syntax::visit;
import syntax::print;

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

fn new_def_hash<V: copy>() -> std::map::hashmap<ast::def_id, V> {
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

fn log_expr(e: ast::expr) {
    log(debug, print::pprust::expr_to_str(@e));
}

fn log_expr_err(e: ast::expr) {
    log(error, print::pprust::expr_to_str(@e));
}

fn log_ty_err(t: @ty) {
    log(error, print::pprust::ty_to_str(t));
}

fn log_pat_err(p: @pat) {
    log(error, print::pprust::pat_to_str(p));
}

fn log_block(b: ast::blk) {
    log(debug, print::pprust::block_to_str(b));
}

fn log_block_err(b: ast::blk) {
    log(error, print::pprust::block_to_str(b));
}

fn log_item_err(i: @ast::item) {
    log(error, print::pprust::item_to_str(i));
}
fn log_stmt(st: ast::stmt) {
    log(debug, print::pprust::stmt_to_str(st));
}

fn log_stmt_err(st: ast::stmt) {
    log(error, print::pprust::stmt_to_str(st));
}

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

fn is_main_name(path: [ast::ident]) -> bool {
    str::eq(option::get(vec::last(path)), "main")
}



//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
