
import std::map;
import std::map::hashmap;
import std::uint;
import std::int;
import std::vec;
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
import syntax::walk;
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

fn def_eq(&ast::def_id a, &ast::def_id b) -> bool {
    ret a._0 == b._0 && a._1 == b._1;
}

fn hash_def(&ast::def_id d) -> uint {
    auto h = 5381u;
    h = (h << 5u) + h ^ (d._0 as uint);
    h = (h << 5u) + h ^ (d._1 as uint);
    ret h;
}

fn new_def_hash[V]() -> std::map::hashmap[ast::def_id, V] {
    let std::map::hashfn[ast::def_id] hasher = hash_def;
    let std::map::eqfn[ast::def_id] eqer = def_eq;
    ret std::map::mk_hashmap[ast::def_id, V](hasher, eqer);
}

fn elt_expr(&ast::elt e) -> @ast::expr { ret e.expr; }

fn elt_exprs(&ast::elt[] elts) -> (@ast::expr)[] {
    auto es = ~[];
    for (ast::elt e in elts) { es += ~[e.expr]; }
    ret es;
}

fn field_expr(&ast::field f) -> @ast::expr { ret f.node.expr; }

fn field_exprs(&ast::field[] fields) -> (@ast::expr)[] {
    auto es = ~[];
    for (ast::field f in fields) { es += ~[f.node.expr]; }
    ret es;
}

fn log_expr(&ast::expr e) { log print::pprust::expr_to_str(@e); }

fn log_expr_err(&ast::expr e) { log_err print::pprust::expr_to_str(@e); }

fn log_ty_err(&ty t) { log_err print::pprust::ty_to_str(t); }

fn log_pat_err(&@pat p) { log_err print::pprust::pat_to_str(p); }

fn log_block(&ast::block b) { log print::pprust::block_to_str(b); }

fn log_block_err(&ast::block b) { log_err print::pprust::block_to_str(b); }

fn log_item_err(&@ast::item i) { log_err print::pprust::item_to_str(i); }

fn log_fn(&ast::_fn f, str name, &ast::ty_param[] params) {
    log print::pprust::fun_to_str(f, name, params);
}

fn log_fn_err(&ast::_fn f, str name, &ast::ty_param[] params) {
    log_err print::pprust::fun_to_str(f, name, params);
}

fn log_stmt(&ast::stmt st) { log print::pprust::stmt_to_str(st); }

fn log_stmt_err(&ast::stmt st) { log_err print::pprust::stmt_to_str(st); }

fn has_nonlocal_exits(&ast::block b) -> bool {
    auto has_exits = @mutable false;
    fn visit_expr(@mutable bool flag, &@ast::expr e) {
        alt (e.node) {
            case (ast::expr_break) { *flag = true; }
            case (ast::expr_cont) { *flag = true; }
            case (_) { }
        }
    }
    auto v =
        rec(visit_expr_pre=bind visit_expr(has_exits, _)
            with walk::default_visitor());
    walk::walk_block(v, b);
    ret *has_exits;
}

fn local_rhs_span(&@ast::local l, &span def) -> span {
    alt (l.node.init) {
        case (some(?i)) { ret i.expr.span; }
        case (_) { ret def; }
    }
}

fn lit_eq(&@ast::lit l, &@ast::lit m) -> bool {
    alt (l.node) {
        case (ast::lit_str(?s, ?kind_s)) {
            alt (m.node) {
                case (ast::lit_str(?t, ?kind_t)) {
                    ret s == t && kind_s == kind_t;
                }
                case (_) { ret false; }
            }
        }
        case (ast::lit_char(?c)) {
            alt (m.node) {
                case (ast::lit_char(?d)) { ret c == d; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_int(?i)) {
            alt (m.node) {
                case (ast::lit_int(?j)) { ret i == j; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_uint(?i)) {
            alt (m.node) {
                case (ast::lit_uint(?j)) { ret i == j; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_mach_int(_, ?i)) {
            alt (m.node) {
                case (ast::lit_mach_int(_, ?j)) { ret i == j; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_float(?s)) {
            alt (m.node) {
                case (ast::lit_float(?t)) { ret s == t; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_mach_float(_, ?s)) {
            alt (m.node) {
                case (ast::lit_mach_float(_, ?t)) { ret s == t; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_nil) {
            alt (m.node) {
                case (ast::lit_nil) { ret true; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_bool(?b)) {
            alt (m.node) {
                case (ast::lit_bool(?c)) { ret b == c; }
                case (_) { ret false; }
            }
        }
    }
}

// FIXME move to vec
fn any[T](&fn(&T) -> bool f, &vec[T] v) -> bool {
    for (T t in v) {
        if (f(t)) { ret true; } 
    }
    ret false;
}

tag call_kind {
    kind_call;
    kind_spawn;
    kind_bind;
}

fn call_kind_str(call_kind c) -> str {
    alt (c) {
        case (kind_call)  { "Call" }
        case (kind_spawn) { "Spawn" }
        case (kind_bind)  { "Bind" }
    }
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
