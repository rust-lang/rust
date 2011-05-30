import std::map;
import std::map::hashmap;
import std::uint;
import std::int;
import std::vec;
import std::option::none;
import std::option::some;
import front::ast;
import front::ast::ty;
import front::ast::pat;
import front::codemap::codemap;
import middle::walk;

import std::io::stdout;
import std::io::str_writer;
import std::io::string_writer;
import pretty::pprust::print_block;
import pretty::pprust::print_item;
import pretty::pprust::print_expr;
import pretty::pprust::print_decl;
import pretty::pprust::print_fn;
import pretty::pprust::print_type;
import pretty::pprust::mo_untyped;
import pretty::pp::mk_printer;

type filename = str;
type span = rec(uint lo, uint hi);
type spanned[T] = rec(T node, span span);
type flag = hashmap[str, ()];

tag ty_mach {
    ty_i8;
    ty_i16;
    ty_i32;
    ty_i64;

    ty_u8;
    ty_u16;
    ty_u32;
    ty_u64;

    ty_f32;
    ty_f64;
}

tag ty_or_bang[T] {
    a_ty(T);
    a_bang;
}

fn ty_mach_to_str(ty_mach tm) -> str {
    alt (tm) {
        case (ty_u8) { ret "u8"; }
        case (ty_u16) { ret "u16"; }
        case (ty_u32) { ret "u32"; }
        case (ty_u64) { ret "u64"; }

        case (ty_i8) { ret "i8"; }
        case (ty_i16) { ret "i16"; }
        case (ty_i32) { ret "i32"; }
        case (ty_i64) { ret "i64"; }

        case (ty_f32) { ret "f32"; }
        case (ty_f64) { ret "f64"; }
    }
}

fn new_str_hash[V]() -> std::map::hashmap[str,V] {
    let std::map::hashfn[str] hasher = std::str::hash;
    let std::map::eqfn[str] eqer = std::str::eq;
    ret std::map::mk_hashmap[str,V](hasher, eqer);
}

fn def_eq(&ast::def_id a, &ast::def_id b) -> bool {
    ret a._0 == b._0 && a._1 == b._1;
}

fn hash_def(&ast::def_id d) -> uint {
    auto h = 5381u;
    h = ((h << 5u) + h) ^ (d._0 as uint);
    h = ((h << 5u) + h) ^ (d._1 as uint);
    ret h;
}

fn new_def_hash[V]() -> std::map::hashmap[ast::def_id,V] {
    let std::map::hashfn[ast::def_id] hasher = hash_def;
    let std::map::eqfn[ast::def_id] eqer = def_eq;
    ret std::map::mk_hashmap[ast::def_id,V](hasher, eqer);
}

fn new_int_hash[V]() -> std::map::hashmap[int,V] {
    fn hash_int(&int x) -> uint { ret x as uint; }
    fn eq_int(&int a, &int b) -> bool { ret a == b; }
    auto hasher = hash_int;
    auto eqer = eq_int;
    ret std::map::mk_hashmap[int,V](hasher, eqer);
}

fn new_uint_hash[V]() -> std::map::hashmap[uint,V] {
    fn hash_uint(&uint x) -> uint { ret x; }
    fn eq_uint(&uint a, &uint b) -> bool { ret a == b; }
    auto hasher = hash_uint;
    auto eqer = eq_uint;
    ret std::map::mk_hashmap[uint,V](hasher, eqer);
}

fn istr(int i) -> str {
    ret int::to_str(i, 10u);
}

fn uistr(uint i) -> str {
    ret uint::to_str(i, 10u);
}

fn elt_expr(&ast::elt e) -> @ast::expr { ret e.expr; }

fn elt_exprs(&vec[ast::elt] elts) -> vec[@ast::expr] {
    auto f = elt_expr;
    ret vec::map[ast::elt, @ast::expr](f, elts);
}

fn field_expr(&ast::field f) -> @ast::expr { ret f.node.expr; }

fn field_exprs(vec[ast::field] fields) -> vec [@ast::expr] {
    auto f = field_expr;
    ret vec::map[ast::field, @ast::expr](f, fields);
}

fn expr_to_str(&@ast::expr e) -> str {
  let str_writer s = string_writer();
  auto out_ = mk_printer(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  cm=none[codemap],
                  comments=none[vec[front::lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);
  print_expr(out, e);
  ret s.get_str();
}

fn ty_to_str(&ty t) -> str {
  let str_writer s = string_writer();
  auto out_ = mk_printer(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  cm=none[codemap],
                  comments=none[vec[front::lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);
  print_type(out, @t);
  ret s.get_str();
}

fn log_expr(&ast::expr e) -> () {
    log(expr_to_str(@e));
}

fn log_expr_err(&ast::expr e) -> () {
    log_err(expr_to_str(@e));
}

fn log_ty_err(&ty t) -> () {
    log_err(ty_to_str(t));
}

fn log_pat_err(&@pat p) -> () {
    log_err(pretty::pprust::pat_to_str(p));
}

fn block_to_str(&ast::block b) -> str {
  let str_writer s = string_writer();
  auto out_ = mk_printer(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  cm=none[codemap],
                  comments=none[vec[front::lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);

  print_block(out, b);
  ret s.get_str();
}

fn item_to_str(&@ast::item i) -> str {
  let str_writer s = string_writer();
  auto out_ = mk_printer(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  cm=none[codemap],
                  comments=none[vec[front::lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);
  print_item(out, i);
  ret s.get_str();
}

fn log_block(&ast::block b) -> () {
    log(block_to_str(b));
}

fn log_block_err(&ast::block b) -> () {
    log_err(block_to_str(b));
}

fn log_item_err(&@ast::item i) -> () {
    log_err(item_to_str(i));
}

fn fun_to_str(&ast::_fn f, str name, vec[ast::ty_param] params) -> str {
 let str_writer s = string_writer();
  auto out_ = mk_printer(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  cm=none[codemap],
                  comments=none[vec[front::lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);

  print_fn(out, f.decl, name, params);
  ret s.get_str();
}

fn log_fn(&ast::_fn f, str name, vec[ast::ty_param] params) -> () {
    log(fun_to_str(f, name, params));
}

fn log_fn_err(&ast::_fn f, str name, vec[ast::ty_param] params) -> () {
    log_err(fun_to_str(f, name, params));
}

fn stmt_to_str(&ast::stmt st) -> str {
  let str_writer s = string_writer();
  auto out_ = mk_printer(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  cm=none[codemap],
                  comments=none[vec[front::lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);
  alt (st.node) {
    case (ast::stmt_decl(?decl,_)) {
      print_decl(out, decl);
    }
    case (ast::stmt_expr(?ex,_)) {
      print_expr(out, ex);
    }
    case (_) { /* do nothing */ }
  }
  ret s.get_str();
}

fn log_stmt(&ast::stmt st) -> () {
    log(stmt_to_str(st));
}

fn log_stmt_err(&ast::stmt st) -> () {
    log_err(stmt_to_str(st));
}

fn decl_lhs(@ast::decl d) -> ast::def_id {
    alt (d.node) {
        case (ast::decl_local(?l)) {
            ret l.id;
        }
        case (ast::decl_item(?an_item)) {
            alt (an_item.node) {
                case (ast::item_const(_,_,_,?d,_)) {
                    ret d;
                }
                case (ast::item_fn(_,_,_,?d,_)) {
                    ret d;
                }
                case (ast::item_mod(_,_,?d)) {
                    ret d;
                }
                case (ast::item_native_mod(_,_,?d)) {
                    ret d;
                }
                case (ast::item_ty(_,_,_,?d,_)) {
                    ret d;
                }
                case (ast::item_tag(_,_,_,?d,_)) {
                    ret d;
                }
                case (ast::item_obj(_,_,_,?d,_)) {
                    ret d.ctor; /* This doesn't really make sense */
                }
            }
        }
    }
}

fn has_nonlocal_exits(&ast::block b) -> bool {
   auto has_exits = @mutable false;

   fn visit_expr(@mutable bool flag, &@ast::expr e) {
       alt (e.node) {
           case (ast::expr_break(_)) { *flag = true; }
           case (ast::expr_cont(_)) { *flag = true; }
           case (_) { }
       }
   }

    auto v = rec(visit_expr_pre=bind visit_expr(has_exits, _)
                 with walk::default_visitor());
    walk::walk_block(v, b);
    ret *has_exits;
}

fn local_rhs_span(&@ast::local l, &ast::span def) -> ast::span {
    alt (l.init) {
        case (some(?i)) { ret i.expr.span; }
        case (_) { ret def; }
    }
}

fn respan[T](&span sp, &T t) -> spanned[T] {
    ret rec(node=t, span=sp);
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
