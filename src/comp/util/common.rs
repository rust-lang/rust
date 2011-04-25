import std._uint;
import std._int;
import std._vec;
import std.option.none;
import front.ast;
import util.typestate_ann.ts_ann;

import std.io.stdout;
import std.io.str_writer;
import std.io.string_writer;
import pretty.pprust.print_block;
import pretty.pprust.print_expr;
import pretty.pprust.print_decl;
import pretty.pp.mkstate;

type filename = str;
type span = rec(uint lo, uint hi);
type spanned[T] = rec(T node, span span);

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

fn new_str_hash[V]() -> std.map.hashmap[str,V] {
    let std.map.hashfn[str] hasher = std._str.hash;
    let std.map.eqfn[str] eqer = std._str.eq;
    ret std.map.mk_hashmap[str,V](hasher, eqer);
}

fn def_eq(&ast.def_id a, &ast.def_id b) -> bool {
    ret a._0 == b._0 && a._1 == b._1;
}

fn hash_def(&ast.def_id d) -> uint {
    auto h = 5381u;
    h = ((h << 5u) + h) ^ (d._0 as uint);
    h = ((h << 5u) + h) ^ (d._1 as uint);
    ret h;
}

fn new_def_hash[V]() -> std.map.hashmap[ast.def_id,V] {
    let std.map.hashfn[ast.def_id] hasher = hash_def;
    let std.map.eqfn[ast.def_id] eqer = def_eq;
    ret std.map.mk_hashmap[ast.def_id,V](hasher, eqer);
}

fn new_int_hash[V]() -> std.map.hashmap[int,V] {
    fn hash_int(&int x) -> uint { ret x as uint; }
    fn eq_int(&int a, &int b) -> bool { ret a == b; }
    auto hasher = hash_int;
    auto eqer = eq_int;
    ret std.map.mk_hashmap[int,V](hasher, eqer);
}

fn new_uint_hash[V]() -> std.map.hashmap[uint,V] {
    fn hash_uint(&uint x) -> uint { ret x; }
    fn eq_uint(&uint a, &uint b) -> bool { ret a == b; }
    auto hasher = hash_uint;
    auto eqer = eq_uint;
    ret std.map.mk_hashmap[uint,V](hasher, eqer);
}

fn istr(int i) -> str {
    ret _int.to_str(i, 10u);
}

fn uistr(uint i) -> str {
    ret _uint.to_str(i, 10u);
}

fn elt_expr(&ast.elt e) -> @ast.expr { ret e.expr; }

fn elt_exprs(vec[ast.elt] elts) -> vec[@ast.expr] {
    auto f = elt_expr;
    be _vec.map[ast.elt, @ast.expr](f, elts);
}

fn field_expr(&ast.field f) -> @ast.expr { ret f.expr; }

fn field_exprs(vec[ast.field] fields) -> vec [@ast.expr] {
    auto f = field_expr;
    ret _vec.map[ast.field, @ast.expr](f, fields);
}

fn plain_ann(middle.ty.ctxt tcx) -> ast.ann {
  ret ast.ann_type(middle.ty.mk_nil(tcx),
                   none[vec[middle.ty.t]], none[@ts_ann]);
}

fn log_expr(&ast.expr e) -> () {
  let str_writer s = string_writer();
  auto out_ = mkstate(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  comments=none[vec[front.lexer.cmnt]],
                  mutable cur_cmnt=0u);

  print_expr(out, @e);
  log(s.get_str());
}

fn log_block(&ast.block b) -> () {
  let str_writer s = string_writer();
  auto out_ = mkstate(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  comments=none[vec[front.lexer.cmnt]],
                  mutable cur_cmnt=0u);

  print_block(out, b);
  log(s.get_str());
}

fn log_ann(&ast.ann a) -> () {
    alt (a) {
        case (ast.ann_none) {
            log("ann_none");
        }
        case (ast.ann_type(_,_,_)) {
            log("ann_type");
        }
    }
}

fn log_stmt(ast.stmt st) -> () {
  let str_writer s = string_writer();
  auto out_ = mkstate(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  comments=none[vec[front.lexer.cmnt]],
                  mutable cur_cmnt=0u);
  alt (st.node) {
    case (ast.stmt_decl(?decl,_)) {
      print_decl(out, decl);
    }
    case (ast.stmt_expr(?ex,_)) {
      print_expr(out, ex);
    }
    case (_) { /* do nothing */ }
  }
  log(s.get_str());
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
