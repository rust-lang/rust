import codemap::span;
import ast::*;

fn respan<T: copy>(sp: span, t: T) -> spanned<T> {
    ret {node: t, span: sp};
}

/* assuming that we're not in macro expansion */
fn mk_sp(lo: uint, hi: uint) -> span {
    ret {lo: lo, hi: hi, expn_info: none};
}

// make this a const, once the compiler supports it
fn dummy_sp() -> span { ret mk_sp(0u, 0u); }

fn path_name(p: @path) -> str { path_name_i(p.node.idents) }

fn path_name_i(idents: [ident]) -> str { str::connect(idents, "::") }

fn local_def(id: node_id) -> def_id { ret {crate: local_crate, node: id}; }

fn stmt_id(s: stmt) -> node_id {
    alt s.node {
      stmt_decl(_, id) { id }
      stmt_expr(_, id) { id }
      stmt_semi(_, id) { id }
    }
}

fn variant_def_ids(d: def) -> {enm: def_id, var: def_id} {
    alt d { def_variant(enum_id, var_id) {
            ret {enm: enum_id, var: var_id}; }
        _ { fail "non-variant in variant_def_ids"; } }
}

fn def_id_of_def(d: def) -> def_id {
    alt d {
      def_fn(id, _) | def_mod(id) |
      def_native_mod(id) | def_const(id) |
      def_variant(_, id) | def_ty(id) | def_ty_param(id, _) |
      def_use(id) |
      def_class(id) | def_class_field(_, id) | def_class_method(_, id) { id }

      def_arg(id, _) | def_local(id, _) | def_self(id) |
      def_upvar(id, _, _) | def_binding(id) {
        local_def(id)
      }

      def_prim_ty(_) { fail; }
    }
}

fn binop_to_str(op: binop) -> str {
    alt op {
      add { ret "+"; }
      subtract { ret "-"; }
      mul { ret "*"; }
      div { ret "/"; }
      rem { ret "%"; }
      and { ret "&&"; }
      or { ret "||"; }
      bitxor { ret "^"; }
      bitand { ret "&"; }
      bitor { ret "|"; }
      lsl { ret "<<"; }
      lsr { ret ">>"; }
      asr { ret ">>>"; }
      eq { ret "=="; }
      lt { ret "<"; }
      le { ret "<="; }
      ne { ret "!="; }
      ge { ret ">="; }
      gt { ret ">"; }
    }
}

pure fn lazy_binop(b: binop) -> bool {
    alt b { and { true } or { true } _ { false } }
}

pure fn is_shift_binop(b: binop) -> bool {
    alt b {
      lsl { true }
      lsr { true }
      asr { true }
      _ { false }
    }
}

fn unop_to_str(op: unop) -> str {
    alt op {
      box(mt) { if mt == m_mutbl { ret "@mut "; } ret "@"; }
      uniq(mt) { if mt == m_mutbl { ret "~mut "; } ret "~"; }
      deref { ret "*"; }
      not { ret "!"; }
      neg { ret "-"; }
      addr_of { ret "&"; }
    }
}

fn is_path(e: @expr) -> bool {
    ret alt e.node { expr_path(_) { true } _ { false } };
}

fn int_ty_to_str(t: int_ty) -> str {
    alt t {
      ty_char { "u8" } // ???
      ty_i { "" } ty_i8 { "i8" } ty_i16 { "i16" }
      ty_i32 { "i32" } ty_i64 { "i64" }
    }
}

fn int_ty_max(t: int_ty) -> u64 {
    alt t {
      ty_i8 { 0x80u64 }
      ty_i16 { 0x800u64 }
      ty_i | ty_char | ty_i32 { 0x80000000u64 } // actually ni about ty_i
      ty_i64 { 0x8000000000000000u64 }
    }
}

fn uint_ty_to_str(t: uint_ty) -> str {
    alt t {
      ty_u { "u" } ty_u8 { "u8" } ty_u16 { "u16" }
      ty_u32 { "u32" } ty_u64 { "u64" }
    }
}

fn uint_ty_max(t: uint_ty) -> u64 {
    alt t {
      ty_u8 { 0xffu64 }
      ty_u16 { 0xffffu64 }
      ty_u | ty_u32 { 0xffffffffu64 } // actually ni about ty_u
      ty_u64 { 0xffffffffffffffffu64 }
    }
}

fn float_ty_to_str(t: float_ty) -> str {
    alt t { ty_f { "" } ty_f32 { "f32" } ty_f64 { "f64" } }
}

fn is_exported(i: ident, m: _mod) -> bool {
    let local = false;
    let parent_enum : option<ident> = none;
    for it: @item in m.items {
        if it.ident == i { local = true; }
        alt it.node {
          item_enum(variants, _) {
            for v: variant in variants {
                if v.node.name == i {
                   local = true;
                   parent_enum = some(it.ident);
                }
            }
          }
          _ { }
        }
        if local { break; }
    }
    let has_explicit_exports = false;
    for vi: @view_item in m.view_items {
        alt vi.node {
          view_item_export(vps) {
            has_explicit_exports = true;
            for vp in vps {
                alt vp.node {
                  ast::view_path_simple(id, _, _) {
                    if id == i { ret true; }
                    alt parent_enum {
                      some(parent_enum_id) {
                        if id == parent_enum_id { ret true; }
                      }
                      _ {}
                    }
                  }

                  ast::view_path_list(path, ids, _) {
                    if vec::len(*path) == 1u {
                        if i == path[0] { ret true; }
                        for id in ids {
                            if id.node.name == i { ret true; }
                        }
                    } else {
                        fail "export of path-qualified list";
                    }
                  }

                  // FIXME: glob-exports aren't supported yet.
                  _ {}
                }
            }
          }
          _ {}
        }
    }
    // If there are no declared exports then
    // everything not imported is exported
    // even if it's local (since it's explicit)
    ret !has_explicit_exports && local;
}

pure fn is_call_expr(e: @expr) -> bool {
    alt e.node { expr_call(_, _, _) { true } _ { false } }
}

fn is_constraint_arg(e: @expr) -> bool {
    alt e.node {
      expr_lit(_) { ret true; }
      expr_path(_) { ret true; }
      _ { ret false; }
    }
}

fn eq_ty(&&a: @ty, &&b: @ty) -> bool { ret box::ptr_eq(a, b); }

fn hash_ty(&&t: @ty) -> uint {
    let res = (t.span.lo << 16u) + t.span.hi;
    ret res;
}

fn hash_def_id(&&id: def_id) -> uint {
    (id.crate as uint << 16u) + (id.node as uint)
}

fn eq_def_id(&&a: def_id, &&b: def_id) -> bool {
    a == b
}

fn new_def_id_hash<T: copy>() -> std::map::hashmap<def_id, T> {
    std::map::mk_hashmap(hash_def_id, eq_def_id)
}

fn block_from_expr(e: @expr) -> blk {
    let blk_ = default_block([], option::some::<@expr>(e), e.id);
    ret {node: blk_, span: e.span};
}

fn default_block(stmts1: [@stmt], expr1: option<@expr>, id1: node_id) ->
   blk_ {
    {view_items: [], stmts: stmts1, expr: expr1, id: id1, rules: default_blk}
}

// FIXME this doesn't handle big integer/float literals correctly (nor does
// the rest of our literal handling)
enum const_val {
    const_float(float),
    const_int(i64),
    const_uint(u64),
    const_str(str),
}

// FIXME: issue #1417
fn eval_const_expr(e: @expr) -> const_val {
    fn fromb(b: bool) -> const_val { const_int(b as i64) }
    alt e.node {
      expr_unary(neg, inner) {
        alt eval_const_expr(inner) {
          const_float(f) { const_float(-f) }
          const_int(i) { const_int(-i) }
          const_uint(i) { const_uint(-i) }
          _ { fail "eval_const_expr: bad neg argument"; }
        }
      }
      expr_unary(not, inner) {
        alt eval_const_expr(inner) {
          const_int(i) { const_int(!i) }
          const_uint(i) { const_uint(!i) }
          _ { fail "eval_const_expr: bad not argument"; }
        }
      }
      expr_binary(op, a, b) {
        alt (eval_const_expr(a), eval_const_expr(b)) {
          (const_float(a), const_float(b)) {
            alt op {
              add { const_float(a + b) } subtract { const_float(a - b) }
              mul { const_float(a * b) } div { const_float(a / b) }
              rem { const_float(a % b) } eq { fromb(a == b) }
              lt { fromb(a < b) } le { fromb(a <= b) } ne { fromb(a != b) }
              ge { fromb(a >= b) } gt { fromb(a > b) }
              _ { fail "eval_const_expr: can't apply this binop to floats"; }
            }
          }
          (const_int(a), const_int(b)) {
            alt op {
              add { const_int(a + b) } subtract { const_int(a - b) }
              mul { const_int(a * b) } div { const_int(a / b) }
              rem { const_int(a % b) } and | bitand { const_int(a & b) }
              or | bitor { const_int(a | b) } bitxor { const_int(a ^ b) }
              lsl { const_int(a << b) } lsr { const_int(a >> b) }
              asr { const_int(a >>> b) }
              eq { fromb(a == b) } lt { fromb(a < b) }
              le { fromb(a <= b) } ne { fromb(a != b) }
              ge { fromb(a >= b) } gt { fromb(a > b) }
              _ { fail "eval_const_expr: can't apply this binop to ints"; }
            }
          }
          (const_uint(a), const_uint(b)) {
            alt op {
              add { const_uint(a + b) } subtract { const_uint(a - b) }
              mul { const_uint(a * b) } div { const_uint(a / b) }
              rem { const_uint(a % b) } and | bitand { const_uint(a & b) }
              or | bitor { const_uint(a | b) } bitxor { const_uint(a ^ b) }
              lsl { const_int((a << b) as i64) }
              lsr { const_int((a >> b) as i64) }
              asr { const_int((a >>> b) as i64) }
              eq { fromb(a == b) } lt { fromb(a < b) }
              le { fromb(a <= b) } ne { fromb(a != b) }
              ge { fromb(a >= b) } gt { fromb(a > b) }
              _ { fail "eval_const_expr: can't apply this binop to uints"; }
            }
          }
          _ { fail "eval_constr_expr: bad binary arguments"; }
        }
      }
      expr_lit(lit) { lit_to_const(lit) }
      // Precondition?
      _ {
          fail "eval_const_expr: non-constant expression";
      }
    }
}

fn lit_to_const(lit: @lit) -> const_val {
    alt lit.node {
      lit_str(s) { const_str(s) }
      lit_int(n, _) { const_int(n) }
      lit_uint(n, _) { const_uint(n) }
      lit_float(n, _) { const_float(option::get(float::from_str(n))) }
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

fn compare_lit_exprs(a: @expr, b: @expr) -> int {
  compare_const_vals(eval_const_expr(a), eval_const_expr(b))
}

fn lit_expr_eq(a: @expr, b: @expr) -> bool { compare_lit_exprs(a, b) == 0 }

fn lit_eq(a: @lit, b: @lit) -> bool {
    compare_const_vals(lit_to_const(a), lit_to_const(b)) == 0
}

fn ident_to_path(s: span, i: ident) -> @path {
    @respan(s, {global: false, idents: [i], types: []})
}

pure fn is_unguarded(&&a: arm) -> bool {
    alt a.guard {
      none { true }
      _    { false }
    }
}

pure fn unguarded_pat(a: arm) -> option<[@pat]> {
    if is_unguarded(a) { some(a.pats) } else { none }
}

// Provides an extra node_id to hang callee information on, in case the
// operator is deferred to a user-supplied method. The parser is responsible
// for reserving this id.
fn op_expr_callee_id(e: @expr) -> node_id { e.id - 1 }

pure fn class_item_ident(ci: @class_item) -> ident {
    alt ci.node.decl {
      instance_var(i,_,_,_) { i }
      class_method(it) { it.ident }
    }
}

impl inlined_item_methods for inlined_item {
    fn ident() -> ident {
        alt self {
          ii_item(i) { i.ident }
          ii_method(_, m) { m.ident }
        }
    }

    fn id() -> ast::node_id {
        alt self {
          ii_item(i) { i.id }
          ii_method(_, m) { m.id }
        }
    }

    fn accept<E>(e: E, v: visit::vt<E>) {
        alt self {
          ii_item(i) { v.visit_item(i, e, v) }
          ii_method(_, m) { visit::visit_method_helper(m, e, v) }
        }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
