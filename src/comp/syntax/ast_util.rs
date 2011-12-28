import codemap::span;
import ast::*;

fn respan<copy T>(sp: span, t: T) -> spanned<T> {
    ret {node: t, span: sp};
}

/* assuming that we're not in macro expansion */
fn mk_sp(lo: uint, hi: uint) -> span {
    ret {lo: lo, hi: hi, expanded_from: codemap::os_none};
}

// make this a const, once the compiler supports it
fn dummy_sp() -> span { ret mk_sp(0u, 0u); }

fn path_name(p: @path) -> str { path_name_i(p.node.idents) }

fn path_name_i(idents: [ident]) -> str { str::connect(idents, "::") }

fn local_def(id: node_id) -> def_id { ret {crate: local_crate, node: id}; }

fn variant_def_ids(d: def) -> {tg: def_id, var: def_id} {
    alt d { def_variant(tag_id, var_id) { ret {tg: tag_id, var: var_id}; } }
}

fn def_id_of_def(d: def) -> def_id {
    alt d {
      def_fn(id, _) | def_obj_field(id, _) | def_self(id) | def_mod(id) |
      def_native_mod(id) | def_const(id) | def_arg(id, _) | def_local(id, _) |
      def_variant(_, id) | def_ty(id) | def_ty_param(id, _) |
      def_binding(id) | def_use(id) | def_native_ty(id) |
      def_native_fn(id, _) | def_upvar(id, _, _) { id }
    }
}

type pat_id_map = std::map::hashmap<str, node_id>;

// This is used because same-named variables in alternative patterns need to
// use the node_id of their namesake in the first pattern.
fn pat_id_map(pat: @pat) -> pat_id_map {
    let map = std::map::new_str_hash::<node_id>();
    pat_bindings(pat) {|bound|
        let name = alt bound.node { pat_bind(n, _) { n } };
        map.insert(name, bound.id);
    };
    ret map;
}

// FIXME: could return a constrained type
fn pat_bindings(pat: @pat, it: block(@pat)) {
    alt pat.node {
      pat_bind(_, option::none.) { it(pat); }
      pat_bind(_, option::some(sub)) { it(pat); pat_bindings(sub, it); }
      pat_tag(_, sub) { for p in sub { pat_bindings(p, it); } }
      pat_rec(fields, _) { for f in fields { pat_bindings(f.pat, it); } }
      pat_tup(elts) { for elt in elts { pat_bindings(elt, it); } }
      pat_box(sub) { pat_bindings(sub, it); }
      pat_uniq(sub) { pat_bindings(sub, it); }
      pat_wild. | pat_lit(_) | pat_range(_, _) { }
    }
}

fn pat_binding_ids(pat: @pat) -> [node_id] {
    let found = [];
    pat_bindings(pat) {|b| found += [b.id]; };
    ret found;
}

fn binop_to_str(op: binop) -> str {
    alt op {
      add. { ret "+"; }
      sub. { ret "-"; }
      mul. { ret "*"; }
      div. { ret "/"; }
      rem. { ret "%"; }
      and. { ret "&&"; }
      or. { ret "||"; }
      bitxor. { ret "^"; }
      bitand. { ret "&"; }
      bitor. { ret "|"; }
      lsl. { ret "<<"; }
      lsr. { ret ">>"; }
      asr. { ret ">>>"; }
      eq. { ret "=="; }
      lt. { ret "<"; }
      le. { ret "<="; }
      ne. { ret "!="; }
      ge. { ret ">="; }
      gt. { ret ">"; }
    }
}

pure fn lazy_binop(b: binop) -> bool {
    alt b { and. { true } or. { true } _ { false } }
}

fn unop_to_str(op: unop) -> str {
    alt op {
      box(mt) { if mt == mut { ret "@mutable "; } ret "@"; }
      uniq(mt) { if mt == mut { ret "~mutable "; } ret "~"; }
      deref. { ret "*"; }
      not. { ret "!"; }
      neg. { ret "-"; }
    }
}

fn is_path(e: @expr) -> bool {
    ret alt e.node { expr_path(_) { true } _ { false } };
}

fn int_ty_to_str(t: int_ty) -> str {
    alt t {
      ty_i. { "" } ty_i8. { "i8" } ty_i16. { "i16" }
      ty_i32. { "i32" } ty_i64. { "i64" }
    }
}

fn int_ty_max(t: int_ty) -> u64 {
    alt t {
      ty_i8. { 0x80u64 }
      ty_i16. { 0x800u64 }
      ty_char. | ty_i32. { 0x80000000u64 }
      ty_i64. { 0x8000000000000000u64 }
    }
}

fn uint_ty_to_str(t: uint_ty) -> str {
    alt t {
      ty_u. { "" } ty_u8. { "u8" } ty_u16. { "u16" }
      ty_u32. { "u32" } ty_u64. { "u64" }
    }
}

fn uint_ty_max(t: uint_ty) -> u64 {
    alt t {
      ty_u8. { 0xffu64 }
      ty_u16. { 0xffffu64 }
      ty_u32. { 0xffffffffu64 }
      ty_u64. { 0xffffffffffffffffu64 }
    }
}

fn float_ty_to_str(t: float_ty) -> str {
    alt t { ty_f. { "" } ty_f32. { "f32" } ty_f64. { "f64" } }
}

fn is_exported(i: ident, m: _mod) -> bool {
    let nonlocal = true;
    for it: @item in m.items {
        if it.ident == i { nonlocal = false; }
        alt it.node {
          item_tag(variants, _) {
            for v: variant in variants {
                if v.node.name == i { nonlocal = false; }
            }
          }
          _ { }
        }
        if !nonlocal { break; }
    }
    let count = 0u;
    for vi: @view_item in m.view_items {
        alt vi.node {
          view_item_export(ids, _) {
            for id in ids { if str::eq(i, id) { ret true; } }
            count += 1u;
          }
          _ {/* fall through */ }
        }
    }
    // If there are no declared exports then
    // everything not imported is exported
    // even if it's nonlocal (since it's explicit)
    ret count == 0u && !nonlocal;
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

fn new_def_id_hash<copy T>() -> std::map::hashmap<def_id, T> {
    std::map::mk_hashmap(hash_def_id, eq_def_id)
}

fn block_from_expr(e: @expr) -> blk {
    let blk_ = default_block([], option::some::<@expr>(e), e.id);
    ret {node: blk_, span: e.span};
}

fn default_block(stmts1: [@stmt], expr1: option::t<@expr>, id1: node_id) ->
   blk_ {
    {view_items: [], stmts: stmts1, expr: expr1, id: id1, rules: default_blk}
}

fn obj_field_from_anon_obj_field(f: anon_obj_field) -> obj_field {
    ret {mut: f.mut, ty: f.ty, ident: f.ident, id: f.id};
}

// This is a convenience function to transfor ternary expressions to if
// expressions so that they can be treated the same
fn ternary_to_if(e: @expr) -> @expr {
    alt e.node {
      expr_ternary(cond, then, els) {
        let then_blk = block_from_expr(then);
        let els_blk = block_from_expr(els);
        let els_expr =
            @{id: els.id, node: expr_block(els_blk), span: els.span};
        ret @{id: e.id,
              node: expr_if(cond, then_blk, option::some(els_expr)),
              span: e.span};
      }
      _ { fail; }
    }
}

// FIXME this doesn't handle big integer/float literals correctly (nor does
// the rest of our literal handling)
tag const_val {
    const_float(float);
    const_int(i64);
    const_uint(u64);
    const_str(str);
}

fn eval_const_expr(e: @expr) -> const_val {
    fn fromb(b: bool) -> const_val { const_int(b as i64) }
    alt e.node {
      expr_unary(neg., inner) {
        alt eval_const_expr(inner) {
          const_float(f) { const_float(-f) }
          const_int(i) { const_int(-i) }
          const_uint(i) { const_uint(-i) }
        }
      }
      expr_unary(not., inner) {
        alt eval_const_expr(inner) {
          const_int(i) { const_int(!i) }
          const_uint(i) { const_uint(!i) }
        }
      }
      expr_binary(op, a, b) {
        alt (eval_const_expr(a), eval_const_expr(b)) {
          (const_float(a), const_float(b)) {
            alt op {
              add. { const_float(a + b) } sub. { const_float(a - b) }
              mul. { const_float(a * b) } div. { const_float(a / b) }
              rem. { const_float(a % b) } eq. { fromb(a == b) }
              lt. { fromb(a < b) } le. { fromb(a <= b) } ne. { fromb(a != b) }
              ge. { fromb(a >= b) } gt. { fromb(a > b) }
            }
          }
          (const_int(a), const_int(b)) {
            alt op {
              add. { const_int(a + b) } sub. { const_int(a - b) }
              mul. { const_int(a * b) } div. { const_int(a / b) }
              rem. { const_int(a % b) } and. | bitand. { const_int(a & b) }
              or. | bitor. { const_int(a | b) } bitxor. { const_int(a ^ b) }
              eq. { fromb(a == b) } lt. { fromb(a < b) }
              le. { fromb(a <= b) } ne. { fromb(a != b) }
              ge. { fromb(a >= b) } gt. { fromb(a > b) }
            }
          }
          (const_uint(a), const_uint(b)) {
            alt op {
              add. { const_uint(a + b) } sub. { const_uint(a - b) }
              mul. { const_uint(a * b) } div. { const_uint(a / b) }
              rem. { const_uint(a % b) } and. | bitand. { const_uint(a & b) }
              or. | bitor. { const_uint(a | b) } bitxor. { const_uint(a ^ b) }
              eq. { fromb(a == b) } lt. { fromb(a < b) }
              le. { fromb(a <= b) } ne. { fromb(a != b) }
              ge. { fromb(a >= b) } gt. { fromb(a > b) }
            }
          }
        }
      }
      expr_lit(lit) { lit_to_const(lit) }
    }
}

fn lit_to_const(lit: @lit) -> const_val {
    alt lit.node {
      lit_str(s) { const_str(s) }
      lit_int(n, _) { const_int(n) }
      lit_uint(n, _) { const_uint(n) }
      lit_float(n, _) { const_float(float::from_str(n)) }
      lit_nil. { const_int(0i64) }
      lit_bool(b) { const_int(b as i64) }
    }
}

fn compare_const_vals(a: const_val, b: const_val) -> int {
  alt (a, b) {
    (const_int(a), const_int(b)) { a == b ? 0 : a < b ? -1 : 1 }
    (const_uint(a), const_uint(b)) { a == b ? 0 : a < b ? -1 : 1 }
    (const_float(a), const_float(b)) { a == b ? 0 : a < b ? -1 : 1 }
    (const_str(a), const_str(b)) { a == b ? 0 : a < b ? -1 : 1 }
  }
}

fn compare_lit_exprs(a: @expr, b: @expr) -> int {
  compare_const_vals(eval_const_expr(a), eval_const_expr(b))
}

fn lit_expr_eq(a: @expr, b: @expr) -> bool { compare_lit_exprs(a, b) == 0 }

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
