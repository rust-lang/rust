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

fn local_def(id: node_id) -> def_id { {crate: local_crate, node: id} }

pure fn is_local(did: ast::def_id) -> bool { did.crate == local_crate }

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
      def_upvar(id, _, _) | def_binding(id) | def_region(id) {
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
    let mut local = false;
    let mut parent_enum : option<ident> = none;
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
    let mut has_explicit_exports = false;
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
    std::map::hashmap(hash_def_id, eq_def_id)
}

fn block_from_expr(e: @expr) -> blk {
    let blk_ = default_block([], option::some::<@expr>(e), e.id);
    ret {node: blk_, span: e.span};
}

fn default_block(stmts1: [@stmt], expr1: option<@expr>, id1: node_id) ->
   blk_ {
    {view_items: [], stmts: stmts1, expr: expr1, id: id1, rules: default_blk}
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

type ivar = {ident: ident, ty: @ty, cm: class_mutability,
             id: node_id, privacy: privacy};

type cmethod = {privacy: privacy, meth: @method};

fn public_methods(cms: [cmethod]) -> [@method] {
    vec::filter_map(cms, {|cm| alt cm.privacy {
                    pub { some(cm.meth) }
                    _   { none }}})
}

fn ignore_privacy(cms: [cmethod]) -> [@method] {
    vec::map(cms, {|cm| cm.meth})
}

fn split_class_items(cs: [@class_item]) -> ([ivar], [cmethod]) {
    let mut vs = [], ms = [];
    for c in cs {
      alt c.node.decl {
        instance_var(i, t, cm, id) {
          vs += [{ident: i, ty: t, cm: cm, id: id, privacy: c.node.privacy}];
        }
        class_method(m) { ms += [{privacy: c.node.privacy, meth: m}]; }
      }
    }
    (vs, ms)
}

impl inlined_item_methods for inlined_item {
    fn ident() -> ident {
        alt self {
          ii_item(i) { i.ident }
          ii_native(i) { i.ident }
          ii_method(_, m) { m.ident }
        }
    }

    fn id() -> ast::node_id {
        alt self {
          ii_item(i) { i.id }
          ii_native(i) { i.id }
          ii_method(_, m) { m.id }
        }
    }

    fn accept<E>(e: E, v: visit::vt<E>) {
        alt self {
          ii_item(i) { v.visit_item(i, e, v) }
          ii_native(i) { v.visit_native_item(i, e, v) }
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
