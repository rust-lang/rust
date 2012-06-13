import codemap::span;
import ast::*;

pure fn spanned<T>(lo: uint, hi: uint, +t: T) -> spanned<T> {
    respan(mk_sp(lo, hi), t)
}

pure fn respan<T>(sp: span, +t: T) -> spanned<T> {
    {node: t, span: sp}
}

pure fn dummy_spanned<T>(+t: T) -> spanned<T> {
    respan(dummy_sp(), t)
}

/* assuming that we're not in macro expansion */
pure fn mk_sp(lo: uint, hi: uint) -> span {
    {lo: lo, hi: hi, expn_info: none}
}

// make this a const, once the compiler supports it
pure fn dummy_sp() -> span { ret mk_sp(0u, 0u); }

pure fn path_name(p: @path) -> str { path_name_i(p.idents) }

pure fn path_name_i(idents: [ident]) -> str {
    // FIXME: Bad copies (#2543 -- same for everything else that says "bad")
    str::connect(idents.map({|i|*i}), "::")
}

pure fn path_to_ident(p: @path) -> ident { vec::last(p.idents) }

pure fn local_def(id: node_id) -> def_id { {crate: local_crate, node: id} }

pure fn is_local(did: ast::def_id) -> bool { did.crate == local_crate }

pure fn stmt_id(s: stmt) -> node_id {
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

pure fn def_id_of_def(d: def) -> def_id {
    alt d {
      def_fn(id, _) | def_mod(id) |
      def_native_mod(id) | def_const(id) |
      def_variant(_, id) | def_ty(id) | def_ty_param(id, _) |
      def_use(id) | def_class(id) { id }
      def_arg(id, _) | def_local(id, _) | def_self(id) |
      def_upvar(id, _, _) | def_binding(id) | def_region(id) {
        local_def(id)
      }

      def_prim_ty(_) { fail; }
    }
}

pure fn binop_to_str(op: binop) -> str {
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
      shl { ret "<<"; }
      shr { ret ">>"; }
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
      shl { true }
      shr { true }
      _ { false }
    }
}

pure fn unop_to_str(op: unop) -> str {
    alt op {
      box(mt) { if mt == m_mutbl { ret "@mut "; } ret "@"; }
      uniq(mt) { if mt == m_mutbl { ret "~mut "; } ret "~"; }
      deref { ret "*"; }
      not { ret "!"; }
      neg { ret "-"; }
    }
}

pure fn is_path(e: @expr) -> bool {
    ret alt e.node { expr_path(_) { true } _ { false } };
}

pure fn int_ty_to_str(t: int_ty) -> str {
    alt t {
      ty_char { "u8" } // ???
      ty_i { "" } ty_i8 { "i8" } ty_i16 { "i16" }
      ty_i32 { "i32" } ty_i64 { "i64" }
    }
}

pure fn int_ty_max(t: int_ty) -> u64 {
    alt t {
      ty_i8 { 0x80u64 }
      ty_i16 { 0x8000u64 }
      ty_i | ty_char | ty_i32 { 0x80000000u64 } // actually ni about ty_i
      ty_i64 { 0x8000000000000000u64 }
    }
}

pure fn uint_ty_to_str(t: uint_ty) -> str {
    alt t {
      ty_u { "u" } ty_u8 { "u8" } ty_u16 { "u16" }
      ty_u32 { "u32" } ty_u64 { "u64" }
    }
}

pure fn uint_ty_max(t: uint_ty) -> u64 {
    alt t {
      ty_u8 { 0xffu64 }
      ty_u16 { 0xffffu64 }
      ty_u | ty_u32 { 0xffffffffu64 } // actually ni about ty_u
      ty_u64 { 0xffffffffffffffffu64 }
    }
}

pure fn float_ty_to_str(t: float_ty) -> str {
    alt t { ty_f { "" } ty_f32 { "f32" } ty_f64 { "f64" } }
}

fn is_exported(i: ident, m: _mod) -> bool {
    let mut local = false;
    let mut parent_enum : option<ident> = none;
    for m.items.each {|it|
        if it.ident == i { local = true; }
        alt it.node {
          item_enum(variants, _, _) {
            for variants.each {|v|
                if v.node.name == i {
                   local = true;
                   parent_enum = some(/* FIXME: bad */ copy it.ident);
                }
            }
          }
          _ { }
        }
        if local { break; }
    }
    let mut has_explicit_exports = false;
    for m.view_items.each {|vi|
        alt vi.node {
          view_item_export(vps) {
            has_explicit_exports = true;
            for vps.each {|vp|
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
                    if vec::len(path.idents) == 1u {
                        if i == path.idents[0] { ret true; }
                        for ids.each {|id|
                            if id.node.name == i { ret true; }
                        }
                    } else {
                        fail "export of path-qualified list";
                    }
                  }

                  // FIXME: glob-exports aren't supported yet. (#2006)
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

fn def_eq(a: ast::def_id, b: ast::def_id) -> bool {
    ret a.crate == b.crate && a.node == b.node;
}

fn hash_def(d: ast::def_id) -> uint {
    let mut h = 5381u;
    h = (h << 5u) + h ^ (d.crate as uint);
    h = (h << 5u) + h ^ (d.node as uint);
    ret h;
}

fn new_def_hash<V: copy>() -> std::map::hashmap<ast::def_id, V> {
    let hasher: std::map::hashfn<ast::def_id> = hash_def;
    let eqer: std::map::eqfn<ast::def_id> = def_eq;
    ret std::map::hashmap::<ast::def_id, V>(hasher, eqer);
}

fn block_from_expr(e: @expr) -> blk {
    let blk_ = default_block([], option::some::<@expr>(e), e.id);
    ret {node: blk_, span: e.span};
}

fn default_block(+stmts1: [@stmt], expr1: option<@expr>, id1: node_id) ->
   blk_ {
    {view_items: [], stmts: stmts1, expr: expr1, id: id1, rules: default_blk}
}

fn ident_to_path(s: span, +i: ident) -> @path {
    @{span: s, global: false, idents: [i],
      rp: none, types: []}
}

pure fn is_unguarded(&&a: arm) -> bool {
    alt a.guard {
      none { true }
      _    { false }
    }
}

pure fn unguarded_pat(a: arm) -> option<[@pat]> {
    if is_unguarded(a) { some(/* FIXME: bad */ copy a.pats) } else { none }
}

// Provides an extra node_id to hang callee information on, in case the
// operator is deferred to a user-supplied method. The parser is responsible
// for reserving this id.
fn op_expr_callee_id(e: @expr) -> node_id { e.id - 1 }

pure fn class_item_ident(ci: @class_member) -> ident {
    alt ci.node {
      instance_var(i,_,_,_,_) { /* FIXME: bad */ copy i }
      class_method(it) { /* FIXME: bad */ copy it.ident }
    }
}

type ivar = {ident: ident, ty: @ty, cm: class_mutability,
             id: node_id, vis: visibility};

fn public_methods(ms: [@method]) -> [@method] {
    vec::filter(ms, {|m| alt m.vis {
                    public { true }
                    _   { false }}})
}

fn split_class_items(cs: [@class_member]) -> ([ivar], [@method]) {
    let mut vs = [], ms = [];
    for cs.each {|c|
      alt c.node {
        instance_var(i, t, cm, id, vis) {
          vs += [{ident: /* FIXME: bad */ copy i,
                  ty: t,
                  cm: cm,
                  id: id,
                  vis: vis}];
        }
        class_method(m) { ms += [m]; }
      }
    };
    (vs, ms)
}

pure fn class_member_visibility(ci: @class_member) -> visibility {
  alt ci.node {
     instance_var(_, _, _, _, vis) { vis }
     class_method(m) { m.vis }
  }
}

impl inlined_item_methods for inlined_item {
    fn ident() -> ident {
        alt self {
          ii_item(i) { /* FIXME: bad */ copy i.ident }
          ii_native(i) { /* FIXME: bad */ copy i.ident }
          ii_method(_, m) { /* FIXME: bad */ copy m.ident }
          ii_ctor(_, nm, _, _) { /* FIXME: bad */ copy nm }
          ii_dtor(_, nm, _, _) { /* FIXME: bad */ copy nm }
        }
    }

    fn id() -> ast::node_id {
        alt self {
          ii_item(i) { i.id }
          ii_native(i) { i.id }
          ii_method(_, m) { m.id }
          ii_ctor(ctor, _, _, _) { ctor.node.id }
          ii_dtor(dtor, _, _, _) { dtor.node.id }
        }
    }

    fn accept<E>(e: E, v: visit::vt<E>) {
        alt self {
          ii_item(i) { v.visit_item(i, e, v) }
          ii_native(i) { v.visit_native_item(i, e, v) }
          ii_method(_, m) { visit::visit_method_helper(m, e, v) }
          ii_ctor(ctor, nm, tps, parent_id) {
              visit::visit_class_ctor_helper(ctor, nm, tps, parent_id, e, v);
          }
          ii_dtor(dtor, nm, tps, parent_id) {
              visit::visit_class_dtor_helper(dtor, tps, parent_id, e, v);
          }
        }
    }
}

/* True if d is either a def_self, or a chain of def_upvars
 referring to a def_self */
fn is_self(d: ast::def) -> bool {
  alt d {
    def_self(_)        { true }
    def_upvar(_, d, _) { is_self(*d) }
    _                  { false }
  }
}

#[doc = "Maps a binary operator to its precedence"]
fn operator_prec(op: ast::binop) -> uint {
  alt op {
      mul | div | rem   { 12u }
      // 'as' sits between here with 11
      add | subtract    { 10u }
      shl | shr         {  9u }
      bitand            {  8u }
      bitxor            {  7u }
      bitor             {  6u }
      lt | le | ge | gt {  4u }
      eq | ne           {  3u }
      and               {  2u }
      or                {  1u }
  }
}

fn dtor_dec() -> fn_decl {
    let nil_t = @{id: 0, node: ty_nil, span: dummy_sp()};
    // dtor has one argument, of type ()
    {inputs: [{mode: ast::expl(ast::by_ref),
               ty: nil_t, ident: @"_", id: 0}],
     output: nil_t, purity: impure_fn, cf: return_val, constraints: []}
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

#[auto_serialize]
type id_range = {min: node_id, max: node_id};

fn empty(range: id_range) -> bool {
    range.min >= range.max
}

fn id_visitor(vfn: fn@(node_id)) -> visit::vt<()> {
    visit::mk_simple_visitor(@{
        visit_mod: fn@(_m: _mod, _sp: span, id: node_id) {
            vfn(id)
        },

        visit_view_item: fn@(vi: @view_item) {
            alt vi.node {
              view_item_use(_, _, id) { vfn(id) }
              view_item_import(vps) | view_item_export(vps) {
                vec::iter(vps) {|vp|
                    alt vp.node {
                      view_path_simple(_, _, id) { vfn(id) }
                      view_path_glob(_, id) { vfn(id) }
                      view_path_list(_, _, id) { vfn(id) }
                    }
                }
              }
            }
        },

        visit_native_item: fn@(ni: @native_item) {
            vfn(ni.id)
        },

        visit_item: fn@(i: @item) {
            vfn(i.id);
            alt i.node {
              item_res(_, _, _, d_id, c_id, _) { vfn(d_id); vfn(c_id); }
              item_enum(vs, _, _) { for vs.each {|v| vfn(v.node.id); } }
              _ {}
            }
        },

        visit_local: fn@(l: @local) {
            vfn(l.node.id);
        },

        visit_block: fn@(b: blk) {
            vfn(b.node.id);
        },

        visit_stmt: fn@(s: @stmt) {
            vfn(ast_util::stmt_id(*s));
        },

        visit_arm: fn@(_a: arm) { },

        visit_pat: fn@(p: @pat) {
            vfn(p.id)
        },

        visit_decl: fn@(_d: @decl) {
        },

        visit_expr: fn@(e: @expr) {
            vfn(e.id);
            alt e.node {
              expr_index(*) | expr_assign_op(*) |
              expr_unary(*) | expr_binary(*) {
                vfn(ast_util::op_expr_callee_id(e));
              }
              _ { /* fallthrough */ }
            }
        },

        visit_ty: fn@(t: @ty) {
            alt t.node {
              ty_path(_, id) {
                vfn(id)
              }
              _ { /* fall through */ }
            }
        },

        visit_ty_params: fn@(ps: [ty_param]) {
            vec::iter(ps) {|p| vfn(p.id) }
        },

        visit_constr: fn@(_p: @path, _sp: span, id: node_id) {
            vfn(id);
        },

        visit_fn: fn@(fk: visit::fn_kind, d: ast::fn_decl,
                      _b: ast::blk, _sp: span, id: ast::node_id) {
            vfn(id);

            alt fk {
              visit::fk_ctor(nm, tps, self_id, parent_id) {
                vec::iter(tps) {|tp| vfn(tp.id)}
                vfn(id);
                vfn(self_id);
                vfn(parent_id.node);
              }
              visit::fk_dtor(tps, self_id, parent_id) {
                vec::iter(tps) {|tp| vfn(tp.id)}
                vfn(id);
                vfn(self_id);
                vfn(parent_id.node);
              }
              visit::fk_item_fn(_, tps) |
              visit::fk_res(_, tps, _) {
                vec::iter(tps) {|tp| vfn(tp.id)}
              }
              visit::fk_method(_, tps, m) {
                vfn(m.self_id);
                vec::iter(tps) {|tp| vfn(tp.id)}
              }
              visit::fk_anon(_, capture_clause)
              | visit::fk_fn_block(capture_clause) {
                for vec::each(*capture_clause) {|clause|
                    vfn(clause.id);
                }
              }
            }

            vec::iter(d.inputs) {|arg|
                vfn(arg.id)
            }
        },

        visit_class_item: fn@(c: @class_member) {
            alt c.node {
              instance_var(_, _, _, id,_) {
                vfn(id)
              }
              class_method(_) {
              }
            }
        }
    })
}

fn visit_ids_for_inlined_item(item: inlined_item, vfn: fn@(node_id)) {
    item.accept((), id_visitor(vfn));
}

fn compute_id_range(visit_ids_fn: fn(fn@(node_id))) -> id_range {
    let min = @mut int::max_value;
    let max = @mut int::min_value;
    visit_ids_fn { |id|
        *min = int::min(*min, id);
        *max = int::max(*max, id + 1);
    }
    ret {min:*min, max:*max};
}

fn compute_id_range_for_inlined_item(item: inlined_item) -> id_range {
    compute_id_range { |f| visit_ids_for_inlined_item(item, f) }
}

pure fn is_item_impl(item: @ast::item) -> bool {
    alt item.node {
       item_impl(*) { true }
       _            { false }
    }
}

fn walk_pat(pat: @pat, it: fn(@pat)) {
    it(pat);
    alt pat.node {
      pat_ident(pth, some(p)) { walk_pat(p, it); }
      pat_rec(fields, _) { for fields.each {|f| walk_pat(f.pat, it); } }
      pat_enum(_, some(s)) | pat_tup(s) { for s.each {|p| walk_pat(p, it); } }
      pat_box(s) | pat_uniq(s) { walk_pat(s, it); }
      pat_wild | pat_lit(_) | pat_range(_, _) | pat_ident(_, _)
        | pat_enum(_, _) {}
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
