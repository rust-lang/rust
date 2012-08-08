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
pure fn dummy_sp() -> span { return mk_sp(0u, 0u); }

pure fn path_name(p: @path) -> ~str { path_name_i(p.idents) }

pure fn path_name_i(idents: ~[ident]) -> ~str {
    // FIXME: Bad copies (#2543 -- same for everything else that says "bad")
    str::connect(idents.map(|i|*i), ~"::")
}

pure fn path_to_ident(p: @path) -> ident { vec::last(p.idents) }

pure fn local_def(id: node_id) -> def_id { {crate: local_crate, node: id} }

pure fn is_local(did: ast::def_id) -> bool { did.crate == local_crate }

pure fn stmt_id(s: stmt) -> node_id {
    match s.node {
      stmt_decl(_, id) => id,
      stmt_expr(_, id) => id,
      stmt_semi(_, id) => id
    }
}

fn variant_def_ids(d: def) -> {enm: def_id, var: def_id} {
    match d {
      def_variant(enum_id, var_id) => {
        return {enm: enum_id, var: var_id}
      }
      _ => fail ~"non-variant in variant_def_ids"
    }
}

pure fn def_id_of_def(d: def) -> def_id {
    match d {
      def_fn(id, _) | def_static_method(id, _) | def_mod(id) |
      def_foreign_mod(id) | def_const(id) |
      def_variant(_, id) | def_ty(id) | def_ty_param(id, _) |
      def_use(id) | def_class(id, _) => {
        id
      }
      def_arg(id, _) | def_local(id, _) | def_self(id) |
      def_upvar(id, _, _) | def_binding(id, _) | def_region(id)
      | def_typaram_binder(id) => {
        local_def(id)
      }

      def_prim_ty(_) => fail
    }
}

pure fn binop_to_str(op: binop) -> ~str {
    match op {
      add => return ~"+",
      subtract => return ~"-",
      mul => return ~"*",
      div => return ~"/",
      rem => return ~"%",
      and => return ~"&&",
      or => return ~"||",
      bitxor => return ~"^",
      bitand => return ~"&",
      bitor => return ~"|",
      shl => return ~"<<",
      shr => return ~">>",
      eq => return ~"==",
      lt => return ~"<",
      le => return ~"<=",
      ne => return ~"!=",
      ge => return ~">=",
      gt => return ~">"
    }
}

pure fn binop_to_method_name(op: binop) -> option<~str> {
    match op {
      add => return some(~"add"),
      subtract => return some(~"sub"),
      mul => return some(~"mul"),
      div => return some(~"div"),
      rem => return some(~"modulo"),
      bitxor => return some(~"bitxor"),
      bitand => return some(~"bitand"),
      bitor => return some(~"bitor"),
      shl => return some(~"shl"),
      shr => return some(~"shr"),
      and | or | eq | lt | le | ne | ge | gt => return none
    }
}

pure fn lazy_binop(b: binop) -> bool {
    match b {
      and => true,
      or => true,
      _ => false
    }
}

pure fn is_shift_binop(b: binop) -> bool {
    match b {
      shl => true,
      shr => true,
      _ => false
    }
}

pure fn unop_to_str(op: unop) -> ~str {
    match op {
      box(mt) => if mt == m_mutbl { ~"@mut " } else { ~"@" },
      uniq(mt) => if mt == m_mutbl { ~"~mut " } else { ~"~" },
      deref => ~"*",
      not => ~"!",
      neg => ~"-"
    }
}

pure fn is_path(e: @expr) -> bool {
    return match e.node { expr_path(_) => true, _ => false };
}

pure fn int_ty_to_str(t: int_ty) -> ~str {
    match t {
      ty_char => ~"u8", // ???
      ty_i => ~"",
      ty_i8 => ~"i8",
      ty_i16 => ~"i16",
      ty_i32 => ~"i32",
      ty_i64 => ~"i64"
    }
}

pure fn int_ty_max(t: int_ty) -> u64 {
    match t {
      ty_i8 => 0x80u64,
      ty_i16 => 0x8000u64,
      ty_i | ty_char | ty_i32 => 0x80000000u64, // actually ni about ty_i
      ty_i64 => 0x8000000000000000u64
    }
}

pure fn uint_ty_to_str(t: uint_ty) -> ~str {
    match t {
      ty_u => ~"u",
      ty_u8 => ~"u8",
      ty_u16 => ~"u16",
      ty_u32 => ~"u32",
      ty_u64 => ~"u64"
    }
}

pure fn uint_ty_max(t: uint_ty) -> u64 {
    match t {
      ty_u8 => 0xffu64,
      ty_u16 => 0xffffu64,
      ty_u | ty_u32 => 0xffffffffu64, // actually ni about ty_u
      ty_u64 => 0xffffffffffffffffu64
    }
}

pure fn float_ty_to_str(t: float_ty) -> ~str {
    match t { ty_f => ~"f", ty_f32 => ~"f32", ty_f64 => ~"f64" }
}

fn is_exported(i: ident, m: _mod) -> bool {
    let mut local = false;
    let mut parent_enum : option<ident> = none;
    for m.items.each |it| {
        if it.ident == i { local = true; }
        match it.node {
          item_enum(enum_definition, _) =>
            for enum_definition.variants.each |v| {
                if v.node.name == i {
                    local = true;
                    parent_enum = some(/* FIXME (#2543) */ copy it.ident);
                }
            },
          _ => ()
        }
        if local { break; }
    }
    let mut has_explicit_exports = false;
    for m.view_items.each |vi| {
        match vi.node {
          view_item_export(vps) => {
            has_explicit_exports = true;
            for vps.each |vp| {
                match vp.node {
                  ast::view_path_simple(id, _, _) => {
                    if id == i { return true; }
                    match parent_enum {
                      some(parent_enum_id) => {
                        if id == parent_enum_id { return true; }
                      }
                      _ => ()
                    }
                  }

                  ast::view_path_list(path, ids, _) => {
                    if vec::len(path.idents) == 1u {
                        if i == path.idents[0] { return true; }
                        for ids.each |id| {
                            if id.node.name == i { return true; }
                        }
                    } else {
                        fail ~"export of path-qualified list";
                    }
                  }

                  // FIXME: glob-exports aren't supported yet. (#2006)
                  _ => ()
                }
            }
          }
          _ => ()
        }
    }
    // If there are no declared exports then
    // everything not imported is exported
    // even if it's local (since it's explicit)
    return !has_explicit_exports && local;
}

pure fn is_call_expr(e: @expr) -> bool {
    match e.node { expr_call(_, _, _) => true, _ => false }
}

pure fn eq_ty(a: &@ty, b: &@ty) -> bool { box::ptr_eq(*a, *b) }

pure fn hash_ty(t: &@ty) -> uint {
    let res = (t.span.lo << 16u) + t.span.hi;
    return res;
}

pure fn def_eq(a: &ast::def_id, b: &ast::def_id) -> bool {
    a.crate == b.crate && a.node == b.node
}

pure fn hash_def(d: &ast::def_id) -> uint {
    let mut h = 5381u;
    h = (h << 5u) + h ^ (d.crate as uint);
    h = (h << 5u) + h ^ (d.node as uint);
    return h;
}

fn new_def_hash<V: copy>() -> std::map::hashmap<ast::def_id, V> {
    let hasher: std::map::hashfn<ast::def_id> = hash_def;
    let eqer: std::map::eqfn<ast::def_id> = def_eq;
    return std::map::hashmap::<ast::def_id, V>(hasher, eqer);
}

fn block_from_expr(e: @expr) -> blk {
    let blk_ = default_block(~[], option::some::<@expr>(e), e.id);
    return {node: blk_, span: e.span};
}

fn default_block(+stmts1: ~[@stmt], expr1: option<@expr>, id1: node_id) ->
   blk_ {
    {view_items: ~[], stmts: stmts1,
     expr: expr1, id: id1, rules: default_blk}
}

fn ident_to_path(s: span, +i: ident) -> @path {
    @{span: s, global: false, idents: ~[i],
      rp: none, types: ~[]}
}

pure fn is_unguarded(&&a: arm) -> bool {
    match a.guard {
      none => true,
      _    => false
    }
}

pure fn unguarded_pat(a: arm) -> option<~[@pat]> {
    if is_unguarded(a) { some(/* FIXME (#2543) */ copy a.pats) } else { none }
}

pure fn class_item_ident(ci: @class_member) -> ident {
    match ci.node {
      instance_var(i,_,_,_,_) => /* FIXME (#2543) */ copy i,
      class_method(it) => /* FIXME (#2543) */ copy it.ident
    }
}

type ivar = {ident: ident, ty: @ty, cm: class_mutability,
             id: node_id, vis: visibility};

fn public_methods(ms: ~[@method]) -> ~[@method] {
    vec::filter(ms,
                |m| match m.vis {
                    public => true,
                    _   => false
                })
}

fn split_class_items(cs: ~[@class_member]) -> (~[ivar], ~[@method]) {
    let mut vs = ~[], ms = ~[];
    for cs.each |c| {
      match c.node {
        instance_var(i, t, cm, id, vis) => {
          vec::push(vs, {ident: /* FIXME (#2543) */ copy i,
                         ty: t,
                         cm: cm,
                         id: id,
                         vis: vis});
        }
        class_method(m) => vec::push(ms, m)
      }
    };
    (vs, ms)
}

// extract a ty_method from a trait_method. if the trait_method is
// a default, pull out the useful fields to make a ty_method
fn trait_method_to_ty_method(method: trait_method) -> ty_method {
    match method {
      required(m) => m,
      provided(m) => {
        {ident: m.ident, attrs: m.attrs,
         decl: m.decl, tps: m.tps, self_ty: m.self_ty,
         id: m.id, span: m.span}
      }
    }
}

fn split_trait_methods(trait_methods: ~[trait_method])
    -> (~[ty_method], ~[@method]) {
    let mut reqd = ~[], provd = ~[];
    for trait_methods.each |trt_method| {
        match trt_method {
          required(tm) => vec::push(reqd, tm),
          provided(m) => vec::push(provd, m)
        }
    };
    (reqd, provd)
}

pure fn class_member_visibility(ci: @class_member) -> visibility {
  match ci.node {
     instance_var(_, _, _, _, vis) => vis,
     class_method(m) => m.vis
  }
}

trait inlined_item_utils {
    fn ident() -> ident;
    fn id() -> ast::node_id;
    fn accept<E>(e: E, v: visit::vt<E>);
}

impl inlined_item: inlined_item_utils {
    fn ident() -> ident {
        match self {
          ii_item(i) => /* FIXME (#2543) */ copy i.ident,
          ii_foreign(i) => /* FIXME (#2543) */ copy i.ident,
          ii_method(_, m) => /* FIXME (#2543) */ copy m.ident,
          ii_ctor(_, nm, _, _) => /* FIXME (#2543) */ copy nm,
          ii_dtor(_, nm, _, _) => /* FIXME (#2543) */ copy nm
        }
    }

    fn id() -> ast::node_id {
        match self {
          ii_item(i) => i.id,
          ii_foreign(i) => i.id,
          ii_method(_, m) => m.id,
          ii_ctor(ctor, _, _, _) => ctor.node.id,
          ii_dtor(dtor, _, _, _) => dtor.node.id
        }
    }

    fn accept<E>(e: E, v: visit::vt<E>) {
        match self {
          ii_item(i) => v.visit_item(i, e, v),
          ii_foreign(i) => v.visit_foreign_item(i, e, v),
          ii_method(_, m) => visit::visit_method_helper(m, e, v),
          ii_ctor(ctor, nm, tps, parent_id) => {
              visit::visit_class_ctor_helper(ctor, nm, tps, parent_id, e, v);
          }
          ii_dtor(dtor, nm, tps, parent_id) => {
              visit::visit_class_dtor_helper(dtor, tps, parent_id, e, v);
          }
        }
    }
}

/* True if d is either a def_self, or a chain of def_upvars
 referring to a def_self */
fn is_self(d: ast::def) -> bool {
  match d {
    def_self(_)        => true,
    def_upvar(_, d, _) => is_self(*d),
    _                  => false
  }
}

/// Maps a binary operator to its precedence
fn operator_prec(op: ast::binop) -> uint {
  match op {
      mul | div | rem   => 12u,
      // 'as' sits between here with 11
      add | subtract    => 10u,
      shl | shr         =>  9u,
      bitand            =>  8u,
      bitxor            =>  7u,
      bitor             =>  6u,
      lt | le | ge | gt =>  4u,
      eq | ne           =>  3u,
      and               =>  2u,
      or                =>  1u
  }
}

fn dtor_dec() -> fn_decl {
    let nil_t = @{id: 0, node: ty_nil, span: dummy_sp()};
    // dtor has one argument, of type ()
    {inputs: ~[{mode: ast::expl(ast::by_ref),
                ty: nil_t, ident: @~"_", id: 0}],
     output: nil_t, purity: impure_fn, cf: return_val}
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
            match vi.node {
              view_item_use(_, _, id) => vfn(id),
              view_item_import(vps) | view_item_export(vps) => {
                do vec::iter(vps) |vp| {
                    match vp.node {
                      view_path_simple(_, _, id) => vfn(id),
                      view_path_glob(_, id) => vfn(id),
                      view_path_list(_, _, id) => vfn(id)
                    }
                }
              }
            }
        },

        visit_foreign_item: fn@(ni: @foreign_item) {
            vfn(ni.id)
        },

        visit_item: fn@(i: @item) {
            vfn(i.id);
            match i.node {
              item_enum(enum_definition, _) =>
                for enum_definition.variants.each |v| { vfn(v.node.id); },
              _ => ()
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
            vfn(e.callee_id);
            vfn(e.id);
        },

        visit_expr_post: fn@(_e: @expr) {
        },

        visit_ty: fn@(t: @ty) {
            match t.node {
              ty_path(_, id) => vfn(id),
              _ => { /* fall through */ }
            }
        },

        visit_ty_params: fn@(ps: ~[ty_param]) {
            vec::iter(ps, |p| vfn(p.id))
        },

        visit_fn: fn@(fk: visit::fn_kind, d: ast::fn_decl,
                      _b: ast::blk, _sp: span, id: ast::node_id) {
            vfn(id);

            match fk {
              visit::fk_ctor(nm, _, tps, self_id, parent_id) => {
                vec::iter(tps, |tp| vfn(tp.id));
                vfn(id);
                vfn(self_id);
                vfn(parent_id.node);
              }
              visit::fk_dtor(tps, _, self_id, parent_id) => {
                vec::iter(tps, |tp| vfn(tp.id));
                vfn(id);
                vfn(self_id);
                vfn(parent_id.node);
              }
              visit::fk_item_fn(_, tps) => {
                vec::iter(tps, |tp| vfn(tp.id));
              }
              visit::fk_method(_, tps, m) => {
                vfn(m.self_id);
                vec::iter(tps, |tp| vfn(tp.id));
              }
              visit::fk_anon(_, capture_clause)
              | visit::fk_fn_block(capture_clause) => {
                for vec::each(*capture_clause) |clause| {
                    vfn(clause.id);
                }
              }
            }

            do vec::iter(d.inputs) |arg| {
                vfn(arg.id)
            }
        },

        visit_ty_method: fn@(_ty_m: ty_method) {
        },

        visit_trait_method: fn@(_ty_m: trait_method) {
        },

        visit_struct_def: fn@(_sd: @struct_def, _id: ident, _tps: ~[ty_param],
                              _id: node_id) {
        },

        visit_class_item: fn@(c: @class_member) {
            match c.node {
              instance_var(_, _, _, id,_) => vfn(id),
              class_method(_) => ()
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
    do visit_ids_fn |id| {
        *min = int::min(*min, id);
        *max = int::max(*max, id + 1);
    }
    return {min:*min, max:*max};
}

fn compute_id_range_for_inlined_item(item: inlined_item) -> id_range {
    compute_id_range(|f| visit_ids_for_inlined_item(item, f))
}

pure fn is_item_impl(item: @ast::item) -> bool {
    match item.node {
       item_impl(*) => true,
       _            => false
    }
}

fn walk_pat(pat: @pat, it: fn(@pat)) {
    it(pat);
    match pat.node {
      pat_ident(_, pth, some(p)) => walk_pat(p, it),
      pat_rec(fields, _) | pat_struct(_, fields, _) =>
        for fields.each |f| { walk_pat(f.pat, it) },
      pat_enum(_, some(s)) | pat_tup(s) => for s.each |p| {
        walk_pat(p, it)
      },
      pat_box(s) | pat_uniq(s) => walk_pat(s, it),
      pat_wild | pat_lit(_) | pat_range(_, _) | pat_ident(_, _, _)
        | pat_enum(_, _) => ()
    }
}

fn view_path_id(p: @view_path) -> node_id {
    match p.node {
      view_path_simple(_, _, id) | view_path_glob(_, id) |
      view_path_list(_, _, id) => id
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
