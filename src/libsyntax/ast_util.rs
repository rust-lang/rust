// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast::*;
use ast;
use ast_util;
use codemap::{span, BytePos};
use parse::token;
use visit;

use core::cmp;
use core::int;
use core::option;
use core::str;
use core::to_bytes;
use core::vec;

pure fn spanned<T>(+lo: BytePos, +hi: BytePos, +t: T) -> spanned<T> {
    respan(mk_sp(lo, hi), move t)
}

pure fn respan<T>(sp: span, +t: T) -> spanned<T> {
    spanned {node: t, span: sp}
}

pure fn dummy_spanned<T>(+t: T) -> spanned<T> {
    respan(dummy_sp(), move t)
}

/* assuming that we're not in macro expansion */
pure fn mk_sp(+lo: BytePos, +hi: BytePos) -> span {
    span {lo: lo, hi: hi, expn_info: None}
}

// make this a const, once the compiler supports it
pure fn dummy_sp() -> span { return mk_sp(BytePos(0), BytePos(0)); }



pure fn path_name_i(idents: &[ident], intr: @token::ident_interner) -> ~str {
    // FIXME: Bad copies (#2543 -- same for everything else that says "bad")
    str::connect(idents.map(|i| *intr.get(*i)), ~"::")
}


pure fn path_to_ident(p: @path) -> ident { vec::last(p.idents) }

pure fn local_def(id: node_id) -> def_id {
    ast::def_id { crate: local_crate, node: id }
}

pure fn is_local(did: ast::def_id) -> bool { did.crate == local_crate }

pure fn stmt_id(s: stmt) -> node_id {
    match s.node {
      stmt_decl(_, id) => id,
      stmt_expr(_, id) => id,
      stmt_semi(_, id) => id,
      stmt_mac(*) => fail ~"attempted to analyze unexpanded stmt",
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
      def_fn(id, _) | def_static_method(id, _, _) | def_mod(id) |
      def_foreign_mod(id) | def_const(id) |
      def_variant(_, id) | def_ty(id) | def_ty_param(id, _) |
      def_use(id) | def_struct(id) => {
        id
      }
      def_arg(id, _) | def_local(id, _) | def_self(id, _) | def_self_ty(id)
      | def_upvar(id, _, _, _) | def_binding(id, _) | def_region(id)
      | def_typaram_binder(id) | def_label(id) => {
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

pure fn binop_to_method_name(op: binop) -> Option<~str> {
    match op {
      add => return Some(~"add"),
      subtract => return Some(~"sub"),
      mul => return Some(~"mul"),
      div => return Some(~"div"),
      rem => return Some(~"modulo"),
      bitxor => return Some(~"bitxor"),
      bitand => return Some(~"bitand"),
      bitor => return Some(~"bitor"),
      shl => return Some(~"shl"),
      shr => return Some(~"shr"),
      lt => return Some(~"lt"),
      le => return Some(~"le"),
      ge => return Some(~"ge"),
      gt => return Some(~"gt"),
      eq => return Some(~"eq"),
      ne => return Some(~"ne"),
      and | or => return None
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
    let mut parent_enum : Option<ident> = None;
    for m.items.each |it| {
        if it.ident == i { local = true; }
        match it.node {
          item_enum(ref enum_definition, _) =>
            for (*enum_definition).variants.each |v| {
                if v.node.name == i {
                    local = true;
                    parent_enum = Some(/* FIXME (#2543) */ copy it.ident);
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
                  ast::view_path_simple(id, _, _, _) => {
                    if id == i { return true; }
                    match parent_enum {
                      Some(parent_enum_id) => {
                        if id == parent_enum_id { return true; }
                      }
                      _ => ()
                    }
                  }

                  ast::view_path_list(path, ref ids, _) => {
                    if vec::len(path.idents) == 1u {
                        if i == path.idents[0] { return true; }
                        for (*ids).each |id| {
                            if id.node.name == i { return true; }
                        }
                    } else {
                        fail ~"export of path-qualified list";
                    }
                  }

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

// This makes def_id hashable
impl def_id : to_bytes::IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_2(&self.crate, &self.node, lsb0, f);
    }
}

fn block_from_expr(e: @expr) -> blk {
    let blk_ = default_block(~[], option::Some::<@expr>(e), e.id);
    return spanned {node: blk_, span: e.span};
}

fn default_block(
    +stmts1: ~[@stmt],
    expr1: Option<@expr>,
    id1: node_id
) -> blk_ {
    ast::blk_ {
        view_items: ~[],
        stmts: stmts1,
        expr: expr1,
        id: id1,
        rules: default_blk,
    }
}

fn ident_to_path(s: span, +i: ident) -> @path {
    @ast::path { span: s,
                 global: false,
                 idents: ~[i],
                 rp: None,
                 types: ~[] }
}

fn ident_to_pat(id: node_id, s: span, +i: ident) -> @pat {
    @ast::pat { id: id,
                node: pat_ident(bind_by_value, ident_to_path(s, i), None),
                span: s }
}

pure fn is_unguarded(a: &arm) -> bool {
    match a.guard {
      None => true,
      _    => false
    }
}

pure fn unguarded_pat(a: &arm) -> Option<~[@pat]> {
    if is_unguarded(a) { Some(/* FIXME (#2543) */ copy a.pats) } else { None }
}

fn public_methods(ms: ~[@method]) -> ~[@method] {
    do ms.filtered |m| {
        match m.vis {
            public => true,
            _   => false
        }
    }
}

// extract a ty_method from a trait_method. if the trait_method is
// a default, pull out the useful fields to make a ty_method
fn trait_method_to_ty_method(method: trait_method) -> ty_method {
    match method {
        required(ref m) => (*m),
        provided(m) => {
            ty_method {
                ident: m.ident,
                attrs: m.attrs,
                purity: m.purity,
                decl: m.decl,
                tps: m.tps,
                self_ty: m.self_ty,
                id: m.id,
                span: m.span,
            }
        }
    }
}

fn split_trait_methods(trait_methods: ~[trait_method])
    -> (~[ty_method], ~[@method]) {
    let mut reqd = ~[], provd = ~[];
    for trait_methods.each |trt_method| {
        match *trt_method {
          required(ref tm) => reqd.push((*tm)),
          provided(m) => provd.push(m)
        }
    };
    (reqd, provd)
}

pure fn struct_field_visibility(field: ast::struct_field) -> visibility {
    match field.node.kind {
        ast::named_field(_, _, visibility) => visibility,
        ast::unnamed_field => ast::public
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
          ii_dtor(_, nm, _, _) => /* FIXME (#2543) */ copy nm
        }
    }

    fn id() -> ast::node_id {
        match self {
          ii_item(i) => i.id,
          ii_foreign(i) => i.id,
          ii_method(_, m) => m.id,
          ii_dtor(ref dtor, _, _, _) => (*dtor).node.id
        }
    }

    fn accept<E>(e: E, v: visit::vt<E>) {
        match self {
          ii_item(i) => (v.visit_item)(i, e, v),
          ii_foreign(i) => (v.visit_foreign_item)(i, e, v),
          ii_method(_, m) => visit::visit_method_helper(m, e, v),
          ii_dtor(ref dtor, _, tps, parent_id) => {
              visit::visit_struct_dtor_helper((*dtor), tps, parent_id, e, v);
          }
        }
    }
}

/* True if d is either a def_self, or a chain of def_upvars
 referring to a def_self */
fn is_self(d: ast::def) -> bool {
  match d {
    def_self(*)           => true,
    def_upvar(_, d, _, _) => is_self(*d),
    _                     => false
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

fn dtor_ty() -> @ast::Ty { @ast::Ty {id: 0, node: ty_nil, span: dummy_sp()} }

fn dtor_dec() -> fn_decl {
    let nil_t = dtor_ty();
    // dtor has no args
    ast::fn_decl {
        inputs: ~[],
        output: nil_t,
        cf: return_val,
    }
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

#[auto_encode]
#[auto_decode]
struct id_range {
    min: node_id,
    max: node_id,
}

fn empty(range: id_range) -> bool {
    range.min >= range.max
}

fn id_visitor(vfn: fn@(node_id)) -> visit::vt<()> {
    visit::mk_simple_visitor(@visit::SimpleVisitor {
        visit_mod: |_m, _sp, id| vfn(id),

        visit_view_item: fn@(vi: @view_item) {
            match vi.node {
              view_item_use(_, _, id) => vfn(id),
              view_item_import(vps) | view_item_export(vps) => {
                  for vec::each(vps) |vp| {
                      match vp.node {
                          view_path_simple(_, _, _, id) => vfn(id),
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
              item_enum(ref enum_definition, _) =>
                for (*enum_definition).variants.each |v| { vfn(v.node.id); },
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

        visit_ty: fn@(t: @Ty) {
            match t.node {
              ty_path(_, id) => vfn(id),
              _ => { /* fall through */ }
            }
        },

        visit_ty_params: fn@(ps: ~[ty_param]) {
            for vec::each(ps) |p| {
                vfn(p.id);
            }
        },

        visit_fn: fn@(fk: visit::fn_kind, d: ast::fn_decl,
                      _b: ast::blk, _sp: span, id: ast::node_id) {
            vfn(id);

            match fk {
                visit::fk_dtor(tps, _, self_id, parent_id) => {
                    for vec::each(tps) |tp| { vfn(tp.id); }
                    vfn(id);
                    vfn(self_id);
                    vfn(parent_id.node);
                }
                visit::fk_item_fn(_, tps, _) => {
                    for vec::each(tps) |tp| { vfn(tp.id); }
                }
                visit::fk_method(_, tps, m) => {
                    vfn(m.self_id);
                    for vec::each(tps) |tp| { vfn(tp.id); }
                }
                visit::fk_anon(_, capture_clause) |
                visit::fk_fn_block(capture_clause) => {
                    for vec::each(*capture_clause) |clause| {
                        vfn(clause.id);
                    }
                }
            }

            for vec::each(d.inputs) |arg| {
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

        visit_struct_field: fn@(f: @struct_field) {
            vfn(f.node.id);
        },

        visit_struct_method: fn@(_m: @method) {
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
    id_range { min: *min, max: *max }
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
        pat_ident(_, _, Some(p)) => walk_pat(p, it),
        pat_rec(fields, _) | pat_struct(_, fields, _) => {
            for fields.each |f| {
                walk_pat(f.pat, it)
            }
        }
        pat_enum(_, Some(s)) | pat_tup(s) => {
            for s.each |p| {
                walk_pat(*p, it)
            }
        }
        pat_box(s) | pat_uniq(s) | pat_region(s) => {
            walk_pat(s, it)
        }
        pat_vec(elts, tail) => {
            for elts.each |p| {
                walk_pat(*p, it)
            }
            do option::iter(&tail) |tail| {
                walk_pat(*tail, it)
            }
        }
        pat_wild | pat_lit(_) | pat_range(_, _) | pat_ident(_, _, _) |
        pat_enum(_, _) => {
        }
    }
}

fn view_path_id(p: @view_path) -> node_id {
    match p.node {
      view_path_simple(_, _, _, id) | view_path_glob(_, id) |
      view_path_list(_, _, id) => id
    }
}

/// Returns true if the given struct def is tuple-like; i.e. that its fields
/// are unnamed.
fn struct_def_is_tuple_like(struct_def: @ast::struct_def) -> bool {
    struct_def.ctor_id.is_some()
}


fn visibility_to_privacy(visibility: visibility,
                         legacy_exports: bool) -> Privacy {
    if legacy_exports {
        match visibility {
            inherited | public => Public,
            private => Private
        }
    } else {
        match visibility {
            public => Public,
            inherited | private => Private
        }
    }
}

enum Privacy {
    Private,
    Public
}

impl Privacy : cmp::Eq {
    pure fn eq(&self, other: &Privacy) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &Privacy) -> bool { !(*self).eq(other) }
}

fn has_legacy_export_attr(attrs: &[attribute]) -> bool {
    for attrs.each |attribute| {
        match attribute.node.value.node {
          meta_word(ref w) if (*w) == ~"legacy_exports" => {
            return true;
          }
          _ => {}
        }
    }
    return false;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
