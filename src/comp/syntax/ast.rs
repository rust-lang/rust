// The Rust abstract syntax tree.

import std::ivec;
import std::option;
import std::str;
import codemap::span;
import codemap::filename;

type spanned[T] = {node: T, span: span};
fn respan[T](sp: &span, t: &T) -> spanned[T] { ret {node: t, span: sp}; }

type ident = str;
// Functions may or may not have names.
type fn_ident = option::t[ident];

// FIXME: with typestate constraint, could say
// idents and types are the same length, and are
// non-empty
type path_ = {global: bool, idents: [ident], types: [@ty]};

type path = spanned[path_];

fn path_name(p: &path) -> str { path_name_i(p.node.idents) }

fn path_name_i(idents: &[ident]) -> str { str::connect(idents, "::") }

type crate_num = int;
type node_id = int;
type def_id = {crate: crate_num, node: node_id};

const local_crate: crate_num = 0;
fn local_def(id: node_id) -> def_id { ret {crate: local_crate, node: id}; }

type ty_param = {ident: ident, kind: kind};

tag def {
    def_fn(def_id, purity);
    def_obj_field(def_id);
    def_mod(def_id);
    def_native_mod(def_id);
    def_const(def_id);
    def_arg(def_id);
    def_local(def_id);
    def_variant(def_id, /* tag */def_id);
    /* variant */
    def_ty(def_id);
    def_ty_arg(uint, kind);
    def_binding(def_id);
    def_use(def_id);
    def_native_ty(def_id);
    def_native_fn(def_id);
    /* A "fake" def for upvars. This never appears in the def_map, but
     * freevars::def_lookup will return it for a def that is an upvar.
     * It contains the actual def. */
    def_upvar(def_id, @def);
}

fn variant_def_ids(d: &def) -> {tg: def_id, var: def_id} {
    alt d { def_variant(tag_id, var_id) { ret {tg: tag_id, var: var_id}; } }
}

fn def_id_of_def(d: def) -> def_id {
    alt d {
      def_fn(id, _) { ret id; }
      def_obj_field(id) { ret id; }
      def_mod(id) { ret id; }
      def_native_mod(id) { ret id; }
      def_const(id) { ret id; }
      def_arg(id) { ret id; }
      def_local(id) { ret id; }
      def_variant(_, id) { ret id; }
      def_ty(id) { ret id; }
      def_ty_arg(_,_) { fail; }
      def_binding(id) { ret id; }
      def_use(id) { ret id; }
      def_native_ty(id) { ret id; }
      def_native_fn(id) { ret id; }
      def_upvar(id, _) { ret id; }
    }
}

// The set of meta_items that define the compilation environment of the crate,
// used to drive conditional compilation
type crate_cfg = [@meta_item];

type crate = spanned[crate_];

type crate_ =
    {directives: [@crate_directive],
     module: _mod,
     attrs: [attribute],
     config: crate_cfg};

tag crate_directive_ {
    cdir_src_mod(ident, option::t[filename], [attribute]);
    cdir_dir_mod(ident,
                 option::t[filename],
                 [@crate_directive],
                 [attribute]);
    cdir_view_item(@view_item);
    cdir_syntax(path);
    cdir_auth(path, _auth);
}

type crate_directive = spanned[crate_directive_];

type meta_item = spanned[meta_item_];

tag meta_item_ {
    meta_word(ident);
    meta_list(ident, [@meta_item]);
    meta_name_value(ident, lit);
}

type blk = spanned[blk_];

type blk_ = {stmts: [@stmt], expr: option::t[@expr], id: node_id};

type pat = {id: node_id, node: pat_, span: span};

type field_pat = {ident: ident, pat: @pat};

tag pat_ {
    pat_wild;
    pat_bind(ident);
    pat_lit(@lit);
    pat_tag(path, [@pat]);
    pat_rec([field_pat], bool);
    pat_box(@pat);
}

type pat_id_map = std::map::hashmap[str, ast::node_id];

// This is used because same-named variables in alternative patterns need to
// use the node_id of their namesake in the first pattern.
fn pat_id_map(pat: &@pat) -> pat_id_map {
    let map = std::map::new_str_hash[node_id]();
    fn walk(map: &pat_id_map, pat: &@pat) {
        alt pat.node {
          pat_bind(name) { map.insert(name, pat.id); }
          pat_tag(_, sub) { for p: @pat in sub { walk(map, p); } }
          pat_rec(fields, _) {
            for f: field_pat  in fields { walk(map, f.pat); }
          }
          pat_box(inner) { walk(map, inner); }
          _ { }
        }
    }
    walk(map, pat);
    ret map;
}

iter pat_bindings(pat: &@pat) -> @pat {
    alt pat.node {
      pat_bind(_) { put pat; }
      pat_tag(_, sub) {
        for p in sub {
            for each b in pat_bindings(p) { put b; }
        }
      }
      pat_rec(fields, _) {
        for f in fields {
            for each b in pat_bindings(f.pat) { put b; }
        }
      }
      pat_box(sub) {
        for each b in pat_bindings(sub) { put b; }
      }
      pat_wild. | pat_lit(_) {}
    }
}

fn pat_binding_ids(pat: &@pat) -> [node_id] {
    let found = ~[];
    for each b in pat_bindings(pat) { found += ~[b.id]; }
    ret found;
}

tag mutability { mut; imm; maybe_mut; }

tag kind { kind_pinned; kind_shared; kind_unique; }

tag _auth { auth_unsafe; }

tag proto { proto_iter; proto_fn; proto_block; proto_closure; }

tag binop {
    add;
    sub;
    mul;
    div;
    rem;
    and;
    or;
    bitxor;
    bitand;
    bitor;
    lsl;
    lsr;
    asr;
    eq;
    lt;
    le;
    ne;
    ge;
    gt;
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

pred lazy_binop(b: binop) -> bool {
    alt b { and. { true } or. { true } _ { false } }
}

tag unop { box(mutability); deref; not; neg; }

fn unop_to_str(op: unop) -> str {
    alt op {
      box(mt) { if mt == mut { ret "@mutable "; } ret "@"; }
      deref. { ret "*"; }
      not. { ret "!"; }
      neg. { ret "-"; }
    }
}

tag mode { val; alias(bool); move; }

type stmt = spanned[stmt_];

tag stmt_ {
    stmt_decl(@decl, node_id);
    stmt_expr(@expr, node_id);
    // These only exist in crate-level blocks.
    stmt_crate_directive(@crate_directive);
}

tag init_op { init_assign; init_move; }

type initializer = {op: init_op, expr: @expr};

type local_ = {ty: @ty,
               pat: @pat,
               init: option::t[initializer],
               id: node_id};

type local = spanned[local_];

type decl = spanned[decl_];

tag decl_ { decl_local([@local]); decl_item(@item); }

type arm = {pats: [@pat], block: blk};

type field_ = {mut: mutability, ident: ident, expr: @expr};

type field = spanned[field_];

tag spawn_dom { dom_implicit; dom_thread; }

tag check_mode { checked; unchecked; }

// FIXME: temporary
tag seq_kind { sk_unique; sk_rc; }

type expr = {id: node_id, node: expr_, span: span};

tag expr_ {
    expr_vec([@expr], mutability, seq_kind);
    expr_rec([field], option::t[@expr]);
    expr_call(@expr, [@expr]);
    expr_tup([@expr]);
    expr_self_method(ident);
    expr_bind(@expr, [option::t[@expr]]);
    expr_spawn(spawn_dom, option::t[str], @expr, [@expr]);
    expr_binary(binop, @expr, @expr);
    expr_unary(unop, @expr);
    expr_lit(@lit);
    expr_cast(@expr, @ty);
    expr_if(@expr, blk, option::t[@expr]);
    expr_ternary(@expr, @expr, @expr);
    expr_while(@expr, blk);
    expr_for(@local, @expr, blk);
    expr_for_each(@local, @expr, blk);
    expr_do_while(blk, @expr);
    expr_alt(@expr, [arm]);
    expr_fn(_fn);
    expr_block(blk);
    /*
     * FIXME: many of these @exprs should be constrained with
     * is_lval once we have constrained types working.
     */
    expr_move(@expr, @expr);
    expr_assign(@expr, @expr);
    expr_swap(@expr, @expr);
    expr_assign_op(binop, @expr, @expr);
    expr_send(@expr, @expr);
    expr_recv(@expr, @expr);
    expr_field(@expr, ident);
    expr_index(@expr, @expr);
    expr_path(path);
    expr_fail(option::t[@expr]);
    expr_break;
    expr_cont;
    expr_ret(option::t[@expr]);
    expr_put(option::t[@expr]);
    expr_be(@expr);
    expr_log(int, @expr);
    /* just an assert, no significance to typestate */
    expr_assert(@expr);
    /* preds that typestate is aware of */
    expr_check(check_mode, @expr);
    /* FIXME Would be nice if expr_check desugared
       to expr_if_check. */
    expr_if_check(@expr, blk, option::t[@expr]);
    expr_port(@ty);
    expr_chan(@expr);
    expr_anon_obj(anon_obj);
    expr_mac(mac);
}

type mac = spanned[mac_];

tag mac_ {
    mac_invoc(path, @expr, option::t[str]);
    mac_embed_type(@ty);
    mac_embed_block(blk);
    mac_ellipsis;
}

type lit = spanned[lit_];

tag lit_ {
    lit_str(str, seq_kind);
    lit_char(char);
    lit_int(int);
    lit_uint(uint);
    lit_mach_int(ty_mach, int);
    lit_float(str);
    lit_mach_float(ty_mach, str);
    lit_nil;
    lit_bool(bool);
}

fn is_path(e: &@expr) -> bool {
    ret alt e.node { expr_path(_) { true } _ { false } };
}


// NB: If you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
type mt = {ty: @ty, mut: mutability};

type ty_field_ = {ident: ident, mt: mt};

type ty_arg_ = {mode: mode, ty: @ty};

type ty_method_ =
    {proto: proto,
     ident: ident,
     inputs: [ty_arg],
     output: @ty,
     cf: controlflow,
     constrs: [@constr]};

type ty_field = spanned[ty_field_];

type ty_arg = spanned[ty_arg_];

type ty_method = spanned[ty_method_];

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

fn ty_mach_to_str(tm: ty_mach) -> str {
    alt tm {
      ty_u8. { ret "u8"; }
      ty_u16. { ret "u16"; }
      ty_u32. { ret "u32"; }
      ty_u64. { ret "u64"; }
      ty_i8. { ret "i8"; }
      ty_i16. { ret "i16"; }
      ty_i32. { ret "i32"; }
      ty_i64. { ret "i64"; }
      ty_f32. { ret "f32"; }
      ty_f64. { ret "f64"; }
    }
}

type ty = spanned[ty_];

tag ty_ {
    ty_nil;
    ty_bot; /* return type of ! functions and type of
             ret/fail/break/cont. there is no syntax
             for this type. */
     /* bot represents the value of functions that don't return a value
        locally to their context. in contrast, things like log that do
        return, but don't return a meaningful value, have result type nil. */
    ty_bool;
    ty_int;
    ty_uint;
    ty_float;
    ty_machine(ty_mach);
    ty_char;
    ty_str;
    ty_istr; // interior string
    ty_box(mt);
    ty_vec(mt);
    ty_ivec(mt); // interior vector
    ty_ptr(mt);
    ty_task;
    ty_port(@ty);
    ty_chan(@ty);
    ty_rec([ty_field]);
    ty_fn(proto, [ty_arg], @ty, controlflow, [@constr]);
    ty_obj([ty_method]);
    ty_tup([@ty]);
    ty_path(path, node_id);
    ty_type;
    ty_constr(@ty, [@ty_constr]);
    ty_mac(mac);
    // ty_infer means the type should be inferred instead of it having been
    // specified. This should only appear at the "top level" of a type and not
    // nested in one.
    ty_infer;
}


/*
A constraint arg that's a function argument is referred to by its position
rather than name.  This is so we could have higher-order functions that have
constraints (potentially -- right now there's no way to write that), and also
so that the typestate pass doesn't have to map a function name onto its decl.
So, the constr_arg type is parameterized: it's instantiated with uint for
declarations, and ident for uses.
*/
tag constr_arg_general_[T] { carg_base; carg_ident(T); carg_lit(@lit); }

type fn_constr_arg = constr_arg_general_[uint];
type sp_constr_arg[T] = spanned[constr_arg_general_[T]];
type ty_constr_arg = sp_constr_arg[path];
type constr_arg = spanned[fn_constr_arg];

// Constrained types' args are parameterized by paths, since
// we refer to paths directly and not by indices.
// The implicit root of such path, in the constraint-list for a
// constrained type, is * (referring to the base record)

type constr_general_[ARG, ID] =
    {path: path, args: [@spanned[constr_arg_general_[ARG]]], id: ID};

// In the front end, constraints have a node ID attached.
// Typeck turns this to a def_id, using the output of resolve.
type constr_general[ARG] = spanned[constr_general_[ARG, node_id]];
type constr_ = constr_general_[uint, node_id];
type constr = spanned[constr_general_[uint, node_id]];
type ty_constr_ = ast::constr_general_[ast::path, ast::node_id];
type ty_constr = spanned[ty_constr_];

/* The parser generates ast::constrs; resolve generates
 a mapping from each function to a list of ty::constr_defs,
 corresponding to these. */
type arg = {mode: mode, ty: @ty, ident: ident, id: node_id};

tag inlineness { il_normal; il_inline; }

type fn_decl =
    {inputs: [arg],
     output: @ty,
     purity: purity,
     il: inlineness,
     cf: controlflow,
     constraints: [@constr]};

tag purity {
    pure_fn; // declared with "pred"
    impure_fn; // declared with "fn"
}

tag controlflow {
    noreturn; // functions with return type _|_ that always
              // raise an error or exit (i.e. never return to the caller)
    return; // everything else
}

type _fn = {decl: fn_decl, proto: proto, body: blk};

type method_ = {ident: ident, meth: _fn, id: node_id};

type method = spanned[method_];

type obj_field = {mut: mutability, ty: @ty, ident: ident, id: node_id};
type anon_obj_field =
    {mut: mutability, ty: @ty, expr: @expr, ident: ident, id: node_id};

type _obj = {fields: [obj_field], methods: [@method]};

type anon_obj =
    // New fields and methods, if they exist.
    {fields: option::t[[anon_obj_field]],
     methods: [@method],
     // inner_obj: the original object being extended, if it exists.
     inner_obj: option::t[@expr]};

type _mod = {view_items: [@view_item], items: [@item]};

tag native_abi {
    native_abi_rust;
    native_abi_cdecl;
    native_abi_llvm;
    native_abi_rust_intrinsic;
    native_abi_x86stdcall;
}

type native_mod =
    {native_name: str,
     abi: native_abi,
     view_items: [@view_item],
     items: [@native_item]};

type variant_arg = {ty: @ty, id: node_id};

type variant_ = {name: str, args: [variant_arg], id: node_id};

type variant = spanned[variant_];

type view_item = spanned[view_item_];

tag view_item_ {
    view_item_use(ident, [@meta_item], node_id);
    view_item_import(ident, [ident], node_id);
    view_item_import_glob([ident], node_id);
    view_item_export(ident, node_id);
}

type obj_def_ids = {ty: node_id, ctor: node_id};


// Meta-data associated with an item
type attribute = spanned[attribute_];


// Distinguishes between attributes that decorate items and attributes that
// are contained as statements within items. These two cases need to be
// distinguished for pretty-printing.
tag attr_style { attr_outer; attr_inner; }

type attribute_ = {style: attr_style, value: meta_item};

type item =  // For objs and resources, this is the type def_id
    {ident: ident, attrs: [attribute], id: node_id, node: item_, span: span};

tag item_ {
    item_const(@ty, @expr);
    item_fn(_fn, [ty_param]);
    item_mod(_mod);
    item_native_mod(native_mod);
    item_ty(@ty, [ty_param]);
    item_tag([variant], [ty_param]);
    item_obj(_obj, [ty_param], /* constructor id */node_id);
    item_res(_fn, /* dtor */
             node_id, /* dtor id */
             [ty_param],
             node_id /* ctor id */);
}

type native_item =
    {ident: ident,
     attrs: [attribute],
     node: native_item_,
     id: node_id,
     span: span};

tag native_item_ {
    native_item_ty;
    native_item_fn(option::t[str], fn_decl, [ty_param]);
}

fn is_exported(i: ident, m: _mod) -> bool {
    let nonlocal = true;
    for it: @ast::item  in m.items {
        if it.ident == i { nonlocal = false; }
        alt it.node {
          item_tag(variants, _) {
            for v: variant  in variants {
                if v.node.name == i { nonlocal = false; }
            }
          }
          _ { }
        }
        if !nonlocal { break; }
    }
    let count = 0u;
    for vi: @ast::view_item  in m.view_items {
        alt vi.node {
          ast::view_item_export(id, _) {
            if str::eq(i, id) {
                // even if it's nonlocal (since it's explicit)

                ret true;
            }
            count += 1u;
          }
          _ {/* fall through */ }
        }
    }
    // If there are no declared exports then
    // everything not imported is exported

    ret count == 0u && !nonlocal;
}

fn is_call_expr(e: @expr) -> bool {
    alt e.node { expr_call(_, _) { ret true; } _ { ret false; } }
}

fn is_constraint_arg(e: @expr) -> bool {
    alt e.node {
      expr_lit(_) { ret true; }
      expr_path(_) { ret true; }
      _ { ret false; }
    }
}

fn eq_ty(a: &@ty, b: &@ty) -> bool { ret std::box::ptr_eq(a, b); }

fn hash_ty(t: &@ty) -> uint { ret t.span.lo << 16u + t.span.hi; }

fn block_from_expr(e: @expr) -> blk {
    let blk_ = {stmts: ~[], expr: option::some[@expr](e), id: e.id};
    ret {node: blk_, span: e.span};
}


fn obj_field_from_anon_obj_field(f: &anon_obj_field) -> obj_field {
    ret {mut: f.mut, ty: f.ty, ident: f.ident, id: f.id};
}

// This is a convenience function to transfor ternary expressions to if
// expressions so that they can be treated the same
fn ternary_to_if(e: &@expr) -> @ast::expr {
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
