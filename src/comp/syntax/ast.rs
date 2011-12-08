// The Rust abstract syntax tree.

import std::option;
import codemap::{span, filename};

type spanned<T> = {node: T, span: span};

type ident = str;

// Functions may or may not have names.
type fn_ident = option::t<ident>;

// FIXME: with typestate constraint, could say
// idents and types are the same length, and are
// non-empty
type path_ = {global: bool, idents: [ident], types: [@ty]};

type path = spanned<path_>;

type crate_num = int;
type node_id = int;
type def_id = {crate: crate_num, node: node_id};

const local_crate: crate_num = 0;

type ty_param = {ident: ident, kind: kind};

tag def {
    def_fn(def_id, purity);
    def_obj_field(def_id, mutability);
    def_mod(def_id);
    def_native_mod(def_id);
    def_const(def_id);
    def_arg(def_id, mode);
    def_local(def_id, let_style);
    def_variant(def_id, /* tag */def_id);

    /* variant */
    def_ty(def_id);
    def_ty_param(uint, kind);
    def_binding(def_id);
    def_use(def_id);
    def_native_ty(def_id);
    def_native_fn(def_id, purity);
    def_upvar(def_id, @def, /* writable */bool);
}

// The set of meta_items that define the compilation environment of the crate,
// used to drive conditional compilation
type crate_cfg = [@meta_item];

type crate = spanned<crate_>;

type crate_ =
    {directives: [@crate_directive],
     module: _mod,
     attrs: [attribute],
     config: crate_cfg};

tag crate_directive_ {
    cdir_src_mod(ident, [attribute]);
    cdir_dir_mod(ident, [@crate_directive], [attribute]);

    // NB: cdir_view_item is *not* processed by the rest of the compiler; the
    // attached view_items are sunk into the crate's module during parsing,
    // and processed (resolved, imported, etc.) there. This tag-variant exists
    // only to preserve the view items in order in case we decide to
    // pretty-print crates in the future.
    cdir_view_item(@view_item);

    cdir_syntax(@path);
}

type crate_directive = spanned<crate_directive_>;

type meta_item = spanned<meta_item_>;

tag meta_item_ {
    meta_word(ident);
    meta_list(ident, [@meta_item]);
    meta_name_value(ident, lit);
}

type blk = spanned<blk_>;

type blk_ = {view_items: [@view_item], stmts: [@stmt], expr: option::t<@expr>,
             id: node_id, rules: blk_check_mode};

type pat = {id: node_id, node: pat_, span: span};

type field_pat = {ident: ident, pat: @pat};

tag pat_ {
    pat_wild;
    pat_bind(ident, option::t<@pat>);
    pat_tag(@path, [@pat]);
    pat_rec([field_pat], bool);
    pat_tup([@pat]);
    pat_box(@pat);
    pat_uniq(@pat);
    pat_lit(@expr);
    pat_range(@expr, @expr);
}

tag mutability { mut; imm; maybe_mut; }

tag kind { kind_sendable; kind_copyable; kind_noncopyable; }

tag proto_sugar {
    sugar_normal;
    sugar_sexy;
}

tag proto {
    proto_bare;
    proto_shared(proto_sugar);
    proto_block;
}

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

tag unop {
    box(mutability);
    uniq(mutability);
    deref; not; neg;
}

tag mode { by_ref; by_val; by_mut_ref; by_move; by_copy; mode_infer; }

type stmt = spanned<stmt_>;

tag stmt_ {
    stmt_decl(@decl, node_id);
    stmt_expr(@expr, node_id);
}

tag init_op { init_assign; init_move; }

type initializer = {op: init_op, expr: @expr};

type local_ =  // FIXME: should really be a refinement on pat
    {ty: @ty, pat: @pat, init: option::t<initializer>, id: node_id};

type local = spanned<local_>;

type decl = spanned<decl_>;

tag let_style { let_copy; let_ref; }

tag decl_ { decl_local([(let_style, @local)]); decl_item(@item); }

type arm = {pats: [@pat], guard: option::t<@expr>, body: blk};

type field_ = {mut: mutability, ident: ident, expr: @expr};

type field = spanned<field_>;

tag blk_check_mode { default_blk; unchecked_blk; unsafe_blk; }

tag expr_check_mode { claimed_expr; checked_expr; }

type expr = {id: node_id, node: expr_, span: span};

tag expr_ {
    expr_vec([@expr], mutability);
    expr_rec([field], option::t<@expr>);
    expr_call(@expr, [@expr], bool);
    expr_tup([@expr]);
    expr_self_method(ident);
    expr_bind(@expr, [option::t<@expr>]);
    expr_binary(binop, @expr, @expr);
    expr_unary(unop, @expr);
    expr_lit(@lit);
    expr_cast(@expr, @ty);
    expr_if(@expr, blk, option::t<@expr>);
    expr_ternary(@expr, @expr, @expr);
    expr_while(@expr, blk);
    expr_for(@local, @expr, blk);
    expr_do_while(blk, @expr);
    expr_alt(@expr, [arm]);
    expr_fn(_fn);
    expr_block(blk);

    /*
     * FIXME: many of these @exprs should be constrained with
     * is_lval once we have constrained types working.
     */
    expr_copy(@expr);
    expr_move(@expr, @expr);
    expr_assign(@expr, @expr);
    expr_swap(@expr, @expr);
    expr_assign_op(binop, @expr, @expr);
    expr_field(@expr, ident);
    expr_index(@expr, @expr);
    expr_path(@path);
    expr_fail(option::t<@expr>);
    expr_break;
    expr_cont;
    expr_ret(option::t<@expr>);
    expr_be(@expr);
    expr_log(int, @expr);

    /* just an assert, no significance to typestate */
    expr_assert(@expr);

    /* preds that typestate is aware of */
    expr_check(expr_check_mode, @expr);

    /* FIXME Would be nice if expr_check desugared
       to expr_if_check. */
    expr_if_check(@expr, blk, option::t<@expr>);
    expr_anon_obj(anon_obj);
    expr_mac(mac);
}

/*
// Says whether this is a block the user marked as
// "unchecked"
tag blk_sort {
    blk_unchecked; // declared as "exception to effect-checking rules"
    blk_checked; // all typing rules apply
}
*/

type mac = spanned<mac_>;

tag mac_ {
    mac_invoc(@path, @expr, option::t<str>);
    mac_embed_type(@ty);
    mac_embed_block(blk);
    mac_ellipsis;
}

type lit = spanned<lit_>;

tag lit_ {
    lit_str(str);
    lit_int(i64, int_ty);
    lit_uint(u64, uint_ty);
    lit_float(str, float_ty);
    lit_nil;
    lit_bool(bool);
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
     cf: ret_style,
     constrs: [@constr]};

type ty_field = spanned<ty_field_>;

type ty_arg = spanned<ty_arg_>;

type ty_method = spanned<ty_method_>;

tag int_ty { ty_i; ty_char; ty_i8; ty_i16; ty_i32; ty_i64; }

tag uint_ty { ty_u; ty_u8; ty_u16; ty_u32; ty_u64; }

tag float_ty { ty_f; ty_f32; ty_f64; }

type ty = spanned<ty_>;

tag ty_ {
    ty_nil;
    ty_bot; /* return type of ! functions and type of
             ret/fail/break/cont. there is no syntax
             for this type. */

     /* bot represents the value of functions that don't return a value
        locally to their context. in contrast, things like log that do
        return, but don't return a meaningful value, have result type nil. */
    ty_bool;
    ty_int(int_ty);
    ty_uint(uint_ty);
    ty_float(float_ty);
    ty_str;
    ty_box(mt);
    ty_uniq(mt);
    ty_vec(mt);
    ty_ptr(mt);
    ty_task;
    ty_port(@ty);
    ty_chan(@ty);
    ty_rec([ty_field]);
    ty_fn(proto, [ty_arg], @ty, ret_style, [@constr]);
    ty_obj([ty_method]);
    ty_tup([@ty]);
    ty_path(@path, node_id);
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
tag constr_arg_general_<T> { carg_base; carg_ident(T); carg_lit(@lit); }

type fn_constr_arg = constr_arg_general_<uint>;
type sp_constr_arg<T> = spanned<constr_arg_general_<T>>;
type ty_constr_arg = sp_constr_arg<@path>;
type constr_arg = spanned<fn_constr_arg>;

// Constrained types' args are parameterized by paths, since
// we refer to paths directly and not by indices.
// The implicit root of such path, in the constraint-list for a
// constrained type, is * (referring to the base record)

type constr_general_<ARG, ID> =
    {path: @path, args: [@spanned<constr_arg_general_<ARG>>], id: ID};

// In the front end, constraints have a node ID attached.
// Typeck turns this to a def_id, using the output of resolve.
type constr_general<ARG> = spanned<constr_general_<ARG, node_id>>;
type constr_ = constr_general_<uint, node_id>;
type constr = spanned<constr_general_<uint, node_id>>;
type ty_constr_ = constr_general_<@path, node_id>;
type ty_constr = spanned<ty_constr_>;

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
     cf: ret_style,
     constraints: [@constr]};

tag purity {
    pure_fn; // declared with "pure fn"
    unsafe_fn; // declared with "unsafe fn"
    impure_fn; // declared with "fn"
}

tag ret_style {
    noreturn; // functions with return type _|_ that always
              // raise an error or exit (i.e. never return to the caller)
    return_val; // everything else
}

type _fn = {decl: fn_decl, proto: proto, body: blk};

type method_ = {ident: ident, meth: _fn, id: node_id};

type method = spanned<method_>;

type obj_field = {mut: mutability, ty: @ty, ident: ident, id: node_id};
type anon_obj_field =
    {mut: mutability, ty: @ty, expr: @expr, ident: ident, id: node_id};

type _obj = {fields: [obj_field], methods: [@method]};

type anon_obj =
    // New fields and methods, if they exist.
    // inner_obj: the original object being extended, if it exists.
    {fields: option::t<[anon_obj_field]>,
     methods: [@method],
     inner_obj: option::t<@expr>};

type _mod = {view_items: [@view_item], items: [@item]};

tag native_abi {
    native_abi_rust_intrinsic;
    native_abi_cdecl;
    native_abi_stdcall;
}

type native_mod =
    {view_items: [@view_item],
     items: [@native_item]};

type variant_arg = {ty: @ty, id: node_id};

type variant_ = {name: ident, args: [variant_arg], id: node_id};

type variant = spanned<variant_>;

type view_item = spanned<view_item_>;

// FIXME: May want to just use path here, which would allow things like
// 'import ::foo'
type simple_path = [ident];

type import_ident_ = {name: ident, id: node_id};

type import_ident = spanned<import_ident_>;

tag view_item_ {
    view_item_use(ident, [@meta_item], node_id);
    view_item_import(ident, @simple_path, node_id);
    view_item_import_glob(@simple_path, node_id);
    view_item_import_from(@simple_path, [import_ident], node_id);
    view_item_export([ident], node_id);
}

type obj_def_ids = {ty: node_id, ctor: node_id};


// Meta-data associated with an item
type attribute = spanned<attribute_>;


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
    item_res(_fn /* dtor */,
             node_id /* dtor id */,
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
    native_item_fn(fn_decl, [ty_param]);
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
