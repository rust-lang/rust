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
    def_ty_arg(uint, kind);
    def_binding(def_id);
    def_use(def_id);
    def_native_ty(def_id);
    def_native_fn(def_id);
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
    cdir_src_mod(ident, option::t<filename>, [attribute]);
    cdir_dir_mod(ident, option::t<filename>, [@crate_directive], [attribute]);
    cdir_view_item(@view_item);
    cdir_syntax(path);
    cdir_auth(path, _auth);
}

type crate_directive = spanned<crate_directive_>;

type meta_item = spanned<meta_item_>;

tag meta_item_ {
    meta_word(ident);
    meta_list(ident, [@meta_item]);
    meta_name_value(ident, lit);
}

type blk = spanned<blk_>;

type blk_ =
    {stmts: [@stmt], expr: option::t<@expr>, id: node_id, rules: check_mode};

type pat = {id: node_id, node: pat_, span: span};

type field_pat = {ident: ident, pat: @pat};

tag pat_ {
    pat_wild;
    pat_bind(ident);
    pat_lit(@lit);
    pat_tag(path, [@pat]);
    pat_rec([field_pat], bool);
    pat_tup([@pat]);
    pat_box(@pat);
    pat_uniq(@pat);
    pat_range(@lit, @lit);
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

tag unop {
    box(mutability);
    uniq(mutability);
    deref; not; neg;
}

tag mode { by_ref; by_val; by_mut_ref; by_move; mode_infer; }

type stmt = spanned<stmt_>;

tag stmt_ {
    stmt_decl(@decl, node_id);
    stmt_expr(@expr, node_id);

    // These only exist in crate-level blocks.
    stmt_crate_directive(@crate_directive);
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

tag check_mode { checked; unchecked; }

type expr = {id: node_id, node: expr_, span: span};

tag expr_ {
    expr_vec([@expr], mutability);
    expr_rec([field], option::t<@expr>);
    expr_call(@expr, [@expr]);
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
    expr_for_each(@local, @expr, blk);
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
    expr_path(path);
    expr_fail(option::t<@expr>);
    expr_break;
    expr_cont;
    expr_ret(option::t<@expr>);
    expr_put(option::t<@expr>);
    expr_be(@expr);
    expr_log(int, @expr);

    /* just an assert, no significance to typestate */
    expr_assert(@expr);

    /* preds that typestate is aware of */
    expr_check(check_mode, @expr);

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
    mac_invoc(path, @expr, option::t<str>);
    mac_embed_type(@ty);
    mac_embed_block(blk);
    mac_ellipsis;
}

type lit = spanned<lit_>;

tag lit_ {
    lit_str(str);
    lit_char(char);
    lit_int(int);
    lit_uint(uint);
    lit_mach_int(ty_mach, int);
    lit_float(str);
    lit_mach_float(ty_mach, str);
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
    ty_int;
    ty_uint;
    ty_float;
    ty_machine(ty_mach);
    ty_char;
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
tag constr_arg_general_<T> { carg_base; carg_ident(T); carg_lit(@lit); }

type fn_constr_arg = constr_arg_general_<uint>;
type sp_constr_arg<T> = spanned<constr_arg_general_<T>>;
type ty_constr_arg = sp_constr_arg<path>;
type constr_arg = spanned<fn_constr_arg>;

// Constrained types' args are parameterized by paths, since
// we refer to paths directly and not by indices.
// The implicit root of such path, in the constraint-list for a
// constrained type, is * (referring to the base record)

type constr_general_<ARG, ID> =
    {path: path, args: [@spanned<constr_arg_general_<ARG>>], id: ID};

// In the front end, constraints have a node ID attached.
// Typeck turns this to a def_id, using the output of resolve.
type constr_general<ARG> = spanned<constr_general_<ARG, node_id>>;
type constr_ = constr_general_<uint, node_id>;
type constr = spanned<constr_general_<uint, node_id>>;
type ty_constr_ = ast::constr_general_<ast::path, ast::node_id>;
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
    impure_fn; // declared with "fn"
}

tag ret_style {
    noreturn; // functions with return type _|_ that always
              // raise an error or exit (i.e. never return to the caller)
    return_val; // everything else
    return_ref(bool, uint);
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
    native_abi_rust;
    native_abi_cdecl;
    native_abi_llvm;
    native_abi_rust_intrinsic;
    native_abi_x86stdcall;
    native_abi_c_stack_cdecl;
    native_abi_c_stack_stdcall;
}

type native_mod =
    {native_name: str,
     abi: native_abi,
     view_items: [@view_item],
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
    view_item_import(ident, simple_path, node_id);
    view_item_import_glob(simple_path, node_id);
    view_item_import_from(simple_path, [import_ident], node_id);
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
    item_res(_fn,

             /* dtor */
             node_id,

             /* dtor id */
             [ty_param],

             /* ctor id */
             node_id);
}

type native_item =
    {ident: ident,
     attrs: [attribute],
     node: native_item_,
     id: node_id,
     span: span};

tag native_item_ {
    native_item_ty;
    native_item_fn(option::t<str>, fn_decl, [ty_param]);
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
