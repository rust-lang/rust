// The Rust abstract syntax tree.

import std::ivec;
import std::option;
import std::str;
import std::vec;
import codemap::span;
import codemap::filename;

type spanned[T] = rec(T node, span span);
fn respan[T](&span sp, &T t) -> spanned[T] { ret rec(node=t, span=sp); }

type ident = str;
// Functions may or may not have names.
type fn_ident = option::t[ident];

// FIXME: with typestate constraint, could say
// idents and types are the same length, and are
// non-empty
type path_ = rec(ident[] idents, (@ty)[] types);

type path = spanned[path_];

fn path_name(&path p) -> str { ret str::connect_ivec(p.node.idents, "::"); }

type crate_num = int;
type node_id = int;
type def_id = tup(crate_num, node_id);

const crate_num local_crate = 0;
fn local_def(node_id id) -> def_id {
    ret tup(local_crate, id);
}

type ty_param = ident;

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
    def_ty_arg(uint);
    def_binding(def_id);
    def_use(def_id);
    def_native_ty(def_id);
    def_native_fn(def_id);
}

fn variant_def_ids(&def d) -> tup(def_id, def_id) {
    alt (d) {
        case (def_variant(?tag_id, ?var_id)) { ret tup(tag_id, var_id); }
    }
}

fn def_id_of_def(def d) -> def_id {
    alt (d) {
        case (def_fn(?id,_)) { ret id; }
        case (def_obj_field(?id)) { ret id; }
        case (def_mod(?id)) { ret id; }
        case (def_native_mod(?id)) { ret id; }
        case (def_const(?id)) { ret id; }
        case (def_arg(?id)) { ret id; }
        case (def_local(?id)) { ret id; }
        case (def_variant(_, ?id)) { ret id; }
        case (def_ty(?id)) { ret id; }
        case (def_ty_arg(_)) { fail; }
        case (def_binding(?id)) { ret id; }
        case (def_use(?id)) { ret id; }
        case (def_native_ty(?id)) { ret id; }
        case (def_native_fn(?id)) { ret id; }
    }
    fail;
}

// The set of meta_items that define the compilation environment of the crate,
// used to drive conditional compilation
type crate_cfg = vec[@meta_item];

type crate = spanned[crate_];

type crate_ = rec(vec[@crate_directive] directives,
                  _mod module,
                  vec[attribute] attrs,
                  crate_cfg config);

tag crate_directive_ {
    cdir_src_mod(ident, option::t[filename], vec[attribute]);
    cdir_dir_mod(ident, option::t[filename],
                 vec[@crate_directive], vec[attribute]);
    cdir_view_item(@view_item);
    cdir_syntax(path);
    cdir_auth(path, _auth);
}

type crate_directive = spanned[crate_directive_];

type meta_item = spanned[meta_item_];

tag meta_item_ {
    meta_word(ident);
    meta_list(ident, vec[@meta_item]);
    meta_name_value(ident, lit);
}

type block = spanned[block_];

type block_ = rec(vec[@stmt] stmts, option::t[@expr] expr, node_id id);

type pat = rec(node_id id,
               pat_ node,
               span span);

tag pat_ {
    pat_wild;
    pat_bind(ident);
    pat_lit(@lit);
    pat_tag(path, vec[@pat]);
}

tag mutability { mut; imm; maybe_mut; }

tag layer { layer_value; layer_state; layer_gc; }

tag _auth { auth_unsafe; }

tag proto { proto_iter; proto_fn; }

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

fn binop_to_str(binop op) -> str {
    alt (op) {
        case (add) { ret "+"; }
        case (sub) { ret "-"; }
        case (mul) { ret "*"; }
        case (div) { ret "/"; }
        case (rem) { ret "%"; }
        case (and) { ret "&&"; }
        case (or) { ret "||"; }
        case (bitxor) { ret "^"; }
        case (bitand) { ret "&"; }
        case (bitor) { ret "|"; }
        case (lsl) { ret "<<"; }
        case (lsr) { ret ">>"; }
        case (asr) { ret ">>>"; }
        case (eq) { ret "=="; }
        case (lt) { ret "<"; }
        case (le) { ret "<="; }
        case (ne) { ret "!="; }
        case (ge) { ret ">="; }
        case (gt) { ret ">"; }
    }
}

pred lazy_binop(binop b) -> bool {
    alt (b) {
        case (and) { true }
        case (or)  { true }
        case (_)   { false }
    }
}

tag unop { box(mutability); deref; not; neg; }

fn unop_to_str(unop op) -> str {
    alt (op) {
        case (box(?mt)) { if (mt == mut) { ret "@mutable "; } ret "@"; }
        case (deref) { ret "*"; }
        case (not) { ret "!"; }
        case (neg) { ret "-"; }
    }
}

tag mode { val; alias(bool); }

type stmt = spanned[stmt_];

tag stmt_ {
    stmt_decl(@decl, node_id);
    stmt_expr(@expr, node_id);

    // These only exist in crate-level blocks.
    stmt_crate_directive(@crate_directive);
}

tag init_op { init_assign; init_recv; init_move; }

type initializer = rec(init_op op, @expr expr);

type local_ =
    rec(option::t[@ty] ty,
        bool infer,
        ident ident,
        option::t[initializer] init,
        node_id id);

type local = spanned[local_];

type decl = spanned[decl_];

tag decl_ { decl_local(@local); decl_item(@item); }

type arm = rec(@pat pat, block block);

type elt = rec(mutability mut, @expr expr);

type field_ = rec(mutability mut, ident ident, @expr expr);

type field = spanned[field_];

tag spawn_dom { dom_implicit; dom_thread; }

tag check_mode { checked; unchecked; }

// FIXME: temporary
tag seq_kind { sk_unique; sk_rc; }

type expr = rec(node_id id,
                expr_ node,
                span span);

tag expr_ {
    expr_vec(vec[@expr], mutability, seq_kind);
    expr_tup(vec[elt]);
    expr_rec(vec[field], option::t[@expr]);
    expr_call(@expr, vec[@expr]);
    expr_self_method(ident);
    expr_bind(@expr, vec[option::t[@expr]]);
    expr_spawn(spawn_dom, option::t[str], @expr, vec[@expr]);
    expr_binary(binop, @expr, @expr);
    expr_unary(unop, @expr);
    expr_lit(@lit);
    expr_cast(@expr, @ty);
    expr_if(@expr, block, option::t[@expr]);
    expr_ternary(@expr, @expr, @expr);
    expr_while(@expr, block);
    expr_for(@local, @expr, block);
    expr_for_each(@local, @expr, block);
    expr_do_while(block, @expr);
    expr_alt(@expr, vec[arm]);
    expr_fn(_fn);
    expr_block(block);
    /*
     * FIXME: many of these @exprs should be constrained with
     * is_lval once we have constrained types working.
     */
    expr_move(@expr, @expr);
    expr_assign(@expr,@expr);
    expr_swap(@expr, @expr);
    expr_assign_op(binop, @expr, @expr);
    expr_send(@expr, @expr);
    expr_recv(@expr, @expr);
    expr_field(@expr, ident);
    expr_index(@expr, @expr);
    expr_path(path);
    expr_ext(path, vec[@expr], option::t[str], @expr);
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
    expr_if_check(@expr, block, option::t[@expr]);
    expr_port(option::t[@ty]);
    expr_chan(@expr);
    expr_anon_obj(anon_obj, vec[ty_param]);
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

fn is_path(&@expr e) -> bool {
    ret alt (e.node) {
        case (expr_path(_)) { true }
        case (_) { false }
    };
}


// NB: If you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
type mt = rec(@ty ty, mutability mut);

type ty_field_ = rec(ident ident, mt mt);

type ty_arg_ = rec(mode mode, @ty ty);

type ty_method_ =
    rec(proto proto,
        ident ident,
        vec[ty_arg] inputs,
        @ty output,
        controlflow cf,
        vec[@constr] constrs);

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
    ty_tup(vec[mt]);
    ty_rec(vec[ty_field]);
    ty_fn(proto, vec[ty_arg], @ty, controlflow, vec[@constr]);
    ty_obj(vec[ty_method]);
    ty_path(path, node_id);
    ty_type;
    ty_constr(@ty, vec[@constr]);
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

type constr_arg = constr_arg_general[uint];

type constr_arg_general[T] = spanned[constr_arg_general_[T]];

type constr_ = rec(path path,
                   vec[@constr_arg_general[uint]] args,
                   node_id id);

type constr = spanned[constr_];


/* The parser generates ast::constrs; resolve generates
 a mapping from each function to a list of ty::constr_defs,
 corresponding to these. */
type arg = rec(mode mode, @ty ty, ident ident, node_id id);

type fn_decl =
    rec(vec[arg] inputs,
        @ty output,
        purity purity,
        controlflow cf,
        vec[@constr] constraints);

tag purity {
    pure_fn; // declared with "pred"

    impure_fn; // declared with "fn"

}

tag controlflow {
    noreturn; // functions with return type _|_ that always
              // raise an error or exit (i.e. never return to the caller)

    return; // everything else

}

type _fn = rec(fn_decl decl, proto proto, block body);

type method_ = rec(ident ident, _fn meth, node_id id);

type method = spanned[method_];

type obj_field = rec(mutability mut, @ty ty, ident ident, node_id id);
type anon_obj_field = rec(mutability mut, @ty ty, @expr expr, ident ident,
                          node_id id);

type _obj =
    rec(vec[obj_field] fields, vec[@method] methods, option::t[@method] dtor);

type anon_obj =
    rec(
        // New fields and methods, if they exist.
        option::t[vec[anon_obj_field]] fields,
        vec[@method] methods,

        // with_obj: the original object being extended, if it exists.
        option::t[@expr] with_obj);

type _mod = rec(vec[@view_item] view_items, vec[@item] items);

tag native_abi {
    native_abi_rust;
    native_abi_cdecl;
    native_abi_llvm;
    native_abi_rust_intrinsic;
}

type native_mod =
    rec(str native_name,
        native_abi abi,
        vec[@view_item] view_items,
        vec[@native_item] items);

type variant_arg = rec(@ty ty, node_id id);

type variant_ = rec(str name, vec[variant_arg] args, node_id id);

type variant = spanned[variant_];

type view_item = spanned[view_item_];

tag view_item_ {
    view_item_use(ident, vec[@meta_item], node_id);
    view_item_import(ident, vec[ident], node_id);
    view_item_import_glob(vec[ident], node_id);
    view_item_export(ident, node_id);
}

type obj_def_ids = rec(node_id ty, node_id ctor);


// Meta-data associated with an item
type attribute = spanned[attribute_];


// Distinguishes between attributes that decorate items and attributes that
// are contained as statements within items. These two cases need to be
// distinguished for pretty-printing.
tag attr_style { attr_outer; attr_inner; }

type attribute_ = rec(attr_style style, meta_item value);

type item = rec(ident ident,
                vec[attribute] attrs,
                node_id id, // For objs and resources, this is the type def_id
                item_ node,
                span span);

tag item_ {
    item_const(@ty, @expr);
    item_fn(_fn, vec[ty_param]);
    item_mod(_mod);
    item_native_mod(native_mod);
    item_ty(@ty, vec[ty_param]);
    item_tag(vec[variant], vec[ty_param]);
    item_obj(_obj, vec[ty_param], node_id /* constructor id */);
    item_res(_fn /* dtor */, node_id /* dtor id */,
             vec[ty_param], node_id /* ctor id */);
}

type native_item = rec(ident ident,
                       vec[attribute] attrs,
                       native_item_ node,
                       node_id id,
                       span span);

tag native_item_ {
    native_item_ty;
    native_item_fn(option::t[str], fn_decl, vec[ty_param]);
}

fn is_exported(ident i, _mod m) -> bool {
    auto nonlocal = true;
    for (@ast::item it in m.items) {
        if (it.ident == i) { nonlocal = false; }
        alt (it.node) {
            case (item_tag(?variants, _)) {
                for (variant v in variants) {
                    if (v.node.name == i) { nonlocal = false; }
                }
            }
            case (_) { }
        }
        if (!nonlocal) { break; }
    }
    auto count = 0u;
    for (@ast::view_item vi in m.view_items) {
        alt (vi.node) {
            case (ast::view_item_export(?id, _)) {
                if (str::eq(i, id)) {
                    // even if it's nonlocal (since it's explicit)

                    ret true;
                }
                count += 1u;
            }
            case (_) {/* fall through */ }
        }
    }
    // If there are no declared exports then 
    // everything not imported is exported

    ret count == 0u && !nonlocal;
}

fn is_call_expr(@expr e) -> bool {
    alt (e.node) {
        case (expr_call(_, _)) { ret true; }
        case (_) { ret false; }
    }
}

fn is_constraint_arg(@expr e) -> bool {
    alt (e.node) {
        case (expr_lit(_)) { ret true; }
        case (expr_path(_)) { ret true; }
        case (_) { ret false; }
    }
}

fn eq_ty(&@ty a, &@ty b) -> bool { ret std::box::ptr_eq(a, b); }

fn hash_ty(&@ty t) -> uint { ret t.span.lo << 16u + t.span.hi; }

fn block_from_expr(@expr e) -> block {
    let block_ blk_ =
        rec(stmts=[],
            expr=option::some[@expr](e),
            id=e.id);
    ret rec(node=blk_, span=e.span);
}

// This is a convenience function to transfor ternary expressions to if
// expressions so that they can be treated the same
fn ternary_to_if(&@expr e) -> @ast::expr {
    alt (e.node) {
        case (expr_ternary(?cond, ?then, ?els)) {
            auto then_blk = block_from_expr(then);
            auto els_blk = block_from_expr(els);
            auto els_expr = @rec(id=els.id, node=expr_block(els_blk),
                                 span=els.span);
            ret @rec(id=e.id,
                     node=expr_if(cond, then_blk, option::some(els_expr)),
                     span=e.span);
        }
        case (_) { fail; }
    }
}

// Path stringification
fn path_to_str(&ast::path pth) -> str {
    auto result = str::connect_ivec(pth.node.idents, "::");
    if (ivec::len[@ast::ty](pth.node.types) > 0u) {
        fn f(&@ast::ty t) -> str { ret print::pprust::ty_to_str(*t); }
        result += "[";
        result += str::connect_ivec(ivec::map(f, pth.node.types), ",");
        result += "]";
    }
    ret result;
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
