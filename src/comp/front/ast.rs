
import std.map.hashmap;
import std.option;
import std._vec;
import util.common.span;
import util.common.spanned;
import util.common.ty_mach;
import util.common.filename;

type ident = str;

type path_ = rec(vec[ident] idents, vec[@ty] types);
type path = spanned[path_];

type crate_num = int;
type def_num = int;
type def_id = tup(crate_num, def_num);

type ty_param = rec(ident ident, def_id id);

// Annotations added during successive passes.
tag ann {
    ann_none;
    ann_type(@middle.ty.t);
}

tag def {
    def_fn(def_id);
    def_obj(def_id);
    def_obj_field(def_id);
    def_mod(def_id);
    def_const(def_id);
    def_arg(def_id);
    def_local(def_id);
    def_variant(def_id /* tag */, def_id /* variant */);
    def_ty(def_id);
    def_ty_arg(def_id);
    def_binding(def_id);
    def_use(def_id);
    def_native_ty(def_id);
    def_native_fn(def_id);
}

type crate = spanned[crate_];
type crate_ = rec(_mod module);

tag crate_directive_ {
    cdir_expr(@expr);
    cdir_const(@item);
    cdir_src_mod(ident, option.t[filename]);
    cdir_dir_mod(ident, option.t[filename], vec[@crate_directive]);
    cdir_view_item(@view_item);
    cdir_meta(@meta_item);
    cdir_syntax(path);
    cdir_auth(path, effect);
}
type crate_directive = spanned[crate_directive_];


type meta_item = spanned[meta_item_];
type meta_item_ = rec(ident name, str value);

type block = spanned[block_];
type block_ = rec(vec[@stmt] stmts,
                  option.t[@expr] expr,
                  hashmap[ident,uint] index);

type variant_def = tup(def_id /* tag */, def_id /* variant */);

type pat = spanned[pat_];
tag pat_ {
    pat_wild(ann);
    pat_bind(ident, def_id, ann);
    pat_lit(@lit, ann);
    pat_tag(path, vec[@pat], option.t[variant_def], ann);
}

tag mutability {
    mut;
    imm;
}

tag layer {
    layer_value;
    layer_state;
    layer_gc;
}

tag effect {
    eff_pure;
    eff_impure;
    eff_unsafe;
}

tag proto {
    proto_iter;
    proto_fn;
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
    box;
    deref;
    bitnot;
    not;
    neg;
    _mutable;
}

tag mode {
    val;
    alias;
}

type stmt = spanned[stmt_];
tag stmt_ {
    stmt_decl(@decl);
    stmt_expr(@expr);
}

type local = rec(option.t[@ty] ty,
                 bool infer,
                 ident ident,
                 option.t[@expr] init,
                 def_id id,
                 ann ann);

type decl = spanned[decl_];
tag decl_ {
    decl_local(@local);
    decl_item(@item);
}

type arm = rec(@pat pat, block block, hashmap[ident,def_id] index);

type elt = rec(mutability mut, @expr expr);
type field = rec(mutability mut, ident ident, @expr expr);

type expr = spanned[expr_];
tag expr_ {
    expr_vec(vec[@expr], ann);
    expr_tup(vec[elt], ann);
    expr_rec(vec[field], option.t[@expr], ann);
    expr_call(@expr, vec[@expr], ann);
    expr_bind(@expr, vec[option.t[@expr]], ann);
    expr_binary(binop, @expr, @expr, ann);
    expr_unary(unop, @expr, ann);
    expr_lit(@lit, ann);
    expr_cast(@expr, @ty, ann);
    expr_if(@expr, block, vec[tup(@expr, block)], option.t[block], ann);
    expr_while(@expr, block, ann);
    expr_for(@decl, @expr, block, ann);
    expr_for_each(@decl, @expr, block, ann);
    expr_do_while(block, @expr, ann);
    expr_alt(@expr, vec[arm], ann);
    expr_block(block, ann);
    expr_crate_directive_block(vec[crate_directive_]);
    expr_assign(@expr /* TODO: @expr|is_lval */, @expr, ann);
    expr_assign_op(binop, @expr /* TODO: @expr|is_lval */, @expr, ann);
    expr_field(@expr, ident, ann);
    expr_index(@expr, @expr, ann);
    expr_path(path, option.t[def], ann);
    expr_ext(path, vec[@expr], option.t[@expr], ann);
    expr_fail;
    expr_ret(option.t[@expr]);
    expr_put(option.t[@expr]);
    expr_be(@expr);
    expr_log(@expr);
    expr_check_expr(@expr);
}

type lit = spanned[lit_];
tag lit_ {
    lit_str(str);
    lit_char(char);
    lit_int(int);
    lit_uint(uint);
    lit_mach_int(ty_mach, int);
    lit_nil;
    lit_bool(bool);
}

// NB: If you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.

type ty_field = rec(ident ident, @ty ty);
type ty_arg = rec(mode mode, @ty ty);
// TODO: effect
type ty_method = rec(proto proto, ident ident,
                     vec[ty_arg] inputs, @ty output);
type ty = spanned[ty_];
tag ty_ {
    ty_nil;
    ty_bool;
    ty_int;
    ty_uint;
    ty_machine(util.common.ty_mach);
    ty_char;
    ty_str;
    ty_box(@ty);
    ty_vec(@ty);
    ty_tup(vec[@ty]);
    ty_rec(vec[ty_field]);
    ty_fn(proto, vec[ty_arg], @ty);        // TODO: effect
    ty_obj(vec[ty_method]);
    ty_path(path, option.t[def]);
    ty_mutable(@ty);
    ty_type;
}

type arg = rec(mode mode, @ty ty, ident ident, def_id id);
type fn_decl = rec(effect effect,
                   proto proto,
                   vec[arg] inputs,
                   @ty output);
type _fn = rec(fn_decl decl,
               block body);


type method_ = rec(ident ident, _fn meth, def_id id, ann ann);
type method = spanned[method_];

type obj_field = rec(@ty ty, ident ident, def_id id, ann ann);
type _obj = rec(vec[obj_field] fields,
                vec[@method] methods);

tag mod_index_entry {
    mie_view_item(@view_item);
    mie_item(@item);
    mie_tag_variant(@item /* tag item */, uint /* variant index */);
}

type mod_index = hashmap[ident,mod_index_entry];
type _mod = rec(vec[@view_item] view_items,
                vec[@item] items,
                mod_index index);

tag native_abi {
    native_abi_rust;
    native_abi_cdecl;
}

type native_mod = rec(str native_name,
                      native_abi abi,
                      vec[@native_item] items,
                      native_mod_index index);
type native_mod_index = hashmap[ident,@native_item];

type variant_arg = rec(@ty ty, def_id id);
type variant = rec(str name, vec[variant_arg] args, def_id id, ann ann);

type view_item = spanned[view_item_];
tag view_item_ {
    view_item_use(ident, vec[@meta_item], def_id);
    view_item_import(ident, vec[ident], def_id, option.t[def]);
}

type item = spanned[item_];
tag item_ {
    item_const(ident, @ty, @expr, def_id, ann);
    item_fn(ident, _fn, vec[ty_param], def_id, ann);
    item_mod(ident, _mod, def_id);
    item_native_mod(ident, native_mod, def_id);
    item_ty(ident, @ty, vec[ty_param], def_id, ann);
    item_tag(ident, vec[variant], vec[ty_param], def_id);
    item_obj(ident, _obj, vec[ty_param], def_id, ann);
}

type native_item = spanned[native_item_];
tag native_item_ {
    native_item_ty(ident, def_id);
    native_item_fn(ident, fn_decl, vec[ty_param], def_id, ann);
}

fn index_view_item(mod_index index, @view_item it) {
    alt (it.node) {
        case(ast.view_item_use(?id, _, _)) {
            index.insert(id, ast.mie_view_item(it));
        }
        case(ast.view_item_import(?def_ident,_,_,_)) {
            index.insert(def_ident, ast.mie_view_item(it));
        }
    }
}

fn index_item(mod_index index, @item it) {
    alt (it.node) {
        case (ast.item_const(?id, _, _, _, _)) {
            index.insert(id, ast.mie_item(it));
        }
        case (ast.item_fn(?id, _, _, _, _)) {
            index.insert(id, ast.mie_item(it));
        }
        case (ast.item_mod(?id, _, _)) {
            index.insert(id, ast.mie_item(it));
        }
        case (ast.item_native_mod(?id, _, _)) {
            index.insert(id, ast.mie_item(it));
        }
        case (ast.item_ty(?id, _, _, _, _)) {
            index.insert(id, ast.mie_item(it));
        }
        case (ast.item_tag(?id, ?variants, _, _)) {
            index.insert(id, ast.mie_item(it));
            let uint variant_idx = 0u;
            for (ast.variant v in variants) {
                index.insert(v.name,
                             ast.mie_tag_variant(it, variant_idx));
                variant_idx += 1u;
            }
        }
        case (ast.item_obj(?id, _, _, _, _)) {
            index.insert(id, ast.mie_item(it));
        }
    }
}

fn index_native_item(native_mod_index index, @native_item it) {
    alt (it.node) {
        case (ast.native_item_ty(?id, _)) {
            index.insert(id, it);
        }
        case (ast.native_item_fn(?id, _, _, _, _)) {
            index.insert(id, it);
        }
    }
}

fn is_call_expr(@expr e) -> bool {
    alt (e.node) {
        case (expr_call(_, _, _)) {
            ret true;
        }
        case (_) {
            ret false;
        }
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
