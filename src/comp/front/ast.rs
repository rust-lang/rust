
import std.map.hashmap;
import std.option;
import std._str;
import std._vec;
import util.common.span;
import util.common.spanned;
import util.common.ty_mach;
import util.common.filename;
import util.typestate_ann.ts_ann;

type ident = str;

type path_ = rec(vec[ident] idents, vec[@ty] types);
type path = spanned[path_];

type crate_num = int;
type def_num = int;
type def_id = tup(crate_num, def_num);

type ty_param = ident;

// Annotations added during successive passes.
tag ann {
    ann_none;
    ann_type(middle.ty.t,
             option.t[vec[middle.ty.t]], /* ty param substs */
             option.t[@ts_ann]); /* pre- and postcondition for typestate */
}

tag def {
    def_fn(def_id);
    def_obj(def_id);
    def_obj_field(def_id);
    def_mod(def_id);
    def_native_mod(def_id);
    def_const(def_id);
    def_arg(def_id);
    def_local(def_id);
    def_upvar(def_id);
    def_variant(def_id /* tag */, def_id /* variant */);
    def_ty(def_id);
    def_ty_arg(uint);
    def_binding(def_id);
    def_use(def_id);
    def_native_ty(def_id);
    def_native_fn(def_id);
}

fn def_id_of_def(def d) -> def_id {
    alt (d) {
        case (def_fn(?id)) { ret id; }
        case (def_obj(?id)) { ret id; }
        case (def_obj_field(?id)) { ret id; }
        case (def_mod(?id)) { ret id; }
        case (def_native_mod(?id)) { ret id; }
        case (def_const(?id)) { ret id; }
        case (def_arg(?id)) { ret id; }
        case (def_local(?id)) { ret id; }
        case (def_upvar(?id)) { ret id; }
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

type crate = spanned[crate_];
type crate_ = rec(vec[@crate_directive] directives,
                  _mod module);

tag crate_directive_ {
    cdir_expr(@expr);
    // FIXME: cdir_let should be eliminated
    // and redirected to the use of const stmt_decls inside
    // crate directive blocks.
    cdir_let(ident, @expr, vec[@crate_directive]);
    cdir_src_mod(ident, option.t[filename]);
    cdir_dir_mod(ident, option.t[filename], vec[@crate_directive]);
    cdir_view_item(@view_item);
    cdir_meta(vec[@meta_item]);
    cdir_syntax(path);
    cdir_auth(path, _auth);
}
type crate_directive = spanned[crate_directive_];


type meta_item = spanned[meta_item_];
type meta_item_ = rec(ident name, str value);

type block = spanned[block_];
type block_index = hashmap[ident, block_index_entry];
tag block_index_entry {
    bie_item(@item);
    bie_local(@local);
    bie_tag_variant(@item /* tag item */, uint /* variant index */);
}
type block_ = rec(vec[@stmt] stmts,
                  option.t[@expr] expr,
                  hashmap[ident,block_index_entry] index,
                  ann a); /* ann is only meaningful for the ts_ann field */

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
    maybe_mut;
}

tag layer {
    layer_value;
    layer_state;
    layer_gc;
}

tag _auth {
    auth_unsafe;
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

fn binop_to_str(binop op) -> str {
    alt (op) {
        case (add) {ret "+";}
        case (sub) {ret "-";}
        case (mul) {ret "*";}
        case (div) {ret "/";}
        case (rem) {ret "%";}
        case (and) {ret "&&";}
        case (or) {ret "||";}
        case (bitxor) {ret "^";}
        case (bitand) {ret "&";}
        case (bitor) {ret "|";}
        case (lsl) {ret "<<";}
        case (lsr) {ret ">>";}
        case (asr) {ret ">>>";}
        case (eq) {ret "==";}
        case (lt) {ret "<";}
        case (le) {ret "<=";}
        case (ne) {ret "!=";}
        case (ge) {ret ">=";}
        case (gt) {ret ">";}
    }
}


tag unop {
    box(mutability);
    deref;
    bitnot;
    not;
    neg;
}

fn unop_to_str(unop op) -> str {
    alt (op) {
        case (box(?mt)) {
            if (mt == mut) { ret "@mutable"; }
            ret "@";
        }
        case (deref) {ret "*";}
        case (bitnot) {ret "~";}
        case (not) {ret "!";}
        case (neg) {ret "-";}
    }
}

tag mode {
    val;
    alias;
}

type stmt = spanned[stmt_];
tag stmt_ {
/* Only the ts_ann field is meaningful for statements,
   but we make it an ann to make traversals simpler */
    stmt_decl(@decl, ann); 
    stmt_expr(@expr, ann);
    // These only exist in crate-level blocks.
    stmt_crate_directive(@crate_directive);
}

tag init_op {
    init_assign;
    init_recv;
}

type initializer = rec(init_op op,
                       @expr expr);

type local = rec(option.t[@ty] ty,
                 bool infer,
                 ident ident,
                 option.t[initializer] init,
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

tag spawn_dom {
    dom_implicit;
    dom_thread;
}

type expr = spanned[expr_];
tag expr_ {
    expr_vec(vec[@expr], mutability, ann);
    expr_tup(vec[elt], ann);
    expr_rec(vec[field], option.t[@expr], ann);
    expr_call(@expr, vec[@expr], ann);
    expr_self_method(ident, ann);
    expr_bind(@expr, vec[option.t[@expr]], ann);
    expr_spawn(spawn_dom, option.t[str], @expr, vec[@expr], ann);
    expr_binary(binop, @expr, @expr, ann);
    expr_unary(unop, @expr, ann);
    expr_lit(@lit, ann);
    expr_cast(@expr, @ty, ann);
    expr_if(@expr, block, option.t[@expr], ann);
    expr_while(@expr, block, ann);
    expr_for(@decl, @expr, block, ann);
    expr_for_each(@decl, @expr, block, ann);
    expr_do_while(block, @expr, ann);
    expr_alt(@expr, vec[arm], ann);
    expr_block(block, ann);
    expr_assign(@expr /* TODO: @expr|is_lval */, @expr, ann);
    expr_assign_op(binop, @expr /* TODO: @expr|is_lval */, @expr, ann);
    expr_send(@expr /* TODO: @expr|is_lval */, @expr, ann);
    expr_recv(@expr /* TODO: @expr|is_lval */, @expr, ann);
    expr_field(@expr, ident, ann);
    expr_index(@expr, @expr, ann);
    expr_path(path, option.t[def], ann);
    expr_ext(path, vec[@expr], option.t[str], @expr, ann);
    expr_fail(ann);
    expr_break(ann);
    expr_cont(ann);
    expr_ret(option.t[@expr], ann);
    expr_put(option.t[@expr], ann);
    expr_be(@expr, ann);
    expr_log(int, @expr, ann);
/* just an assert, no significance to typestate */
    expr_assert(@expr, ann);
/* preds that typestate is aware of */
    expr_check(@expr, ann);
    expr_port(ann);
    expr_chan(@expr, ann);
}

type lit = spanned[lit_];
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

type mt = rec(@ty ty, mutability mut);
type ty_field = rec(ident ident, mt mt);
type ty_arg = rec(mode mode, @ty ty);
type ty_method = rec(proto proto, ident ident,
                     vec[ty_arg] inputs, @ty output);
type ty = spanned[ty_];
tag ty_ {
    ty_nil;
    ty_bool;
    ty_int;
    ty_uint;
    ty_float;
    ty_machine(util.common.ty_mach);
    ty_char;
    ty_str;
    ty_box(mt);
    ty_vec(mt);
    ty_port(@ty);
    ty_chan(@ty);
    ty_tup(vec[mt]);
    ty_rec(vec[ty_field]);
    ty_fn(proto, vec[ty_arg], @ty);
    ty_obj(vec[ty_method]);
    ty_path(path, option.t[def]);
    ty_type;
    ty_constr(@ty, vec[@constr]);
}

tag constr_arg_ {
    carg_base;
    carg_ident(ident);
}
type constr_arg = spanned[constr_arg_];
type constr_ = rec(path path, vec[@constr_arg] args);
type constr = spanned[constr_];

type arg = rec(mode mode, @ty ty, ident ident, def_id id);
type fn_decl = rec(vec[arg] inputs,
                   @ty output,
                   purity purity);
tag purity {
    pure_fn;   // declared with "pred"
    impure_fn; // declared with "fn"
}

type _fn = rec(fn_decl decl,
               proto proto,
               block body);


type method_ = rec(ident ident, _fn meth, def_id id, ann ann);
type method = spanned[method_];

type obj_field = rec(@ty ty, ident ident, def_id id, ann ann);
type _obj = rec(vec[obj_field] fields,
                vec[@method] methods,
                option.t[@method] dtor);

tag mod_index_entry {
    mie_view_item(@view_item);
    mie_item(@item);
    mie_tag_variant(@item /* tag item */, uint /* variant index */);
}

tag native_mod_index_entry {
    nmie_view_item(@view_item);
    nmie_item(@native_item);
}

type mod_index = hashmap[ident,mod_index_entry];
type _mod = rec(vec[@view_item] view_items,
                vec[@item] items,
                mod_index index);

tag native_abi {
    native_abi_rust;
    native_abi_cdecl;
    native_abi_llvm;
    native_abi_rust_intrinsic;
}

type native_mod = rec(str native_name,
                      native_abi abi,
                      vec[@view_item] view_items,
                      vec[@native_item] items,
                      native_mod_index index);
type native_mod_index = hashmap[ident,native_mod_index_entry];

type variant_arg = rec(@ty ty, def_id id);
type variant_ = rec(str name, vec[variant_arg] args, def_id id, ann ann);
type variant = spanned[variant_];

type view_item = spanned[view_item_];
tag view_item_ {
    view_item_use(ident, vec[@meta_item], def_id, option.t[int]);
    view_item_import(ident, vec[ident], def_id, option.t[def]);
    view_item_export(ident);
}

type obj_def_ids = rec(def_id ty, def_id ctor);

type item = spanned[item_];
tag item_ {
    item_const(ident, @ty, @expr, def_id, ann);
    item_fn(ident, _fn, vec[ty_param], def_id, ann);
    item_mod(ident, _mod, def_id);
    item_native_mod(ident, native_mod, def_id);
    item_ty(ident, @ty, vec[ty_param], def_id, ann);
    item_tag(ident, vec[variant], vec[ty_param], def_id, ann);
    item_obj(ident, _obj, vec[ty_param], obj_def_ids, ann);
}

type native_item = spanned[native_item_];
tag native_item_ {
    native_item_ty(ident, def_id);
    native_item_fn(ident, option.t[str],
                   fn_decl, vec[ty_param], def_id, ann);
}

fn index_view_item(mod_index index, @view_item it) {
    alt (it.node) {
        case(ast.view_item_use(?id, _, _, _)) {
            index.insert(id, ast.mie_view_item(it));
        }
        case(ast.view_item_import(?def_ident,_,_,_)) {
            index.insert(def_ident, ast.mie_view_item(it));
        }
        case(ast.view_item_export(_)) {
            // NB: don't index these, they might collide with
            // the import or use that they're exporting. Have
            // to do linear search for exports.
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
        case (ast.item_tag(?id, ?variants, _, _, _)) {
            index.insert(id, ast.mie_item(it));
            let uint variant_idx = 0u;
            for (ast.variant v in variants) {
                index.insert(v.node.name,
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
            index.insert(id, ast.nmie_item(it));
        }
        case (ast.native_item_fn(?id, _, _, _, _, _)) {
            index.insert(id, ast.nmie_item(it));
        }
    }
}

fn index_native_view_item(native_mod_index index, @view_item it) {
    alt (it.node) {
        case(ast.view_item_import(?def_ident,_,_,_)) {
            index.insert(def_ident, ast.nmie_view_item(it));
        }
        case(ast.view_item_export(_)) {
            // NB: don't index these, they might collide with
            // the import or use that they're exporting. Have
            // to do linear search for exports.
        }
    }
}

fn index_stmt(block_index index, @stmt s) {
    alt (s.node) {
        case (ast.stmt_decl(?d,_)) {
            alt (d.node) {
                case (ast.decl_local(?loc)) {
                    index.insert(loc.ident, ast.bie_local(loc));
                }
                case (ast.decl_item(?it)) {
                    alt (it.node) {
                        case (ast.item_fn(?i, _, _, _, _)) {
                            index.insert(i, ast.bie_item(it));
                        }
                        case (ast.item_mod(?i, _, _)) {
                            index.insert(i, ast.bie_item(it));
                        }
                        case (ast.item_ty(?i, _, _, _, _)) {
                            index.insert(i, ast.bie_item(it));
                        }
                        case (ast.item_tag(?i, ?variants, _, _, _)) {
                            index.insert(i, ast.bie_item(it));
                            let uint vid = 0u;
                            for (ast.variant v in variants) {
                                auto t = ast.bie_tag_variant(it, vid);
                                index.insert(v.node.name, t);
                                vid += 1u;
                            }
                        }
                        case (ast.item_obj(?i, _, _, _, _)) {
                            index.insert(i, ast.bie_item(it));
                        }
                    }
                }
            }
        }
        case (_) { /* fall through */ }
    }
}

fn is_exported(ident i, _mod m) -> bool {
    auto count = 0;
    for (@ast.view_item vi in m.view_items) {
        alt (vi.node) {
            case (ast.view_item_export(?id)) {
                if (_str.eq(i, id)) {
                    ret true;
                }
                count += 1;
            }
            case (_) { /* fall through */ }
        }
    }
    // If there are no declared exports then everything is exported
    if (count == 0) {
        ret true;
    } else {
        ret false;
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

fn is_constraint_arg(@expr e) -> bool {
    alt (e.node) {
        case (expr_lit(_,_)) {
            ret true;
        }
        case (expr_path(_, option.some[def](def_local(_)), _)) {
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
