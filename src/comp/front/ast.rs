
import util.common.option;
import std.map.hashmap;
import util.common.span;
import util.common.spanned;
import util.common.option;
import util.common.some;
import util.common.none;

type ident = str;

type name_ = rec(ident ident, vec[@ty] types);
type name = spanned[name_];
type path = vec[name];

type crate_num = int;
type slot_num = int;
type item_num = int;

tag slot_id {
    id_slot(crate_num, slot_num);
}

tag item_id {
    id_item(crate_num, slot_num);
}

tag referent {
    ref_slot(slot_id);
    ref_item(item_id);
}

type crate = spanned[crate_];
type crate_ = rec(_mod module);

type block = spanned[block_];
type block_ = vec[@stmt];

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
}

type stmt = spanned[stmt_];
tag stmt_ {
    stmt_decl(@decl);
    stmt_ret(option[@expr]);
    stmt_log(@expr);
    stmt_expr(@expr);
}

type decl = spanned[decl_];
tag decl_ {
    decl_local(ident, option[@ty], option[@expr]);
    decl_item(name, @item);
}

type expr = spanned[expr_];
tag expr_ {
    expr_vec(vec[@expr]);
    expr_tup(vec[@expr]);
    expr_rec(vec[tup(ident,@expr)]);
    expr_call(@expr, vec[@expr]);
    expr_binary(binop, @expr, @expr);
    expr_unary(unop, @expr);
    expr_lit(@lit);
    expr_name(name, option[referent]);
    expr_field(@expr, ident);
    expr_index(@expr, @expr);
    expr_cast(@expr, @ty);
    expr_if(@expr, block, option[block]);
    expr_block(block);
}

type lit = spanned[lit_];
tag lit_ {
    lit_str(str);
    lit_char(char);
    lit_int(int);
    lit_uint(uint);
    lit_nil;
    lit_bool(bool);
}

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
    ty_path(path, option[referent]);
}

tag mode {
    val;
    alias;
}

type slot = rec(@ty ty, mode mode, option[slot_id] id);
type input = rec(slot slot, ident ident);

type _fn = rec(vec[input] inputs,
               ty output,
               block body);

type _mod = hashmap[ident,@item];

type item = spanned[item_];
tag item_ {
    item_fn(_fn, item_id);
    item_mod(_mod);
    item_ty(@ty, item_id);
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
