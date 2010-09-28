
import std.util.option;
import std.map.hashmap;
import util.common.span;

type ident = str;

type crate = rec(_mod module);

type block = vec[@stmt];

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

tag stmt {
    stmt_block(block);
    stmt_decl(@decl);
    stmt_ret(option[@expr]);
    stmt_log(@expr);
}

tag decl {
    decl_local(ident, option[ty]);
    decl_item(ident, @item);
}

tag expr {
    expr_vec(vec[@expr]);
    expr_tup(vec[@expr]);
    expr_rec(vec[tup(ident,@expr)]);
    expr_call(@expr, vec[@expr]);
    expr_binary(binop, @expr, @expr);
    expr_unary(unop, @expr);
    expr_lit(@lit);
    expr_ident(ident);
    expr_field(@expr, ident);
    expr_index(@expr, @expr);
    expr_cast(@expr, ty);
}

tag lit {
    lit_str(str);
    lit_char(char);
    lit_int(int);
    lit_uint(uint);
    lit_nil;
    lit_bool(bool);
}

tag ty {
    ty_nil;
    ty_bool;
    ty_int;
    ty_uint;
    ty_machine(util.common.ty_mach);
    ty_char;
    ty_str;
    ty_box(@ty);
}

tag mode {
    val;
    alias;
}

type slot = rec(ty ty, mode mode);

type _fn = rec(vec[rec(slot slot, ident ident)] inputs,
               slot output,
               block body);

type _mod = hashmap[ident,item];

tag item {
    item_fn(@_fn);
    item_mod(@_mod);
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
