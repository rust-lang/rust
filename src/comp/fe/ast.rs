
import std.util.option;
import std.map.hashmap;
import util.common.span;

type ident = str;

type crate = rec( str filename,
                  _mod module);

type block = vec[@stmt];

tag stmt {
    stmt_block(block);
    stmt_decl(@decl);
    stmt_ret(option[@lval]);
}


tag decl {
    decl_local(ident, option[ty]);
    decl_item(ident, @item);
}

tag lval {
    lval_ident(ident);
    lval_ext(@lval, ident);
    lval_idx(@lval, @atom);
}

tag atom {
    atom_lit(@lit);
    atom_lval(@lval);
}

tag lit {
    lit_char(char);
    lit_int(int);
    lit_nil;
    lit_bool(bool);
}

tag ty {
    ty_nil;
    ty_bool;
    ty_int;
    ty_char;
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
