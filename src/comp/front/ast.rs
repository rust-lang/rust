
import std.map.hashmap;
import std.option;
import middle.typeck;
import util.common.span;
import util.common.spanned;
import util.common.ty_mach;

type ident = str;

type name_ = rec(ident ident, vec[@ty] types);
type name = spanned[name_];
type path = vec[name];

type crate_num = int;
type def_num = int;
type def_id = tup(crate_num, def_num);

// Annotations added during successive passes.
tag ann {
    ann_none;
    ann_type(@typeck.ty);
}

tag def {
    def_fn(def_id);
    def_mod(def_id);
    def_const(def_id);
    def_arg(def_id);
    def_local(def_id);
    def_ty(def_id);
    def_ty_arg(def_id);
}

type crate = spanned[crate_];
type crate_ = rec(_mod module);

type block = spanned[block_];
type block_ = rec(vec[@stmt] stmts,
                  hashmap[ident,uint] index);

type pat = spanned[pat_];
tag pat_ {
    pat_wild;
    pat_bind(ident);
    pat_tag(ident, vec[@pat]);
}

type arm = rec(@pat pat, block block);

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
    stmt_ret(option.t[@expr]);
    stmt_log(@expr);
    stmt_check_expr(@expr);
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

type expr = spanned[expr_];
tag expr_ {
    expr_vec(vec[@expr], ann);
    expr_tup(vec[tup(bool /* mutability */, @expr)], ann);
    expr_rec(vec[tup(ident,@expr)], ann);
    expr_call(@expr, vec[@expr], ann);
    expr_binary(binop, @expr, @expr, ann);
    expr_unary(unop, @expr, ann);
    expr_lit(@lit, ann);
    expr_cast(@expr, @ty, ann);
    expr_if(@expr, block, option.t[block], ann);
    expr_while(@expr, block, ann);
    expr_do_while(block, @expr, ann);
    expr_alt(@expr, vec[arm], ann);
    expr_block(block, ann);
    expr_assign(@expr /* TODO: @expr|is_lval */, @expr, ann);
    expr_field(@expr, ident, ann);
    expr_index(@expr, @expr, ann);
    expr_name(name, option.t[def], ann);
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
// type structure in middle/typeck.rs as well.
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
    ty_tup(vec[tup(bool /* mutability */, @ty)]);
    ty_fn(vec[rec(mode mode, @ty ty)], @ty);        // TODO: effect
    ty_path(path, option.t[def]);
}

tag mode {
    val;
    alias;
}

type arg = rec(mode mode, @ty ty, ident ident, def_id id);
type _fn = rec(vec[arg] inputs,
               @ty output,
               block body);

type _mod = rec(vec[@item] items,
                hashmap[ident,uint] index);

type variant = rec(str name, vec[@ty] args);

type item = spanned[item_];
tag item_ {
    item_fn(ident, _fn, def_id, ann);
    item_mod(ident, _mod, def_id);
    item_ty(ident, @ty, def_id, ann);
    item_tag(ident, vec[variant], def_id);
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
