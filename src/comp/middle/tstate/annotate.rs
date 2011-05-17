import std::_vec;
import std::option;
import std::option::some;
import std::option::none;

import front::ast;

import front::ast::ident;
import front::ast::def_id;
import front::ast::ann;
import front::ast::item;
import front::ast::_fn;
import front::ast::_mod;
import front::ast::crate;
import front::ast::_obj;
import front::ast::ty_param;
import front::ast::item_fn;
import front::ast::item_obj;
import front::ast::item_ty;
import front::ast::item_tag;
import front::ast::item_const;
import front::ast::item_mod;
import front::ast::item_native_mod;
import front::ast::expr;
import front::ast::elt;
import front::ast::field;
import front::ast::decl;
import front::ast::decl_local;
import front::ast::decl_item;
import front::ast::initializer;
import front::ast::local;
import front::ast::arm;
import front::ast::expr_call;
import front::ast::expr_vec;
import front::ast::expr_tup;
import front::ast::expr_path;
import front::ast::expr_field;
import front::ast::expr_index;
import front::ast::expr_log;
import front::ast::expr_block;
import front::ast::expr_rec;
import front::ast::expr_if;
import front::ast::expr_binary;
import front::ast::expr_unary;
import front::ast::expr_assign;
import front::ast::expr_assign_op;
import front::ast::expr_while;
import front::ast::expr_do_while;
import front::ast::expr_alt;
import front::ast::expr_lit;
import front::ast::expr_ret;
import front::ast::expr_self_method;
import front::ast::expr_bind;
import front::ast::expr_spawn;
import front::ast::expr_ext;
import front::ast::expr_fail;
import front::ast::expr_break;
import front::ast::expr_cont;
import front::ast::expr_send;
import front::ast::expr_recv;
import front::ast::expr_put;
import front::ast::expr_port;
import front::ast::expr_chan;
import front::ast::expr_be;
import front::ast::expr_check;
import front::ast::expr_assert;
import front::ast::expr_cast;
import front::ast::expr_for;
import front::ast::expr_for_each;
import front::ast::stmt;
import front::ast::stmt_decl;
import front::ast::stmt_expr;
import front::ast::block;
import front::ast::block_;
import front::ast::method;

import middle::fold;
import middle::fold::respan;
import middle::fold::new_identity_fold;
import middle::fold::fold_crate;
import middle::fold::fold_item;
import middle::fold::fold_method;

import util::common::uistr;
import util::common::span;
import util::common::new_str_hash;

import middle::tstate::aux::fn_info;
import middle::tstate::aux::fn_info_map;
import middle::tstate::aux::num_locals;
import middle::tstate::aux::init_ann;
import middle::tstate::aux::init_blank_ann;
import middle::tstate::aux::get_fn_info;

fn item_fn_anns(&fn_info_map fm, &span sp, ident i, &_fn f,
                vec[ty_param] ty_params, def_id id, ann a) -> @item {

    assert (fm.contains_key(id));
    auto f_info = fm.get(id);

    log(i + " has " + uistr(num_locals(f_info)) + " local vars");

    auto fld0 = new_identity_fold[fn_info]();

    fld0 = @rec(fold_ann = bind init_ann(_,_) 
                    with *fld0);

    ret fold_item[fn_info]
           (f_info, fld0, @respan(sp, item_fn(i, f, ty_params, id, a))); 
}

/* FIXME: rewrite this with walk instead of fold */

/* This is painstakingly written as an explicit recursion b/c the
   standard ast.fold doesn't traverse in the correct order:
   consider
   fn foo() {
      fn bar() {
        auto x = 5;
        log(x);
      }
   }
   With fold, first bar() would be processed and its subexps would
   correctly be annotated with length-1 bit vectors.
   But then, the process would be repeated with (fn bar()...) as
   a subexp of foo, which has 0 local variables -- so then
   the body of bar() would be incorrectly annotated with length-0 bit
   vectors. */
fn annotate_exprs(&fn_info_map fm, &vec[@expr] es) -> vec[@expr] {
    fn one(fn_info_map fm, &@expr e) -> @expr {
        ret annotate_expr(fm, e);
    }
    auto f = bind one(fm,_);
    ret _vec::map[@expr, @expr](f, es);
}
fn annotate_elts(&fn_info_map fm, &vec[elt] es) -> vec[elt] {
    fn one(fn_info_map fm, &elt e) -> elt {
        ret rec(mut=e.mut,
                expr=annotate_expr(fm, e.expr));
    }
    auto f = bind one(fm,_);
    ret _vec::map[elt, elt](f, es);
}
fn annotate_fields(&fn_info_map fm, &vec[field] fs) -> vec[field] {
    fn one(fn_info_map fm, &field f) -> field {
        ret rec(mut=f.mut,
                 ident=f.ident,
                 expr=annotate_expr(fm, f.expr));
    }
    auto f = bind one(fm,_);
    ret _vec::map[field, field](f, fs);
}
fn annotate_option_exp(&fn_info_map fm, &option::t[@expr] o)
  -> option::t[@expr] {
    fn one(fn_info_map fm, &@expr e) -> @expr {
        ret annotate_expr(fm, e);
    }
    auto f = bind one(fm,_);
    ret option::map[@expr, @expr](f, o);
}
fn annotate_option_exprs(&fn_info_map fm, &vec[option::t[@expr]] es)
  -> vec[option::t[@expr]] {
    fn one(fn_info_map fm, &option::t[@expr] o) -> option::t[@expr] {
        ret annotate_option_exp(fm, o);
    }
    auto f = bind one(fm,_);
    ret _vec::map[option::t[@expr], option::t[@expr]](f, es);
}
fn annotate_decl(&fn_info_map fm, &@decl d) -> @decl {
    auto d1 = d.node;
    alt (d.node) {
        case (decl_local(?l)) {
            alt(l.init) {
                case (some[initializer](?init)) {
                    let option::t[initializer] an_i =
                        some[initializer]
                          (rec(expr=annotate_expr(fm, init.expr)
                                 with init));
                    let @local new_l = @rec(init=an_i with *l);
                    d1 = decl_local(new_l);
                }
                case (_) { /* do nothing */ }
            }
        }
        case (decl_item(?item)) {
            d1 = decl_item(annotate_item(fm, item));
        }
    }
    ret @respan(d.span, d1);
}
fn annotate_alts(&fn_info_map fm, &vec[arm] alts) -> vec[arm] {
    fn one(fn_info_map fm, &arm a) -> arm {
        ret rec(pat=a.pat,
                 block=annotate_block(fm, a.block));
    }
    auto f = bind one(fm,_);
    ret _vec::map[arm, arm](f, alts);

}
fn annotate_expr(&fn_info_map fm, &@expr e) -> @expr {
    auto e1 = e.node;
    alt (e.node) {
        case (expr_vec(?es, ?m, ?a)) {
            e1 = expr_vec(annotate_exprs(fm, es), m, a);
        }
        case (expr_tup(?es, ?a)) {
            e1 = expr_tup(annotate_elts(fm, es), a);
        }
        case (expr_rec(?fs, ?maybe_e, ?a)) {
            e1 = expr_rec(annotate_fields(fm, fs),
                          annotate_option_exp(fm, maybe_e), a);
        }
        case (expr_call(?e, ?es, ?a)) {
            e1 = expr_call(annotate_expr(fm, e),
                          annotate_exprs(fm, es), a);
        }
        case (expr_self_method(_,_)) {
            // no change
        }
        case (expr_bind(?e, ?maybe_es, ?a)) {
            e1 = expr_bind(annotate_expr(fm, e),
                           annotate_option_exprs(fm, maybe_es),
                           a);
        }
        case (expr_spawn(?s, ?maybe_s, ?e, ?es, ?a)) {
            e1 = expr_spawn(s, maybe_s, annotate_expr(fm, e),
                            annotate_exprs(fm, es), a);
        }
        case (expr_binary(?bop, ?w, ?x, ?a)) {
            e1 = expr_binary(bop, annotate_expr(fm, w),
                             annotate_expr(fm, x), a);
        }
        case (expr_unary(?uop, ?w, ?a)) {
            e1 = expr_unary(uop, annotate_expr(fm, w), a);
        }
        case (expr_lit(_,_)) {
            /* no change */
        }
        case (expr_cast(?e,?t,?a)) {
            e1 = expr_cast(annotate_expr(fm, e), t, a);
        }
        case (expr_if(?e, ?b, ?maybe_e, ?a)) {
            e1 = expr_if(annotate_expr(fm, e),
                         annotate_block(fm, b),
                         annotate_option_exp(fm, maybe_e), a);
        }
        case (expr_while(?e, ?b, ?a)) {
            e1 = expr_while(annotate_expr(fm, e),
                            annotate_block(fm, b), a);
        }
        case (expr_for(?d, ?e, ?b, ?a)) {
            e1 = expr_for(annotate_decl(fm, d),
                          annotate_expr(fm, e),
                          annotate_block(fm, b), a);
        }
        case (expr_for_each(?d, ?e, ?b, ?a)) {
            e1 = expr_for_each(annotate_decl(fm, d),
                          annotate_expr(fm, e),
                          annotate_block(fm, b), a);
        }
        case (expr_do_while(?b, ?e, ?a)) {
            e1 = expr_do_while(annotate_block(fm, b),
                               annotate_expr(fm, e), a);
        }
        case (expr_alt(?e, ?alts, ?a)) {
            e1 = expr_alt(annotate_expr(fm, e),
                          annotate_alts(fm, alts), a);
        }
        case (expr_block(?b, ?a)) {
            e1 = expr_block(annotate_block(fm, b), a);
        }
        case (expr_assign(?l, ?r, ?a)) {
            e1 = expr_assign(annotate_expr(fm, l), annotate_expr(fm, r), a);
        }
        case (expr_assign_op(?bop, ?l, ?r, ?a)) {
            e1 = expr_assign_op(bop,
               annotate_expr(fm, l), annotate_expr(fm, r), a);
        }
        case (expr_send(?l, ?r, ?a)) {
            e1 = expr_send(annotate_expr(fm, l),
                           annotate_expr(fm, r), a);
        }
        case (expr_recv(?l, ?r, ?a)) {
           e1 = expr_recv(annotate_expr(fm, l),
                           annotate_expr(fm, r), a);
        }
        case (expr_field(?e, ?i, ?a)) {
            e1 = expr_field(annotate_expr(fm, e),
                            i, a);
        }
        case (expr_index(?e, ?sub, ?a)) {
            e1 = expr_index(annotate_expr(fm, e),
                            annotate_expr(fm, sub), a);
        }
        case (expr_path(_,_)) {
            /* no change */
        }
        case (expr_ext(?p, ?es, ?s_opt, ?e, ?a)) {
            e1 = expr_ext(p, annotate_exprs(fm, es),
                          s_opt,
                          annotate_expr(fm, e), a);
        }
        /* no change, next 3 cases */
        case (expr_fail(_)) { }
        case (expr_break(_)) { }
        case (expr_cont(_)) { }
        case (expr_ret(?maybe_e, ?a)) {
            e1 = expr_ret(annotate_option_exp(fm, maybe_e), a);
        }
        case (expr_put(?maybe_e, ?a)) {
            e1 = expr_put(annotate_option_exp(fm, maybe_e), a);
        }
        case (expr_be(?e, ?a)) {
            e1 = expr_be(annotate_expr(fm, e), a);
        }
        case (expr_log(?n, ?e, ?a)) {
            e1 = expr_log(n, annotate_expr(fm, e), a);
        }
        case (expr_assert(?e, ?a)) {
            e1 = expr_assert(annotate_expr(fm, e), a);
        }
        case (expr_check(?e, ?a)) {
            e1 = expr_check(annotate_expr(fm, e), a);
        }
        case (expr_port(_)) { /* no change */ }
        case (expr_chan(?e, ?a)) {
            e1 = expr_chan(annotate_expr(fm, e), a);
        }
    }
    ret @respan(e.span, e1);
}

fn annotate_stmt(&fn_info_map fm, &@stmt s) -> @stmt {
    alt (s.node) {
        case (stmt_decl(?d, ?a)) {
            ret @respan(s.span, stmt_decl(annotate_decl(fm, d), a));
        }
        case (stmt_expr(?e, ?a)) {
            ret @respan(s.span, stmt_expr(annotate_expr(fm, e), a));
        }
    }
}
fn annotate_block(&fn_info_map fm, &block b) -> block {
    let vec[@stmt] new_stmts = [];
   
    for (@stmt s in b.node.stmts) {
        auto new_s = annotate_stmt(fm, s);
        _vec::push[@stmt](new_stmts, new_s);
    }
    fn ann_e(fn_info_map fm, &@expr e) -> @expr {
        ret annotate_expr(fm, e);
    }
    auto f = bind ann_e(fm,_);

    auto new_e = option::map[@expr, @expr](f, b.node.expr);

    ret respan(b.span,
          rec(stmts=new_stmts, expr=new_e with b.node));
}
fn annotate_fn(&fn_info_map fm, &_fn f) -> _fn {
    // subexps have *already* been annotated based on
    // f's number-of-locals
    ret rec(body=annotate_block(fm, f.body) with f);
}
fn annotate_mod(&fn_info_map fm, &_mod m) -> _mod {
    let vec[@item] new_items = [];
   
    for (@item i in m.items) {
        auto new_i = annotate_item(fm, i);
        _vec::push[@item](new_items, new_i);
    }
    ret rec(items=new_items with m);
}
fn annotate_method(&fn_info_map fm, &@method m) -> @method {
    auto f_info = get_fn_info(fm, m.node.id);
    auto fld0 = new_identity_fold[fn_info]();
    fld0 = @rec(fold_ann = bind init_ann(_,_) 
                with *fld0);
    auto outer = fold_method[fn_info](f_info, fld0, m);
    auto new_fn = annotate_fn(fm, outer.node.meth);
    ret @respan(m.span,
                rec(meth=new_fn with m.node));
}

fn annotate_obj(&fn_info_map fm, &_obj o) -> _obj {
    fn one(fn_info_map fm, &@method m) -> @method {
        ret annotate_method(fm, m);
    }
    auto f = bind one(fm,_);
    auto new_methods = _vec::map[@method, @method](f, o.methods);
    auto new_dtor    = option::map[@method, @method](f, o.dtor);
    ret rec(methods=new_methods, dtor=new_dtor with o);
}

 
// Only annotates the components of the item recursively.
fn annotate_item_inner(&fn_info_map fm, &@item item) -> @item {
    alt (item.node) {
        /* FIXME can't skip this case -- exprs contain blocks contain stmts,
         which contain decls */
        case (item_const(_,_,_,_,_)) {
            // this has already been annotated by annotate_item
            ret item;
        }
        case (item_fn(?ident, ?ff, ?tps, ?id, ?ann)) {
            ret @respan(item.span,
                       item_fn(ident, annotate_fn(fm, ff), tps, id, ann));
        }
        case (item_mod(?ident, ?mm, ?id)) {
            ret @respan(item.span,
                       item_mod(ident, annotate_mod(fm, mm), id));
        }
        case (item_native_mod(?ident, ?mm, ?id)) {
            ret item;
        }
        case (item_ty(_,_,_,_,_)) {
            ret item;
        }
        case (item_tag(_,_,_,_,_)) {
            ret item;
        }
        case (item_obj(?ident, ?ob, ?tps, ?odid, ?ann)) {
            ret @respan(item.span,
              item_obj(ident, annotate_obj(fm, ob), tps, odid, ann));
        }
    } 
}

fn annotate_item(&fn_info_map fm, &@item item) -> @item {
    // Using a fold, recursively set all anns in this item
    // to be blank.
    // *Then*, call annotate_item recursively to do the right
    // thing for any nested items inside this one.
    
    alt (item.node) {
        case (item_const(_,_,_,_,_)) {
            auto fld0 = new_identity_fold[()]();
            fld0 = @rec(fold_ann = bind init_blank_ann(_,_) 
                        with *fld0);
            ret fold_item[()]((), fld0, item);
        }
        case (item_fn(?i,?ff,?tps,?id,?ann)) {
            auto f_info = get_fn_info(fm, id);
            auto fld0 = new_identity_fold[fn_info]();
            fld0 = @rec(fold_ann = bind init_ann(_,_) 
                        with *fld0);
            auto outer = fold_item[fn_info](f_info, fld0, item);
            // now recurse into any nested items
            ret annotate_item_inner(fm, outer);
         }
        case (item_mod(?i, ?mm, ?id)) {
            auto fld0 = new_identity_fold[()]();
            fld0 = @rec(fold_ann = bind init_blank_ann(_,_) 
                        with *fld0);
            auto outer = fold_item[()]((), fld0, item);
            ret annotate_item_inner(fm, outer);
        }
        case (item_native_mod(?i, ?nm, ?id)) {
            ret item;
        }
        case (item_ty(_,_,_,_,_)) {
            ret item;
        }
        case (item_tag(_,_,_,_,_)) {
            ret item;
        }
        case (item_obj(?i,?ob,?tps,?odid,?ann)) {
            auto fld0 = new_identity_fold[()]();
            fld0 = @rec(fold_ann = bind init_blank_ann(_,_) 
                        with *fld0);
            auto outer = fold_item[()]((), fld0, item);
            ret annotate_item_inner(fm, outer);
        }
    }
}

fn annotate_module(&fn_info_map fm, &_mod module) -> _mod {
    let vec[@item] new_items = [];
   
    for (@item i in module.items) {
        auto new_item = annotate_item(fm, i);
        _vec::push[@item](new_items, new_item);
    }

    ret rec(items = new_items with module);
}

fn annotate_crate(&fn_info_map fm, &@crate crate) -> @crate {
    ret @respan(crate.span,
               rec(module = annotate_module(fm, crate.node.module)
                   with crate.node));
}
