import syntax::codemap::span;
import ast::*;

import std::ivec;
import std::vec;
import std::option;
import vec::map;

export ast_fold_precursor;
export ast_fold;
export default_ast_fold;
export make_fold;
export dummy_out;
export noop_fold_crate;

type ast_fold = @mutable a_f;

// We may eventually want to be able to fold over type parameters, too

type ast_fold_precursor = 
    rec(fn (&crate_ c, ast_fold) -> crate_                fold_crate,
        fn (&crate_directive_ cd, ast_fold) -> crate_directive_ 
                                                 fold_crate_directive,
        fn (&view_item_ i, ast_fold) -> view_item_        fold_view_item,
        fn (&@native_item i, ast_fold) -> @native_item    fold_native_item,
        fn (&@item i, ast_fold) -> @item                  fold_item,
        //unlike the others, item_ is non-trivial
        fn (&item_ i, ast_fold) -> item_        fold_item_underscore,
        fn (&method_ m, ast_fold) -> method_              fold_method,
        fn (&block_ b, ast_fold) -> block_                fold_block,
        fn (&stmt_ s, ast_fold) -> stmt_                  fold_stmt,
        fn (&arm a, ast_fold) -> arm                      fold_arm,
        fn (&pat_ p, ast_fold) -> pat_                    fold_pat,
        fn (&decl_ d, ast_fold) -> decl_                  fold_decl,
        fn (&expr_ e, ast_fold) -> expr_                  fold_expr,
        fn (&ty_ t, ast_fold) -> ty_                      fold_ty,
        fn (&constr_ c, ast_fold) -> constr_              fold_constr,
        fn (&_fn f, ast_fold) -> _fn                      fold_fn,
        fn (&_mod m, ast_fold) -> _mod                    fold_mod,
        fn (&native_mod, ast_fold) -> native_mod          fold_native_mod,
        fn (&variant_, ast_fold) -> variant_              fold_variant,
        fn (&ident, ast_fold) -> ident                    fold_ident,
        fn (&path_, ast_fold) -> path_                    fold_path,
        fn (&local_, ast_fold) -> local_                  fold_local
        );

type a_f =
    rec(fn (&crate c) -> crate                        fold_crate,
        fn (&@crate_directive cd) -> @crate_directive fold_crate_directive,
        fn (&@view_item i) -> @view_item              fold_view_item,
        fn (&@native_item i) -> @native_item          fold_native_item,
        fn (&@item i) -> @item                        fold_item,
        fn (&item_ i) -> item_                        fold_item_underscore,
        fn (&@method m) -> @method                    fold_method,
        fn (&block b) -> block                        fold_block,
        fn (&@stmt s) -> @stmt                        fold_stmt,
        fn (&arm a) -> arm                            fold_arm,
        fn (&@pat p) -> @pat                          fold_pat,
        fn (&@decl d) -> @decl                        fold_decl,
        fn (&@expr e) -> @expr                        fold_expr,
        fn (&@ty t) -> @ty                            fold_ty,
        fn (&@constr c) -> @constr                    fold_constr,
        fn (&_fn f) -> _fn                            fold_fn,
        fn (&_mod m) -> _mod                          fold_mod,
        fn (&native_mod) -> native_mod                fold_native_mod,
        fn (&variant) -> variant                      fold_variant,
        fn (&ident) -> ident                          fold_ident,
        fn (&path) -> path                            fold_path,
        fn (&@local) -> @local                        fold_local
        );

//fn nf_dummy[T](&T node) -> T { fail; }
fn nf_crate_dummy(&crate c) -> crate { fail; }
fn nf_crate_directive_dummy(&@crate_directive c) 
    -> @crate_directive { fail; }
fn nf_view_item_dummy(&@view_item v) -> @view_item { fail; } 
fn nf_native_item_dummy(&@native_item n) -> @native_item { fail; } 
fn nf_item_dummy(&@item i) -> @item { fail; } 
fn nf_item_underscore_dummy(&item_ i) -> item_ { fail; } 
fn nf_method_dummy(&@method m) -> @method { fail; } 
fn nf_block_dummy(&block b) -> block { fail; } 
fn nf_stmt_dummy(&@stmt s) -> @stmt { fail; } 
fn nf_arm_dummy(&arm a) -> arm { fail; } 
fn nf_pat_dummy(&@pat p) -> @pat { fail; } 
fn nf_decl_dummy(&@decl d) -> @decl { fail; } 
fn nf_expr_dummy(&@expr e) -> @expr { fail; } 
fn nf_ty_dummy(&@ty t) -> @ty { fail; } 
fn nf_constr_dummy(&@constr c) -> @constr { fail; } 
fn nf_fn_dummy(&_fn f) -> _fn { fail; } 
fn nf_mod_dummy(&_mod m) -> _mod { fail; } 
fn nf_native_mod_dummy(&native_mod n) -> native_mod { fail; } 
fn nf_variant_dummy(&variant v) -> variant { fail; } 
fn nf_ident_dummy(&ident i) -> ident { fail; } 
fn nf_path_dummy(&path p) -> path { fail; } 
fn nf_obj_field_dummy(&obj_field o) -> obj_field { fail; }
fn nf_local_dummy(&@local o) -> @local { fail; }

/* some little folds that probably aren't useful to have in ast_fold itself*/

//used in noop_fold_item and noop_fold_crate and noop_fold_crate_directive
fn fold_meta_item_(&@meta_item mi, ast_fold fld) -> @meta_item {
    ret @rec(node=
             alt (mi.node) {
                 case (meta_word(?id)) { meta_word(fld.fold_ident(id)) }
                 case (meta_list(?id, ?mis)) {
                     auto fold_meta_item = bind fold_meta_item_(_,fld);
                     meta_list(id, ivec::map(fold_meta_item, mis))
                 }
                 case (meta_name_value(?id,?s)) {
                     meta_name_value(fld.fold_ident(id),s)
                 }
             },
             span=mi.span);
}
//used in noop_fold_item and noop_fold_crate
fn fold_attribute_(&attribute at, fn(&@meta_item) -> @meta_item fmi) 
    -> attribute {
    ret rec(node=rec(style=at.node.style,
                     value=*fmi(@at.node.value)),
            span=at.span);
}
//used in noop_fold_native_item and noop_fold_fn
fn fold_arg_(&arg a, ast_fold fld) -> arg {
    ret rec(mode=a.mode, ty=fld.fold_ty(a.ty), 
            ident=fld.fold_ident(a.ident), id=a.id);
}




fn noop_fold_crate(&crate_ c, ast_fold fld) -> crate_ {
    auto fold_meta_item = bind fold_meta_item_(_,fld);
    auto fold_attribute = bind fold_attribute_(_,fold_meta_item);

    ret rec(directives=ivec::map(fld.fold_crate_directive, c.directives),
            module=fld.fold_mod(c.module),
            attrs=ivec::map(fold_attribute, c.attrs),
            config=ivec::map(fold_meta_item, c.config));
}

fn noop_fold_crate_directive(&crate_directive_ cd, ast_fold fld) 
    -> crate_directive_ {
    ret alt(cd) {
        case(cdir_src_mod(?id,?fname,?attrs)) { 
            cdir_src_mod(fld.fold_ident(id), fname, attrs)
                }
        case(cdir_dir_mod(?id,?fname,?cds,?attrs)) {
            cdir_dir_mod(fld.fold_ident(id),fname,
                         ivec::map(fld.fold_crate_directive, cds), attrs)
                }
        case(cdir_view_item(?vi)) { 
            cdir_view_item(fld.fold_view_item(vi))
                }
        case(cdir_syntax(_)) { cd }
        case(cdir_auth(_,_)) { cd }
    }
}

fn noop_fold_view_item(&view_item_ vi, ast_fold fld) -> view_item_ {
    ret vi;
}


fn noop_fold_native_item(&@native_item ni, ast_fold fld) -> @native_item {
    auto fold_arg = bind fold_arg_(_, fld);
    auto fold_meta_item = bind fold_meta_item_(_,fld);
    auto fold_attribute = bind fold_attribute_(_,fold_meta_item);

    ret @rec(ident=fld.fold_ident(ni.ident),
             attrs=ivec::map(fold_attribute, ni.attrs),
             node=alt (ni.node) {
                 case (native_item_ty) { native_item_ty }
                 case (native_item_fn(?st, ?fdec, ?typms)) {
                     native_item_fn(st, 
                                    rec(inputs=ivec::map(fold_arg,
                                                         fdec.inputs),
                                        output=fld.fold_ty(fdec.output),
                                        purity=fdec.purity, cf=fdec.cf,
                                        constraints=ivec::map(fld.fold_constr,
                                            fdec.constraints)),
                                    typms)
                 }
             },
             id=ni.id,
             span=ni.span);
}

fn noop_fold_item(&@item i, ast_fold fld) -> @item {
    auto fold_meta_item = bind fold_meta_item_(_,fld);
    auto fold_attribute = bind fold_attribute_(_,fold_meta_item);

    ret @rec(ident=fld.fold_ident(i.ident),
             attrs=ivec::map(fold_attribute,i.attrs),
             id=i.id, node=fld.fold_item_underscore(i.node),
             span=i.span);
}

fn noop_fold_item_underscore(&item_ i, ast_fold fld) -> item_ {
    fn fold_obj_field_(&obj_field of, ast_fold fld) -> obj_field {
        ret rec(mut=of.mut, ty=fld.fold_ty(of.ty), 
                ident=fld.fold_ident(of.ident), id=of.id);
    }
    auto fold_obj_field = bind fold_obj_field_(_,fld);

    ret alt(i) {
        case (item_const(?t, ?e)) {
            item_const(fld.fold_ty(t), fld.fold_expr(e))
        }
        case (item_fn(?f, ?typms)) {
            item_fn(fld.fold_fn(f), typms)
        }
        case (item_mod(?m)) { item_mod(fld.fold_mod(m)) }
        case (item_native_mod(?nm)) {
            item_native_mod(fld.fold_native_mod(nm))
                }
        case (item_ty(?t, ?typms)) {
            item_ty(fld.fold_ty(t), typms)
                }
        case (item_tag(?variants, ?typms)) {
            item_tag(ivec::map(fld.fold_variant, variants), typms)
                }
        case (item_obj(?o, ?typms, ?d)) {
            item_obj(rec(fields=ivec::map(fold_obj_field,o.fields),
                         methods=ivec::map(fld.fold_method,o.methods),
                         dtor=option::map(fld.fold_method,o.dtor)),
                     typms, d)
                }
        case (item_res(?dtor, ?did, ?typms, ?cid)) {
            item_res(fld.fold_fn(dtor), did, typms, cid)
        }
    };
}

fn noop_fold_method(&method_ m, ast_fold fld) -> method_ {
    ret rec(ident=fld.fold_ident(m.ident),
            meth=fld.fold_fn(m.meth), id=m.id); 
}


fn noop_fold_block(&block_ b, ast_fold fld) -> block_ {
    ret rec(stmts=ivec::map(fld.fold_stmt, b.stmts),
            expr=option::map(fld.fold_expr, b.expr), id=b.id);
}

fn noop_fold_stmt(&stmt_ s, ast_fold fld) -> stmt_ {
    ret alt(s) {
        case (stmt_decl(?d, ?nid)) { stmt_decl(fld.fold_decl(d), nid) }
        case (stmt_expr(?e, ?nid)) { stmt_expr(fld.fold_expr(e), nid) }
        case (stmt_crate_directive(?cd)) {
                stmt_crate_directive(fld.fold_crate_directive(cd))
                    }
    };
}

fn noop_fold_arm(&arm a, ast_fold fld) -> arm {
    ret rec(pat=fld.fold_pat(a.pat), block=fld.fold_block(a.block));
}

fn noop_fold_pat(&pat_ p, ast_fold fld) -> pat_ {
    ret alt (p) {
        case (pat_wild) { p }
        case (pat_bind(?ident)) { pat_bind(fld.fold_ident(ident))}
        case (pat_lit(_)) { p }
        case (pat_tag(?pth, ?pats)) {
            pat_tag(fld.fold_path(pth), ivec::map(fld.fold_pat, pats))
        }
    };
}

fn noop_fold_decl(&decl_ d, ast_fold fld) -> decl_ {
    ret alt (d) {
        // local really doesn't need its own fold...
        case (decl_local(?l)) {
            decl_local(fld.fold_local(l))
        }
        case (decl_item(?it)) { decl_item(fld.fold_item(it)) }
    }
}

fn noop_fold_expr(&expr_ e, ast_fold fld) -> expr_ {
    fn fold_elt_(&elt elt, ast_fold fld) -> elt {
        ret rec(mut=elt.mut, expr=fld.fold_expr(elt.expr));
    }
    auto fold_elt = bind fold_elt_(_,fld);
    fn fold_field_(&field field, ast_fold fld) -> field {
        ret rec(node=rec(mut=field.node.mut,
                         ident=fld.fold_ident(field.node.ident),
                         expr=fld.fold_expr(field.node.expr)),
                span=field.span);
    }
    auto fold_field = bind fold_field_(_,fld);
    fn fold_anon_obj_(&anon_obj ao, ast_fold fld) -> anon_obj {
        fn fold_anon_obj_field_(&anon_obj_field aof, ast_fold fld) 
            -> anon_obj_field {
            ret rec(mut=aof.mut, ty=fld.fold_ty(aof.ty), 
                    expr=fld.fold_expr(aof.expr),
                    ident=fld.fold_ident(aof.ident), id=aof.id);
        }
        auto fold_anon_obj_field = bind fold_anon_obj_field_(_,fld);

        ret rec(fields=alt(ao.fields) {
                    case (option::none[anon_obj_field[]]) { ao.fields }
                    case (option::some[anon_obj_field[]](?v)) {
                        option::some[anon_obj_field[]]
                            (ivec::map(fold_anon_obj_field, v))
                    }},
                methods=ivec::map(fld.fold_method, ao.methods),
                with_obj=option::map(fld.fold_expr, ao.with_obj))
    }
    auto fold_anon_obj = bind fold_anon_obj_(_,fld);
    

    ret alt (e) {
        case (expr_vec(?exprs, ?mut, ?seq_kind)) {
            expr_vec(ivec::map(fld.fold_expr, exprs), mut, seq_kind)
                }
        case (expr_tup(?elts)) {
            expr_tup(ivec::map(fold_elt, elts))
                }
        case (expr_rec(?fields, ?maybe_expr)) {
            expr_rec(ivec::map(fold_field, fields),
                     option::map(fld.fold_expr, maybe_expr))
                }
        case (expr_call(?f, ?args)) {
            expr_call(fld.fold_expr(f), ivec::map(fld.fold_expr, args))
                }
        case (expr_self_method(?id)) {
            expr_self_method(fld.fold_ident(id))
                }
        case (expr_bind(?f, ?args)) {
            auto opt_map_se = bind option::map(fld.fold_expr,_);
            expr_bind(fld.fold_expr(f), ivec::map(opt_map_se, args))
                }
        case (expr_spawn(?spawn_dom, ?name, ?f, ?args)) {
            expr_spawn(spawn_dom, name, fld.fold_expr(f), 
                       ivec::map(fld.fold_expr, args))
                }
        case (expr_binary(?binop, ?lhs, ?rhs)) {
            expr_binary(binop, fld.fold_expr(lhs), fld.fold_expr(rhs))
                }
        case (expr_unary(?binop, ?ohs)) {
            expr_unary(binop, fld.fold_expr(ohs))
                }
        case (expr_lit(_)) { e }
        case (expr_cast(?expr, ?ty)) {
            expr_cast(fld.fold_expr(expr), ty)
        }
        case (expr_if(?cond, ?tr, ?fl)) {
            expr_if(fld.fold_expr(cond), fld.fold_block(tr), 
                    option::map(fld.fold_expr, fl))
                }
        case (expr_ternary(?cond, ?tr, ?fl)) {
            expr_ternary(fld.fold_expr(cond),
                         fld.fold_expr(tr),
                         fld.fold_expr(fl))
                }
        case (expr_while(?cond, ?body)) {
            expr_while(fld.fold_expr(cond), fld.fold_block(body))
                }
        case (expr_for(?decl, ?expr, ?block)) {
            expr_for(fld.fold_local(decl), fld.fold_expr(expr), 
                     fld.fold_block(block))
                }
        case (expr_for_each(?decl, ?expr, ?block)) {
            expr_for_each(fld.fold_local(decl), fld.fold_expr(expr), 
                          fld.fold_block(block))
                }
        case (expr_do_while(?block, ?expr)) {
            expr_do_while(fld.fold_block(block), fld.fold_expr(expr))
                }
        case (expr_alt(?expr, ?arms)) {
            expr_alt(fld.fold_expr(expr), ivec::map(fld.fold_arm, arms))
                }
        case (expr_fn(?f)) {
            expr_fn(fld.fold_fn(f))
                }
        case (expr_block(?block)) {
            expr_block(fld.fold_block(block))
                }
        case (expr_move(?el, ?er)) {
            expr_move(fld.fold_expr(el), fld.fold_expr(er))
                }
        case (expr_assign(?el, ?er)) {
            expr_assign(fld.fold_expr(el), fld.fold_expr(er))
                }
        case (expr_swap(?el, ?er)) {
            expr_swap(fld.fold_expr(el), fld.fold_expr(er))
                }
        case (expr_assign_op(?op, ?el, ?er)) {
            expr_assign_op(op, fld.fold_expr(el), fld.fold_expr(er))
                }
        case (expr_send(?el, ?er)) {
            expr_send(fld.fold_expr(el), fld.fold_expr(er))
                }
        case (expr_recv(?el, ?er)) {
            expr_recv(fld.fold_expr(el), fld.fold_expr(er))
                }
        case (expr_field(?el, ?id)) {
            expr_field(fld.fold_expr(el), fld.fold_ident(id))
                }
        case (expr_index(?el, ?er)) {
            expr_index(fld.fold_expr(el), fld.fold_expr(er))
                }
        case (expr_path(?pth)) {
            expr_path(fld.fold_path(pth))
                }
        case (expr_ext(?pth, ?args, ?body, ?expanded)) {
            expr_ext(fld.fold_path(pth), ivec::map(fld.fold_expr, args),
                     body, fld.fold_expr(expanded))
                }
        case (expr_fail(_)) { e }
        case (expr_break()) { e }
        case (expr_cont()) { e }
        case (expr_ret(?e)) { 
            expr_ret(option::map(fld.fold_expr, e))
                }
        case (expr_put(?e)) { 
            expr_put(option::map(fld.fold_expr, e))
                }
        case (expr_be(?e)) { expr_be(fld.fold_expr(e)) }
        case (expr_log(?lv, ?e)) { expr_log(lv, fld.fold_expr(e)) }
        case (expr_assert(?e)) { expr_assert(fld.fold_expr(e)) }
        case (expr_check(?m, ?e)) { expr_check(m, fld.fold_expr(e)) }
        case (expr_if_check(?cond, ?tr, ?fl)) {
            expr_if_check(fld.fold_expr(cond), fld.fold_block(tr), 
                          option::map(fld.fold_expr, fl))
                }
        case (expr_port(?ot)) { 
            expr_port(alt(ot) {
                    case (option::some(?t)) { option::some(fld.fold_ty(t)) }
                    case (option::none) { option::none }
                })
                }
        case (expr_chan(?e)) { expr_chan(fld.fold_expr(e)) }
        case (expr_anon_obj(?ao, ?typms)) {
            expr_anon_obj(fold_anon_obj(ao), typms)
                }
    }
}

fn noop_fold_ty(&ty_ t, ast_fold fld) -> ty_ {
    //drop in ty::fold_ty here if necessary
    ret t;
}

fn noop_fold_constr(&constr_ c, ast_fold fld) -> constr_ {
    ret rec(path=fld.fold_path(c.path), args=c.args, id=c.id);
}

// functions just don't get spans, for some reason
fn noop_fold_fn(&_fn f, ast_fold fld) -> _fn {
    auto fold_arg = bind fold_arg_(_, fld);

    ret rec(decl= rec(inputs=ivec::map(fold_arg, f.decl.inputs),
                      output=fld.fold_ty(f.decl.output),
                      purity=f.decl.purity,
                      cf=f.decl.cf,
                      constraints=ivec::map(fld.fold_constr,
                                            f.decl.constraints)),
            proto = f.proto,
            body = fld.fold_block(f.body));
}

// ...nor do modules
fn noop_fold_mod(&_mod m, ast_fold fld) -> _mod {
    ret rec(view_items=ivec::map(fld.fold_view_item, m.view_items),
            items=ivec::map(fld.fold_item, m.items));
}

fn noop_fold_native_mod(&native_mod nm, ast_fold fld) -> native_mod {
    ret rec(native_name=nm.native_name,
            abi=nm.abi,
            view_items=ivec::map(fld.fold_view_item, nm.view_items),
            items=ivec::map(fld.fold_native_item, nm.items))
}

fn noop_fold_variant(&variant_ v, ast_fold fld) -> variant_ {
    fn fold_variant_arg_(&variant_arg va, ast_fold fld) -> variant_arg {
        ret rec(ty=fld.fold_ty(va.ty), id=va.id);
    }
    auto fold_variant_arg = bind fold_variant_arg_(_,fld);
    ret rec(name=v.name,
            args=ivec::map(fold_variant_arg, v.args),
            id=v.id);
}

fn noop_fold_ident(&ident i, ast_fold fld) -> ident {
    ret i;
}

fn noop_fold_path(&path_ p, ast_fold fld) -> path_ {
    ret rec(idents=ivec::map(fld.fold_ident, p.idents),
            types=ivec::map(fld.fold_ty, p.types));
}

fn noop_fold_local(&local_ l, ast_fold fld) -> local_ {
    ret rec(ty=option::map(fld.fold_ty,l.ty),
            infer=l.infer,
            ident=fld.fold_ident(l.ident),
            init=alt (l.init) {
                case (option::none[initializer]) { l.init }
                case (option::some[initializer](?init)) {
                    option::some[initializer]
                    (rec(op=init.op, 
                         expr=fld.fold_expr(init.expr)))
                }
            },
            id=l.id);
}


fn default_ast_fold() -> @ast_fold_precursor {
    ret @rec(fold_crate = noop_fold_crate,
             fold_crate_directive = noop_fold_crate_directive,
             fold_view_item = noop_fold_view_item,
             fold_native_item = noop_fold_native_item,
             fold_item = noop_fold_item,
             fold_item_underscore = noop_fold_item_underscore,
             fold_method = noop_fold_method,
             fold_block = noop_fold_block,
             fold_stmt = noop_fold_stmt,
             fold_arm = noop_fold_arm,
             fold_pat = noop_fold_pat,
             fold_decl = noop_fold_decl,
             fold_expr = noop_fold_expr,
             fold_ty = noop_fold_ty,
             fold_constr = noop_fold_constr,
             fold_fn = noop_fold_fn,
             fold_mod = noop_fold_mod,
             fold_native_mod = noop_fold_native_mod,
             fold_variant = noop_fold_variant,
             fold_ident = noop_fold_ident,
             fold_path = noop_fold_path,
             fold_local = noop_fold_local);
}

fn dummy_out(ast_fold a) {
    *a = rec(fold_crate = nf_crate_dummy,
                     fold_crate_directive = nf_crate_directive_dummy,
                     fold_view_item = nf_view_item_dummy,
                     fold_native_item = nf_native_item_dummy,
                     fold_item = nf_item_dummy,
                     fold_item_underscore = nf_item_underscore_dummy,
                     fold_method = nf_method_dummy,
                     fold_block = nf_block_dummy,
                     fold_stmt = nf_stmt_dummy,
                     fold_arm = nf_arm_dummy,
                     fold_pat = nf_pat_dummy,
                     fold_decl = nf_decl_dummy,
                     fold_expr = nf_expr_dummy,
                     fold_ty = nf_ty_dummy,
                     fold_constr = nf_constr_dummy,
                     fold_fn = nf_fn_dummy,
                     fold_mod = nf_mod_dummy,
                     fold_native_mod = nf_native_mod_dummy,
                     fold_variant = nf_variant_dummy,
                     fold_ident = nf_ident_dummy,
                     fold_path = nf_path_dummy,
                     fold_local = nf_local_dummy);
}


fn make_fold(&ast_fold_precursor afp) -> ast_fold {
    let ast_fold result = 
        @mutable rec(fold_crate = nf_crate_dummy,
                     fold_crate_directive = nf_crate_directive_dummy,
                     fold_view_item = nf_view_item_dummy,
                     fold_native_item = nf_native_item_dummy,
                     fold_item = nf_item_dummy,
                     fold_item_underscore = nf_item_underscore_dummy,
                     fold_method = nf_method_dummy,
                     fold_block = nf_block_dummy,
                     fold_stmt = nf_stmt_dummy,
                     fold_arm = nf_arm_dummy,
                     fold_pat = nf_pat_dummy,
                     fold_decl = nf_decl_dummy,
                     fold_expr = nf_expr_dummy,
                     fold_ty = nf_ty_dummy,
                     fold_constr = nf_constr_dummy,
                     fold_fn = nf_fn_dummy,
                     fold_mod = nf_mod_dummy,
                     fold_native_mod = nf_native_mod_dummy,
                     fold_variant = nf_variant_dummy,
                     fold_ident = nf_ident_dummy,
                     fold_path = nf_path_dummy,
                     fold_local = nf_local_dummy);

    /* naturally, a macro to write these would be nice */
    fn f_crate(&ast_fold_precursor afp, ast_fold f, &crate c) -> crate {
        ret rec(node=afp.fold_crate(c.node, f),
                span=c.span);
    }
    fn f_crate_directive(&ast_fold_precursor afp, ast_fold f, 
                         &@crate_directive c) -> @crate_directive {
        ret @rec(node=afp.fold_crate_directive(c.node, f),
                 span=c.span);
    }
    fn f_view_item(&ast_fold_precursor afp, ast_fold f, &@view_item x)
        -> @view_item {
        ret @rec(node=afp.fold_view_item(x.node, f), span=x.span);
    }
    fn f_native_item(&ast_fold_precursor afp, ast_fold f, &@native_item x)
        -> @native_item {
        ret afp.fold_native_item(x, f);
    }
    fn f_item(&ast_fold_precursor afp, ast_fold f, &@item i) -> @item {
        ret afp.fold_item(i, f);
    }
    fn f_item_underscore(&ast_fold_precursor afp, ast_fold f, &item_ i) ->
        item_ {
        ret afp.fold_item_underscore(i, f);
    }
    fn f_method(&ast_fold_precursor afp, ast_fold f, &@method x) -> @method {
        ret @rec(node=afp.fold_method(x.node, f), span=x.span);
    }
    fn f_block(&ast_fold_precursor afp, ast_fold f, &block x) -> block {
        ret rec(node=afp.fold_block(x.node, f), span=x.span);
    }
    fn f_stmt(&ast_fold_precursor afp, ast_fold f, &@stmt x) -> @stmt {
        ret @rec(node=afp.fold_stmt(x.node, f), span=x.span);
    }
    fn f_arm(&ast_fold_precursor afp, ast_fold f, &arm x) -> arm {
        ret afp.fold_arm(x, f);
    }
    fn f_pat(&ast_fold_precursor afp, ast_fold f, &@pat x) -> @pat {
        ret @rec(id=x.id, node=afp.fold_pat(x.node, f), span=x.span);
    }
    fn f_decl(&ast_fold_precursor afp, ast_fold f, &@decl x) -> @decl {
        ret @rec(node=afp.fold_decl(x.node, f), span=x.span);
    }
    fn f_expr(&ast_fold_precursor afp, ast_fold f, &@expr x) -> @expr {
        ret @rec(id=x.id, node=afp.fold_expr(x.node, f), span=x.span);
    }
    fn f_ty(&ast_fold_precursor afp, ast_fold f, &@ty x) -> @ty {
        ret @rec(node=afp.fold_ty(x.node, f), span=x.span);
    }
    fn f_constr(&ast_fold_precursor afp, ast_fold f, &@constr x) -> @constr {
        ret @rec(node=afp.fold_constr(x.node, f), span=x.span);
    }
    fn f_fn(&ast_fold_precursor afp, ast_fold f, &_fn x) -> _fn {
        ret afp.fold_fn(x, f);
    }    
    fn f_mod(&ast_fold_precursor afp, ast_fold f, &_mod x) -> _mod {
        ret afp.fold_mod(x, f);
    }
    fn f_native_mod(&ast_fold_precursor afp, ast_fold f, &native_mod x) -> 
        native_mod {
        ret afp.fold_native_mod(x, f);
    }    
    fn f_variant(&ast_fold_precursor afp, ast_fold f, &variant x)
        -> variant {
        ret rec(node=afp.fold_variant(x.node, f), span=x.span);
    }
    fn f_ident(&ast_fold_precursor afp, ast_fold f, &ident x) -> ident {
        ret afp.fold_ident(x, f);
    }
    fn f_path(&ast_fold_precursor afp, ast_fold f, &path x) -> path {
        ret rec(node=afp.fold_path(x.node, f), span=x.span);
    }
    fn f_local(&ast_fold_precursor afp, ast_fold f, &@local x) -> @local {
        ret @rec(node=afp.fold_local(x.node, f), span=x.span);
    }

    *result = rec(fold_crate = bind f_crate(afp,result,_),
                  fold_crate_directive = bind f_crate_directive(afp,result,_),
                  fold_view_item = bind f_view_item(afp,result,_),
                  fold_native_item = bind f_native_item(afp,result,_),
                  fold_item = bind f_item(afp,result,_),
                  fold_item_underscore = bind f_item_underscore(afp,result,_),
                  fold_method = bind f_method(afp,result,_),
                  fold_block = bind f_block(afp,result,_),
                  fold_stmt = bind f_stmt(afp,result,_),
                  fold_arm = bind f_arm(afp, result, _),
                  fold_pat = bind f_pat(afp,result,_),
                  fold_decl = bind f_decl(afp,result,_),
                  fold_expr = bind f_expr(afp,result,_),
                  fold_ty = bind f_ty(afp,result,_),
                  fold_constr = bind f_constr(afp,result,_),
                  fold_fn = bind f_fn(afp,result,_),
                  fold_mod = bind f_mod(afp,result,_),
                  fold_native_mod = bind f_native_mod(afp,result,_),
                  fold_variant = bind f_variant(afp,result,_),
                  fold_ident = bind f_ident(afp,result,_),
                  fold_path = bind f_path(afp,result,_),
                  fold_local = bind f_local(afp,result,_));
    ret result;
    /*
    ret rec(fold_crate = noop_fold_crate,
          fold_crate_directive = noop_fold_crate_drective,
          fold_view_item = noop_fold_view_item,
          fold_native_item = noop_fold_native_item,
          fold_item = noop_fold_item,
          fold_method = noop_fold_method,
          fold_block = noop_fold_block,
          fold_stmt = noop_fold_stmt,
          fold_arm = noop_fold_arm,
          fold_pat = noop_fold_pat,
          fold_decl = noop_fold_decl,
          fold_expr = noop_fold_expr,
          fold_ty = noop_fold_ty,
          fold_constr = noop_fold_constr,
          fold_fn = noop_fold_fn);*/
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
