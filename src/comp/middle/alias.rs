
import syntax::ast;
import ast::ident;
import ast::fn_ident;
import ast::node_id;
import ast::def_id;
import syntax::codemap::span;
import syntax::visit;
import visit::vt;
import std::ivec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import std::option::is_none;

// This is not an alias-analyser (though it would merit from becoming one, or
// getting input from one, to be more precise). It is a pass that checks
// whether aliases are used in a safe way. Beyond that, though it doesn't have
// a lot to do with aliases, it also checks whether assignments are valid
// (using an lval, which is actually mutable), since it already has all the
// information needed to do that (and the typechecker, which would be a
// logical place for such a check, doesn't).

tag valid { valid; overwritten(span, ast::path); val_taken(span, ast::path); }

type restrict =
    @rec(node_id[] root_vars,
         node_id block_defnum,
         node_id[] bindings,
         ty::t[] tys,
         uint[] depends_on,
         mutable valid ok);

type scope = @restrict[];

tag local_info { arg(ast::mode); objfield(ast::mutability); }

type ctx = rec(@ty::ctxt tcx,
               std::map::hashmap[node_id, local_info] local_map);

fn check_crate(@ty::ctxt tcx, &@ast::crate crate) {
    auto cx = @rec(tcx=tcx,
                   // Stores information about object fields and function
                   // arguments that's otherwise not easily available.
                   local_map=std::map::new_int_hash());
    auto v =
        @rec(visit_fn=bind visit_fn(cx, _, _, _, _, _, _, _),
             visit_item=bind visit_item(cx, _, _, _),
             visit_expr=bind visit_expr(cx, _, _, _),
             visit_decl=bind visit_decl(cx, _, _, _)
             with *visit::default_visitor[scope]());
    visit::visit_crate(*crate, @~[], visit::mk_vt(v));
    tcx.sess.abort_if_errors();
}

fn visit_fn(&@ctx cx, &ast::_fn f, &ast::ty_param[] tp, &span sp,
            &fn_ident name, ast::node_id id, &scope sc, &vt[scope] v) {
    visit::visit_fn_decl(f.decl, sc, v);
    for (ast::arg arg_ in f.decl.inputs) {
        cx.local_map.insert(arg_.id, arg(arg_.mode));
    }
    v.visit_block(f.body, @~[], v);
}

fn visit_item(&@ctx cx, &@ast::item i, &scope sc, &vt[scope] v) {
    alt (i.node) {
        case (ast::item_obj(?o, _, _)) {
            for (ast::obj_field f in o.fields) {
                cx.local_map.insert(f.id, objfield(f.mut));
            }
        }
        case (_) { }
    }
    visit::visit_item(i, sc, v);
}

fn visit_expr(&@ctx cx, &@ast::expr ex, &scope sc, &vt[scope] v) {
    auto handled = true;
    alt (ex.node) {
        ast::expr_call(?f, ?args) {
            check_call(*cx, f, args, sc);
            handled = false;
        }
        ast::expr_be(?cl) {
            check_tail_call(*cx, cl);
            visit::visit_expr(cl, sc, v);
        }
        ast::expr_alt(?input, ?arms) {
            check_alt(*cx, input, arms, sc, v);
        }
        ast::expr_put(?val) {
            alt (val) {
                case (some(?ex)) {
                    auto root = expr_root(*cx, ex, false);
                    if (mut_field(root.ds)) {
                        cx.tcx.sess.span_err(ex.span,
                                             "result of put must be" +
                                                 " immutably rooted");
                    }
                    visit_expr(cx, ex, sc, v);
                }
                case (_) { }
            }
        }
        ast::expr_for_each(?decl, ?call, ?block) {
            check_for_each(*cx, decl, call, block, sc, v);
        }
        ast::expr_for(?decl, ?seq, ?block) {
            check_for(*cx, decl, seq, block, sc, v);
        }
        ast::expr_path(?pt) {
            check_var(*cx, ex, pt, ex.id, false, sc);
            handled = false;
        }
        ast::expr_swap(?lhs, ?rhs) {
            check_lval(cx, lhs, sc, v);
            check_lval(cx, rhs, sc, v);
            handled = false;
        }
        ast::expr_move(?dest, ?src) {
            check_assign(cx, dest, src, sc, v);
            check_move_rhs(cx, src, sc, v);
        }
        ast::expr_assign(?dest, ?src) | ast::expr_assign_op(_, ?dest, ?src) {
            check_assign(cx, dest, src, sc, v);
        }
        _ { handled = false; }
    }
    if (!handled) { visit::visit_expr(ex, sc, v); }
}

fn visit_decl(&@ctx cx, &@ast::decl d, &scope sc, &vt[scope] v) {
    visit::visit_decl(d, sc, v);
    alt (d.node) {
        ast::decl_local(?loc) {
            alt (loc.node.init) {
                some(?init) {
                    if (init.op == ast::init_move) {
                        check_move_rhs(cx, init.expr, sc, v);
                    }
                }
                none {}
            }
        }
        _ {}
    }
}

fn check_call(&ctx cx, &@ast::expr f, &(@ast::expr)[] args, &scope sc) ->
   rec(node_id[] root_vars, ty::t[] unsafe_ts) {
    auto fty = ty::expr_ty(*cx.tcx, f);
    auto arg_ts = fty_args(cx, fty);
    let node_id[] roots = ~[];
    let tup(uint, node_id)[] mut_roots = ~[];
    let ty::t[] unsafe_ts = ~[];
    let uint[] unsafe_t_offsets = ~[];
    auto i = 0u;
    for (ty::arg arg_t in arg_ts) {
        if (arg_t.mode != ty::mo_val) {
            auto arg = args.(i);
            auto root = expr_root(cx, arg, false);
            if (arg_t.mode == ty::mo_alias(true)) {
                alt (path_def_id(cx, arg)) {
                  some(?did) { mut_roots += ~[tup(i, did._1)]; }
                  _ {
                    if (!mut_field(root.ds)) {
                        auto m = "passing a temporary value or \
                                 immutable field by mutable alias";
                        cx.tcx.sess.span_err(arg.span, m);
                    }
                  }
                }
            }
            alt (path_def_id(cx, root.ex)) {
              some(?did) { roots += ~[did._1]; }
              _ { }
            }
            alt (inner_mut(root.ds)) {
              some(?t) {
                unsafe_ts += ~[t];
                unsafe_t_offsets += ~[i];
              }
              _ { }
            }
        }
        i += 1u;
    }
    if (ivec::len(unsafe_ts) > 0u) {
        alt (f.node) {
            case (ast::expr_path(_)) {
                if (def_is_local(cx.tcx.def_map.get(f.id), true)) {
                    cx.tcx.sess.span_err(f.span,
                                         #fmt("function may alias with \
                         argument %u, which is not immutably rooted",
                                              unsafe_t_offsets.(0)));
                }
            }
            case (_) { }
        }
    }
    auto j = 0u;
    for (ty::t unsafe in unsafe_ts) {
        auto offset = unsafe_t_offsets.(j);
        j += 1u;
        auto i = 0u;
        for (ty::arg arg_t in arg_ts) {
            auto mut_alias = arg_t.mode == ty::mo_alias(true);
            if (i != offset &&
                    ty_can_unsafely_include(cx, unsafe, arg_t.ty, mut_alias))
               {
                cx.tcx.sess.span_err(args.(i).span,
                                     #fmt("argument %u may alias with \
                     argument %u, which is not immutably rooted",
                                          i, offset));
            }
            i += 1u;
        }
    }
    // Ensure we're not passing a root by mutable alias.

    for (tup(uint, node_id) root in mut_roots) {
        auto mut_alias_to_root = false;
        auto mut_alias_to_root_count = 0u;
        for (node_id r in roots) {
            if root._1 == r {
                mut_alias_to_root_count += 1u;
                if mut_alias_to_root_count > 1u {
                    mut_alias_to_root = true;
                    break;
                }
            }
        }

        if (mut_alias_to_root) {
            cx.tcx.sess.span_err(args.(root._0).span,
                                 "passing a mutable alias to a \
                 variable that roots another alias");
        }
    }
    ret rec(root_vars=roots, unsafe_ts=unsafe_ts);
}

fn check_tail_call(&ctx cx, &@ast::expr call) {
    auto args;
    auto f =
        alt (call.node) {
            case (ast::expr_call(?f, ?args_)) { args = args_; f }
        };
    auto i = 0u;
    for (ty::arg arg_t in fty_args(cx, ty::expr_ty(*cx.tcx, f))) {
        if (arg_t.mode != ty::mo_val) {
            auto mut_a = arg_t.mode == ty::mo_alias(true);
            auto ok = true;
            alt (args.(i).node) {
                case (ast::expr_path(_)) {
                    auto def = cx.tcx.def_map.get(args.(i).id);
                    auto dnum = ast::def_id_of_def(def)._1;
                    alt (cx.local_map.find(dnum)) {
                        case (some(arg(ast::alias(?mut)))) {
                            if (mut_a && !mut) {
                                cx.tcx.sess.span_err(args.(i).span,
                                                      "passing an immutable \
                                     alias by mutable alias");
                            }
                        }
                        case (_) { ok = !def_is_local(def, false); }
                    }
                }
                case (_) { ok = false; }
            }
            if (!ok) {
                cx.tcx.sess.span_err(args.(i).span,
                                      "can not pass a local value by \
                                     alias to a tail call");
            }
        }
        i += 1u;
    }
}

fn check_alt(&ctx cx, &@ast::expr input, &ast::arm[] arms, &scope sc,
             &vt[scope] v) {
    visit::visit_expr(input, sc, v);
    auto root = expr_root(cx, input, true);
    auto roots = alt (path_def_id(cx, root.ex)) {
      some(?did) { ~[did._1] }
      _ { ~[] }
    };
    let ty::t[] forbidden_tp =
        alt (inner_mut(root.ds)) { some(?t) { ~[t] } _ { ~[] } };
    for (ast::arm a in arms) {
        auto dnums = arm_defnums(a);
        auto new_sc = sc;
        if (ivec::len(dnums) > 0u) {
            new_sc = @(*sc + ~[@rec(root_vars=roots,
                                    block_defnum=dnums.(0),
                                    bindings=dnums,
                                    tys=forbidden_tp,
                                    depends_on=deps(sc, roots),
                                    mutable ok=valid)]);
        }
        visit::visit_arm(a, new_sc, v);
    }
}

fn arm_defnums(&ast::arm arm) -> node_id[] {
    auto dnums = ~[];
    fn walk_pat(&mutable node_id[] found, &@ast::pat p) {
        alt (p.node) {
            case (ast::pat_bind(_)) { found += ~[p.id]; }
            case (ast::pat_tag(_, ?children)) {
                for (@ast::pat child in children) { walk_pat(found, child); }
            }
            case (ast::pat_rec(?fields, _)) {
                for (ast::field_pat f in fields) { walk_pat(found, f.pat); }
            }
            case (ast::pat_box(?inner)) { walk_pat(found, inner); }
            case (_) { }
        }
    }
    walk_pat(dnums, arm.pats.(0));
    ret dnums;
}

fn check_for_each(&ctx cx, &@ast::local local, &@ast::expr call,
                  &ast::block block, &scope sc, &vt[scope] v) {
    visit::visit_expr(call, sc, v);
    alt (call.node) {
        case (ast::expr_call(?f, ?args)) {
            auto data = check_call(cx, f, args, sc);
            auto defnum = local.node.id;
            auto new_sc =
                @rec(root_vars=data.root_vars,
                     block_defnum=defnum,
                     bindings=~[defnum],
                     tys=data.unsafe_ts,
                     depends_on=deps(sc, data.root_vars),
                     mutable ok=valid);
            visit::visit_block(block, @(*sc + ~[new_sc]), v);
        }
    }
}

fn check_for(&ctx cx, &@ast::local local, &@ast::expr seq, &ast::block block,
             &scope sc, &vt[scope] v) {
    visit::visit_expr(seq, sc, v);
    auto defnum = local.node.id;
    auto root = expr_root(cx, seq, false);
    auto root_def = alt (path_def_id(cx, root.ex)) {
      some(?did) { ~[did._1] }
      _ { ~[] }
    };
    auto unsafe = alt (inner_mut(root.ds)) { some(?t) { ~[t] } _ { ~[] } };

    // If this is a mutable vector, don't allow it to be touched.
    auto seq_t = ty::expr_ty(*cx.tcx, seq);
    alt (ty::struct(*cx.tcx, seq_t)) {
        ty::ty_vec(?mt) | ty::ty_ivec(?mt) {
            if (mt.mut != ast::imm) { unsafe = ~[seq_t]; }
        }
        ty::ty_str | ty::ty_istr { /* no-op */ }
        _ {
            cx.tcx.sess.span_unimpl(seq.span, "unknown seq type " +
                                    util::ppaux::ty_to_str(*cx.tcx, seq_t));
        }
    }
    auto new_sc =
        @rec(root_vars=root_def,
             block_defnum=defnum,
             bindings=~[defnum],
             tys=unsafe,
             depends_on=deps(sc, root_def),
             mutable ok=valid);
    visit::visit_block(block, @(*sc + ~[new_sc]), v);
}

fn check_var(&ctx cx, &@ast::expr ex, &ast::path p, ast::node_id id,
             bool assign, &scope sc) {
    auto def = cx.tcx.def_map.get(id);
    if (!def_is_local(def, true)) { ret; }
    auto my_defnum = ast::def_id_of_def(def)._1;
    auto var_t = ty::expr_ty(*cx.tcx, ex);
    for (restrict r in *sc) {
        // excludes variables introduced since the alias was made
        if (my_defnum < r.block_defnum) {
            for (ty::t t in r.tys) {
                if (ty_can_unsafely_include(cx, t, var_t, assign)) {
                    r.ok = val_taken(ex.span, p);
                }
            }
        } else if (ivec::member(my_defnum, r.bindings)) {
            test_scope(cx, sc, r, p);
        }
    }
}

fn check_lval(&@ctx cx, &@ast::expr dest, &scope sc, &vt[scope] v) {
    alt (dest.node) {
        case (ast::expr_path(?p)) {
            auto dnum = ast::def_id_of_def(cx.tcx.def_map.get(dest.id))._1;
            if (is_immutable_alias(cx, sc, dnum)) {
                cx.tcx.sess.span_err(dest.span,
                                     "assigning to immutable alias");
            } else if (is_immutable_objfield(cx, dnum)) {
                cx.tcx.sess.span_err(dest.span,
                                     "assigning to immutable obj field");
            }
            for (restrict r in *sc) {
                if (ivec::member(dnum, r.root_vars)) {
                    r.ok = overwritten(dest.span, p);
                }
            }
        }
        case (_) {
            auto root = expr_root(*cx, dest, false);
            if (ivec::len(*root.ds) == 0u) {
                cx.tcx.sess.span_err(dest.span, "assignment to non-lvalue");
            } else if (!root.ds.(0).mut) {
                auto name =
                    alt (root.ds.(0).kind) {
                        case (unbox) { "box" }
                        case (field) { "field" }
                        case (index) { "vec content" }
                    };
                cx.tcx.sess.span_err(dest.span,
                                     "assignment to immutable " + name);
            }
            visit_expr(cx, dest, sc, v);
        }
    }
}

fn check_move_rhs(&@ctx cx, &@ast::expr src, &scope sc, &vt[scope] v) {
    alt (src.node) {
        case (ast::expr_path(?p)) {
            alt (cx.tcx.def_map.get(src.id)) {
                ast::def_obj_field(_) {
                    cx.tcx.sess.span_err
                        (src.span, "may not move out of an obj field");
                }
                _ {}
            }
            check_lval(cx, src, sc, v);
        }
        case (_) {
            auto root = expr_root(*cx, src, false);
            // Not a path and no-derefs means this is a temporary.
            if (ivec::len(*root.ds) != 0u) {
                cx.tcx.sess.span_err
                    (src.span, "moving out of a data structure");
            }
        }
    }
}

fn check_assign(&@ctx cx, &@ast::expr dest, &@ast::expr src, &scope sc,
                &vt[scope] v) {
    visit_expr(cx, src, sc, v);
    check_lval(cx, dest, sc, v);
}


fn is_immutable_alias(&@ctx cx, &scope sc, node_id dnum) -> bool {
    alt (cx.local_map.find(dnum)) {
        case (some(arg(ast::alias(false)))) { ret true; }
        case (_) { }
    }
    for (restrict r in *sc) {
        if (ivec::member(dnum, r.bindings)) { ret true; }
    }
    ret false;
}

fn is_immutable_objfield(&@ctx cx, node_id dnum) -> bool {
    ret cx.local_map.find(dnum) == some(objfield(ast::imm));
}

fn test_scope(&ctx cx, &scope sc, &restrict r, &ast::path p) {
    auto prob = r.ok;
    for (uint dep in r.depends_on) {
        if (prob != valid) { break; }
        prob = sc.(dep).ok;
    }
    if (prob != valid) {
        auto msg =
            alt (prob) {
                case (overwritten(?sp, ?wpt)) {
                    tup(sp, "overwriting " + ast::path_name(wpt))
                }
                case (val_taken(?sp, ?vpt)) {
                    tup(sp, "taking the value of " + ast::path_name(vpt))
                }
            };
        cx.tcx.sess.span_err(msg._0,
                             msg._1 + " will invalidate alias " +
                                 ast::path_name(p) + ", which is still used");
    }
}

fn deps(&scope sc, &node_id[] roots) -> uint[] {
    auto i = 0u;
    auto result = ~[];
    for (restrict r in *sc) {
        for (node_id dn in roots) {
            if ivec::member(dn, r.bindings) { result += ~[i]; }
        }
        i += 1u;
    }
    ret result;
}

tag deref_t { unbox; field; index; }

type deref = @rec(bool mut, deref_t kind, ty::t outer_t);


// Finds the root (the thing that is dereferenced) for the given expr, and a
// vec of dereferences that were used on this root. Note that, in this vec,
// the inner derefs come in front, so foo.bar.baz becomes rec(ex=foo,
// ds=[field(baz),field(bar)])
fn expr_root(&ctx cx, @ast::expr ex, bool autoderef) ->
   rec(@ast::expr ex, @deref[] ds) {
    fn maybe_auto_unbox(&ctx cx, &ty::t t) ->
       rec(ty::t t, option::t[deref] d) {
        alt (ty::struct(*cx.tcx, t)) {
            case (ty::ty_box(?mt)) {
                ret rec(t=mt.ty,
                        d=some(@rec(mut=mt.mut != ast::imm,
                                    kind=unbox,
                                    outer_t=t)));
            }
            case (_) { ret rec(t=t, d=none); }
        }
    }
    fn maybe_push_auto_unbox(&option::t[deref] d, &mutable deref[] ds) {
        alt (d) { case (some(?d)) { ds += ~[d]; } case (none) { } }
    }
    let deref[] ds = ~[];
    while (true) {
        alt ({ ex.node }) {
            case (ast::expr_field(?base, ?ident)) {
                auto auto_unbox =
                    maybe_auto_unbox(cx, ty::expr_ty(*cx.tcx, base));
                auto mut = false;
                alt (ty::struct(*cx.tcx, auto_unbox.t)) {
                    case (ty::ty_tup(?fields)) {
                        auto fnm = ty::field_num(cx.tcx.sess, ex.span, ident);
                        mut = fields.(fnm).mut != ast::imm;
                    }
                    case (ty::ty_rec(?fields)) {
                        for (ty::field fld in fields) {
                            if (str::eq(ident, fld.ident)) {
                                mut = fld.mt.mut != ast::imm;
                                break;
                            }
                        }
                    }
                    case (ty::ty_obj(_)) { }
                }
                ds += ~[@rec(mut=mut, kind=field, outer_t=auto_unbox.t)];
                maybe_push_auto_unbox(auto_unbox.d, ds);
                ex = base;
            }
            case (ast::expr_index(?base, _)) {
                auto auto_unbox =
                    maybe_auto_unbox(cx, ty::expr_ty(*cx.tcx, base));
                alt (ty::struct(*cx.tcx, auto_unbox.t)) {
                    case (ty::ty_vec(?mt)) {
                        ds += ~[@rec(mut=mt.mut != ast::imm,
                                    kind=index,
                                    outer_t=auto_unbox.t)];
                    }
                    case (ty::ty_ivec(?mt)) {
                        ds += ~[@rec(mut=mt.mut != ast::imm,
                                    kind=index,
                                    outer_t=auto_unbox.t)];
                    }
                }
                maybe_push_auto_unbox(auto_unbox.d, ds);
                ex = base;
            }
            case (ast::expr_unary(?op, ?base)) {
                if (op == ast::deref) {
                    auto base_t = ty::expr_ty(*cx.tcx, base);
                    auto mut = false;
                    alt (ty::struct(*cx.tcx, base_t)) {
                        case (ty::ty_box(?mt)) { mut = mt.mut != ast::imm; }
                        case (ty::ty_res(_, _, _)) {}
                        case (ty::ty_tag(_, _)) {}
                        case (ty::ty_ptr(?mt)) { mut = mt.mut != ast::imm; }
                    }
                    ds += ~[@rec(mut=mut, kind=unbox, outer_t=base_t)];
                    ex = base;
                } else { break; }
            }
            case (_) { break; }
        }
    }
    if (autoderef) {
        auto auto_unbox = maybe_auto_unbox(cx, ty::expr_ty(*cx.tcx, ex));
        maybe_push_auto_unbox(auto_unbox.d, ds);
    }
    ret rec(ex=ex, ds=@ds);
}

fn mut_field(&@deref[] ds) -> bool {
    for (deref d in *ds) { if (d.mut) { ret true; } }
    ret false;
}

fn inner_mut(&@deref[] ds) -> option::t[ty::t] {
    for (deref d in *ds) { if (d.mut) { ret some(d.outer_t); } }
    ret none;
}

fn path_def_id(&ctx cx, &@ast::expr ex) -> option::t[ast::def_id] {
    alt (ex.node) {
        case (ast::expr_path(_)) {
            ret some(ast::def_id_of_def(cx.tcx.def_map.get(ex.id)));
        }
        case (_) { ret none; }
    }
}

fn ty_can_unsafely_include(&ctx cx, ty::t needle, ty::t haystack, bool mut) ->
   bool {
    fn get_mut(bool cur, &ty::mt mt) -> bool {
        ret cur || mt.mut != ast::imm;
    }
    fn helper(&ty::ctxt tcx, ty::t needle, ty::t haystack, bool mut) -> bool {
        if (needle == haystack) { ret true; }
        alt (ty::struct(tcx, haystack)) {
            ty::ty_tag(_, ?ts) {
                for (ty::t t in ts) {
                    if (helper(tcx, needle, t, mut)) { ret true; }
                }
                ret false;
            }
            ty::ty_box(?mt) | ty::ty_vec(?mt) | ty::ty_ptr(?mt) {
                ret helper(tcx, needle, mt.ty, get_mut(mut, mt));
            }
            ty::ty_tup(?mts) {
                for (ty::mt mt in mts) {
                    if (helper(tcx, needle, mt.ty, get_mut(mut, mt))) {
                        ret true;
                    }
                }
                ret false;
            }
            ty::ty_rec(?fields) {
                for (ty::field f in fields) {
                    if (helper(tcx, needle, f.mt.ty, get_mut(mut, f.mt))) {
                        ret true;
                    }
                }
                ret false;
            }
            // These may contain anything.
            ty::ty_fn(_, _, _, _, _) {
                ret true;
            }
            ty::ty_obj(_) { ret true; }
            // A type param may include everything, but can only be
            // treated as opaque downstream, and is thus safe unless we
            // saw mutable fields, in which case the whole thing can be
            // overwritten.
            ty::ty_param(_) { ret mut; }
            _ { ret false; }
        }
    }
    ret helper(*cx.tcx, needle, haystack, mut);
}

fn def_is_local(&ast::def d, bool objfields_count) -> bool {
    ret alt (d) {
        ast::def_local(_) | ast::def_arg(_) | ast::def_binding(_) { true }
        ast::def_obj_field(_) { objfields_count }
        _ { false }
    };
}

fn fty_args(&ctx cx, ty::t fty) -> ty::arg[] {
    ret alt (ty::struct(*cx.tcx, ty::type_autoderef(*cx.tcx, fty))) {
        ty::ty_fn(_, ?args, _, _, _) | ty::ty_native_fn(_, ?args, _) { args }
    };
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
