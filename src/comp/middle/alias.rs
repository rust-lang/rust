import front::ast;
import front::ast::ident;
import front::ast::def_num;
import util::common::span;
import visit::vt;
import std::vec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import std::option::is_none;

tag valid {
    valid;
    overwritten(span, ast::path);
    val_taken(span, ast::path);
}

type restrict = @rec(vec[def_num] root_vars,
                     def_num block_defnum,
                     vec[def_num] bindings,
                     vec[ty::t] tys,
                     mutable valid ok);

type scope = vec[restrict];
type ctx = rec(@ty::ctxt tcx,
               resolve::def_map dm);

fn check_crate(@ty::ctxt tcx, resolve::def_map dm, &@ast::crate crate) {
    auto cx = @rec(tcx = tcx, dm = dm);
    auto v = @rec(visit_fn = visit_fn,
                  visit_expr = bind visit_expr(cx, _, _, _)
                  with *visit::default_visitor[scope]());
    visit::visit_crate(*crate, [], visit::vtor(v));
}

fn visit_fn(&ast::_fn f, &span sp, &ident name, &ast::def_id d_id,
            &ast::ann a, &scope sc, &vt[scope] v) {
    visit::visit_fn_decl(f.decl, sc, v);
    vt(v).visit_block(f.body, [], v);
}

fn visit_expr(&@ctx cx, &@ast::expr ex, &scope sc, &vt[scope] v) {
    auto handled = false;
    alt (ex.node) {
        case (ast::expr_call(?f, ?args, _)) {
            check_call(*cx, f, args, sc);
        }
        case (ast::expr_alt(?input, ?arms, _)) {
            check_alt(*cx, input, arms, sc, v);
            handled = true;
        }
        case (ast::expr_put(?val, _)) {
            alt (val) {
                case (some(?ex)) {
                    auto root = expr_root(*cx, ex, false);
                    if (!is_none(root.inner_mut)) {
                        cx.tcx.sess.span_err
                            (ex.span,
                             "result of put must be immutably rooted");
                    }
                    visit_expr(cx, ex, sc, v);
                }
                case (_) {}
            }
            handled = true;
        }
        case (ast::expr_for_each(?decl, ?call, ?block, _)) {
            check_for_each(*cx, decl, call, block, sc, v);
            handled = true;
        }
        case (ast::expr_for(?decl, ?seq, ?block, _)) {
            check_for(*cx, decl, seq, block, sc, v);
            handled = true;
        }

        case (ast::expr_path(?pt, ?ann)) {
            check_var(*cx, ex, pt, ann, false, sc);
        }
        case (ast::expr_move(?dest, ?src, _)) {
            check_assign(cx, dest, src, sc, v);
            handled = true;
        }
        case (ast::expr_assign(?dest, ?src, _)) {
            check_assign(cx, dest, src, sc, v);
            handled = true;
        }
        case (ast::expr_assign_op(_, ?dest, ?src, _)) {
            check_assign(cx, dest, src, sc, v);
            handled = true;
        }

        case (_) {}
    }
    if (!handled) { visit::visit_expr(ex, sc, v); }
}

fn check_call(&ctx cx, &@ast::expr f, &vec[@ast::expr] args, &scope sc)
    -> rec(vec[def_num] root_vars, vec[ty::t] unsafe_ts) {
    auto fty = ty::expr_ty(*cx.tcx, f);
    auto arg_ts = alt (ty::struct(*cx.tcx, fty)) {
        case (ty::ty_fn(_, ?args, _, _)) { args }
        case (ty::ty_native_fn(_, ?args, _)) { args }
    };

    auto i = 0u;
    let vec[def_num] roots = [];
    let vec[ty::t] unsafe_ts = [];
    let vec[uint] unsafe_t_offsets = [];
    for (ty::arg arg_t in arg_ts) {
        if (arg_t.mode != ty::mo_val) {
            auto root = expr_root(cx, args.(i), false);
            alt (path_def_id(cx, root.ex)) {
                case (some(?did)) { vec::push(roots, did._1); }
                case (_) {}
            }
            alt (root.inner_mut) {
                case (some(?t)) {
                    vec::push(unsafe_ts, t);
                    vec::push(unsafe_t_offsets, i);
                }
                case (_) {}
            }
        }
        i += 1u;
    }

    if (vec::len(unsafe_ts) > 0u) {
        alt (f.node) {
            case (ast::expr_path(_, ?ann)) {
                if (def_is_local(cx.dm.get(ann.id))) {
                    cx.tcx.sess.span_err
                        (f.span, #fmt("function may alias with argument \
                         %u, which is not immutably rooted",
                         unsafe_t_offsets.(0)));
                }
            }
            case (_) {}
        }
    }
    auto j = 0u;
    for (ty::t unsafe in unsafe_ts) {
        auto offset = unsafe_t_offsets.(j);
        j += 1u;
        auto i = 0u;
        for (ty::arg arg_t in arg_ts) {
            if (i != offset &&
                // FIXME false should be replace with mutability of alias
                ty_can_unsafely_include(cx, unsafe, arg_t.ty, false)) {
                cx.tcx.sess.span_err
                    (args.(i).span, #fmt("argument %u may alias with \
                     argument %u, which is not immutably rooted", i, offset));
            }
            i += 1u;
        }
    }
    // FIXME when mutable aliases can be distinguished, go over the args again
    // and ensure that we're not passing a root variable by mutable alias
    // (using roots and the scope root vars).

    ret rec(root_vars = roots, unsafe_ts = unsafe_ts);
}

fn check_alt(&ctx cx, &@ast::expr input, &vec[ast::arm] arms,
             &scope sc, &vt[scope] v) {
    visit::visit_expr(input, sc, v);
    auto root = expr_root(cx, input, true);
    auto roots = alt (path_def_id(cx, root.ex)) {
        case (some(?did)) { [did._1] }
        case (_) { [] }
    };
    let vec[ty::t] forbidden_tp = alt (root.inner_mut) {
        case (some(?t)) { [t] }
        case (_) { [] }
    };

    for (ast::arm a in arms) {
        auto dnums = arm_defnums(a);
        auto new_sc = sc;
        if (vec::len(dnums) > 0u) {
            vec::push(new_sc, @rec(root_vars=roots,
                                   block_defnum=dnums.(0),
                                   bindings=dnums,
                                   tys=forbidden_tp,
                                   mutable ok=valid));
        }
        visit::visit_arm(a, new_sc, v);
    }
}

fn arm_defnums(&ast::arm arm) -> vec[def_num] {
    auto dnums = [];
    fn walk_pat(&mutable vec[def_num] found, &@ast::pat p) {
        alt (p.node) {
            case (ast::pat_bind(_, ?did, _)) {
                vec::push(found, did._1);
            }
            case (ast::pat_tag(_, ?children, _)) {
                for (@ast::pat child in children) {
                    walk_pat(found, child);
                }
            }
            case (_) {}
        }
    }
    walk_pat(dnums, arm.pat);
    ret dnums;
}

fn check_for_each(&ctx cx, &@ast::decl decl, &@ast::expr call,
                  &ast::block block, &scope sc, &vt[scope] v) {
    visit::visit_expr(call, sc, v);
    alt (call.node) {
        case (ast::expr_call(?f, ?args, _)) {
            auto data = check_call(cx, f, args, sc);
            auto defnum = alt (decl.node) {
                case (ast::decl_local(?l)) { l.id._1 }
            };
            
            auto new_sc = @rec(root_vars=data.root_vars,
                               block_defnum=defnum,
                               bindings=[defnum],
                               tys=data.unsafe_ts,
                               mutable ok=valid);
            visit::visit_block(block, sc + [new_sc], v);
        }
    }
}

fn check_for(&ctx cx, &@ast::decl decl, &@ast::expr seq,
             &ast::block block, &scope sc, &vt[scope] v) {
    visit::visit_expr(seq, sc, v);
    auto defnum = alt (decl.node) {
        case (ast::decl_local(?l)) { l.id._1 }
    };

    auto root = expr_root(cx, seq, false);
    auto root_def = alt (path_def_id(cx, root.ex)) {
        case (some(?did)) { [did._1] }
        case (_) { [] }
    };
    auto unsafe = alt (root.inner_mut) {
        case (some(?t)) { [t] }
        case (_) { [] }
    };
    // If this is a mutable vector, don't allow it to be touched.
    auto seq_t = ty::expr_ty(*cx.tcx, seq);
    alt (ty::struct(*cx.tcx, seq_t)) {
        case (ty::ty_vec(?mt)) {
            if (mt.mut != ast::imm) { unsafe = [seq_t]; }
        }
        case (ty::ty_str) {}
    }

    auto new_sc = @rec(root_vars=root_def,
                       block_defnum=defnum,
                       bindings=[defnum],
                       tys=unsafe,
                       mutable ok=valid);
    visit::visit_block(block, sc + [new_sc], v);
}

fn check_var(&ctx cx, &@ast::expr ex, &ast::path p, ast::ann ann, bool assign,
             &scope sc) {
    auto def = cx.dm.get(ann.id);
    if (!def_is_local(def)) { ret; }
    auto my_defnum = ast::def_id_of_def(def)._1;
    auto var_t = ty::expr_ty(*cx.tcx, ex);
    for (restrict r in sc) {
        // excludes variables introduced since the alias was made
        if (my_defnum < r.block_defnum) {
            for (ty::t t in r.tys) {
                if (ty_can_unsafely_include(cx, t, var_t, assign)) {
                    r.ok = val_taken(ex.span, p);
                }
            }
        } else if (r.ok != valid && vec::member(my_defnum, r.bindings)) {
            fail_alias(cx, r.ok, p);
        }
    }
}

fn fail_alias(&ctx cx, valid issue, &ast::path pt) {
    auto base = " will invalidate alias " + ast::path_name(pt) +
        ", which is still used";
    alt (issue) {
        case (overwritten(?sp, ?wpt)) {
            cx.tcx.sess.span_err
                (sp, "overwriting " + ast::path_name(wpt) + base);
        }
        case (val_taken(?sp, ?vpt)) {
            cx.tcx.sess.span_err
                (sp, "taking the value of " + ast::path_name(vpt) +
                 base);
        }
    }
}

fn check_assign(&@ctx cx, &@ast::expr dest, &@ast::expr src,
                &scope sc, &vt[scope] v) {
    visit_expr(cx, src, sc, v);
    alt (dest.node) {
        case (ast::expr_path(?p, ?ann)) {
            auto dnum = ast::def_id_of_def(cx.dm.get(ann.id))._1;
            auto var_t = ty::expr_ty(*cx.tcx, dest);
            for (restrict r in sc) {
                if (vec::member(dnum, r.root_vars)) {
                    r.ok = overwritten(dest.span, p);
                }
            }
            check_var(*cx, dest, p, ann, true, sc);
        }
        case (_) {
            visit_expr(cx, dest, sc, v);
        }
    }
}

fn expr_root(&ctx cx, @ast::expr ex, bool autoderef)
    -> rec(@ast::expr ex, option::t[ty::t] inner_mut, bool mut_in_box) {
    let option::t[ty::t] mut = none;
    // This is not currently used but would make it possible to be more
    // liberal -- only stuff in a mutable box needs full type-inclusion
    // checking, things that aren't in a box need only be checked against
    // locally live aliases and their root.
    auto mut_in_box = false;
    while (true) {
        alt ({ex.node}) {
            case (ast::expr_field(?base, ?ident, _)) {
                auto base_t = ty::expr_ty(*cx.tcx, base);
                auto auto_unbox = maybe_auto_unbox(cx, base_t);
                alt (ty::struct(*cx.tcx, auto_unbox.t)) {
                    case (ty::ty_tup(?fields)) {
                        auto fnm = ty::field_num(cx.tcx.sess, ex.span, ident);
                        if (fields.(fnm).mut != ast::imm && is_none(mut)) {
                            mut = some(auto_unbox.t);
                        }
                    }
                    case (ty::ty_rec(?fields)) {
                        for (ty::field fld in fields) {
                            if (str::eq(ident, fld.ident)) {
                                if (fld.mt.mut != ast::imm && is_none(mut)) {
                                    mut = some(auto_unbox.t);
                                }
                                break;
                            }
                        }
                    }
                    case (ty::ty_obj(_)) {}
                }
                if (auto_unbox.done) {
                    if (!is_none(mut)) { mut_in_box = true; }
                    else if (auto_unbox.mut) { mut = some(base_t); }
                }
                ex = base;
            }
            case (ast::expr_index(?base, _, _)) {
                auto base_t = ty::expr_ty(*cx.tcx, base);
                auto auto_unbox = maybe_auto_unbox(cx, base_t);
                alt (ty::struct(*cx.tcx, auto_unbox.t)) {
                    case (ty::ty_vec(?mt)) {
                        if (mt.mut != ast::imm && is_none(mut)) {
                            mut = some(auto_unbox.t);
                        }
                    }
                }
                if (auto_unbox.done) {
                    if (!is_none(mut)) { mut_in_box = true; }
                    else if (auto_unbox.mut) { mut = some(base_t); }
                }
                if (auto_unbox.done && !is_none(mut)) {
                }
                ex = base;
            }
            case (ast::expr_unary(?op, ?base, _)) {
                if (op == ast::deref) {
                    auto base_t = ty::expr_ty(*cx.tcx, base);
                    alt (ty::struct(*cx.tcx, base_t)) {
                        case (ty::ty_box(?mt)) {
                            if (mt.mut != ast::imm && is_none(mut)) {
                                mut = some(base_t);
                            }
                            if (!is_none(mut)) {
                                mut_in_box = true;
                            }
                        }
                    }
                    ex = base;
                } else {
                    break;
                }
            }
            case (_) { break; }
        }
    }
    if (autoderef) {
        auto ex_t = ty::expr_ty(*cx.tcx, ex);
        auto auto_unbox = maybe_auto_unbox(cx, ex_t);
        if (auto_unbox.done) {
            if (!is_none(mut)) { mut_in_box = true; }
            else if (auto_unbox.mut) { mut = some(ex_t); }
        }
    }
    ret rec(ex = ex, inner_mut = mut, mut_in_box = mut_in_box);
}

fn maybe_auto_unbox(&ctx cx, &ty::t t)
    -> rec(ty::t t, bool done, bool mut) {
    alt (ty::struct(*cx.tcx, t)) {
        case (ty::ty_box(?mt)) {
            ret rec(t=mt.ty, done=true, mut=mt.mut != ast::imm);
        }
        case (_) {
            ret rec(t=t, done=false, mut=false);
        }
    }
}

fn path_def_id(&ctx cx, &@ast::expr ex) -> option::t[ast::def_id] {
    alt (ex.node) {
        case (ast::expr_path(_, ?ann)) {
            ret some(ast::def_id_of_def(cx.dm.get(ann.id)));
        }
        case (_) {
            ret none;
        }
    }
}

fn ty_can_unsafely_include(&ctx cx, ty::t needle, ty::t haystack, bool mut)
    -> bool {
    fn get_mut(bool cur, &ty::mt mt) -> bool {
        ret cur || mt.mut != ast::imm;
    }
    fn helper(&ty::ctxt tcx, ty::t needle, ty::t haystack, bool mut) -> bool {
        if (needle == haystack) { ret true; }
        alt (ty::struct(tcx, haystack)) {
            case (ty::ty_tag(_, ?ts)) {
                for (ty::t t in ts) {
                    if (helper(tcx, needle, t, mut)) { ret true; }
                }
                ret false;
            }
            case (ty::ty_box(?mt)) {
                ret helper(tcx, needle, mt.ty, get_mut(mut, mt));
            }
            case (ty::ty_vec(?mt)) {
                ret helper(tcx, needle, mt.ty, get_mut(mut, mt));
            }
            case (ty::ty_ptr(?mt)) {
                ret helper(tcx, needle, mt.ty, get_mut(mut, mt));
            }
            case (ty::ty_tup(?mts)) {
                for (ty::mt mt in mts) {
                    if (helper(tcx, needle, mt.ty, get_mut(mut, mt))) {
                        ret true;
                    }
                }
                ret false;
            }
            case (ty::ty_rec(?fields)) {
                for (ty::field f in fields) {
                    if (helper(tcx, needle, f.mt.ty, get_mut(mut, f.mt))) {
                        ret true;
                    }
                }
                ret false;
            }
            // These may contain anything.
            case (ty::ty_fn(_, _, _, _)) { ret true; }
            case (ty::ty_obj(_)) { ret true; }
            // A type param may include everything, but can only be treated as
            // opaque downstream, and is thus safe unless we saw mutable
            // fields, in which case the whole thing can be overwritten.
            case (ty::ty_param(_)) { ret mut; }
            case (_) { ret false; }
        }
    }
    ret helper(*cx.tcx, needle, haystack, mut);
}

fn def_is_local(&ast::def d) -> bool {
    ret alt (d) {
        case (ast::def_local(_)) { true }
        case (ast::def_arg(_)) { true }
        case (ast::def_obj_field(_)) { true }
        case (ast::def_binding(_)) { true }
        case (_) { false }
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
