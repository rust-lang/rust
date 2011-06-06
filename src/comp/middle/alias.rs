import front::ast;
import front::ast::ident;
import front::ast::def_id;
import std::vec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;

tag deref_t {
    field(ident);
    index;
    unbox;
}
type deref = rec(bool mut, deref_t t);

type ctx = @rec(@ty::ctxt tcx,
                resolve::def_map dm,
                // The current blacklisted (non-assignable) locals
                mutable vec[vec[def_id]] bl,
                // A stack of blacklists for outer function scopes
                mutable vec[vec[vec[def_id]]] blstack);

fn check_crate(@ty::ctxt tcx, resolve::def_map dm, &@ast::crate crate) {
    auto cx = @rec(tcx = tcx,
                   dm = dm,
                   mutable bl = vec::empty[vec[def_id]](),
                   mutable blstack = vec::empty[vec[vec[def_id]]]());
    auto v = rec(visit_item_pre = bind enter_item(cx, _),
                 visit_item_post = bind leave_item(cx, _),
                 visit_method_pre = bind enter_method(cx, _),
                 visit_method_post = bind leave_method(cx, _),
                 visit_expr_pre = bind check_expr(cx, _),
                 visit_expr_post = bind leave_expr(cx, _)
                 with walk::default_visitor());
    walk::walk_crate(v, *crate);
}

fn enter_item(ctx cx, &@ast::item it) {
    alt (it.node) {
        case (ast::item_fn(_, _, _, _, _)) {
            vec::push(cx.blstack, cx.bl);
            cx.bl = [];
        }
        case (_) {}
    }
}
fn leave_item(ctx cx, &@ast::item it) {
    alt (it.node) {
        case (ast::item_fn(_, _, _, _, _)) {
            cx.bl = vec::pop(cx.blstack);
        }
        case (_) {}
    }
}

fn enter_method(ctx cx, &@ast::method mt) {
    vec::push(cx.blstack, cx.bl);
    cx.bl = [];
}
fn leave_method(ctx cx, &@ast::method mt) {
    cx.bl = vec::pop(cx.blstack);
}

fn check_expr(ctx cx, &@ast::expr ex) {
    alt (ex.node) {
        case (ast::expr_call(?f, ?args, _)) {
            auto fty = ty::expr_ty(*cx.tcx, f);
            auto argtys = alt (ty::struct(*cx.tcx, fty)) {
                case (ty::ty_fn(_, ?args, _, _)) { args }
                case (ty::ty_native_fn(_, ?args, _)) { args }
            };
            auto i = 0u;
            let vec[def_id] listed = [];
            for (ty::arg argty in argtys) {
                // FIXME Treat mo_either specially here?
                if (argty.mode != ty::mo_val) {
                    alt (check_rooted(cx, args.(i), false)) {
                        case (some(?did)) {
                            vec::push(listed, did);
                        }
                        case (_) {}
                    }
                }
                i += 1u;
            }
            // FIXME when mutable aliases can be distinguished, go over the
            // args again and ensure that we're not passing a blacklisted
            // variable by mutable alias (using 'listed' and the context
            // blacklist).
        }
        case (ast::expr_put(?val, _)) {
            alt (val) {
                case (some(?ex)) { check_rooted(cx, ex, false); }
                case (_) {}
            }
        }
        case (ast::expr_alt(?input, _, _)) {
            vec::push(cx.bl, alt (check_rooted(cx, input, true)) {
                case (some(?did)) { [did] }
                case (_) { vec::empty[def_id]() }
            });
        }

        case (ast::expr_move(?dest, _, _)) { check_assign(cx, dest); }
        case (ast::expr_assign(?dest, _, _)) { check_assign(cx, dest); }
        case (ast::expr_assign_op(_, ?dest, _, _)) { check_assign(cx, dest); }
        case (_) {}
    }
}

fn leave_expr(ctx cx, &@ast::expr ex) {
    alt (ex.node) {
        case (ast::expr_alt(_, _, _)) { vec::pop(cx.bl); }
        case (_) {}
    }
}

fn check_assign(&ctx cx, &@ast::expr ex) {
    alt (ex.node) {
        case (ast::expr_path(?pt, ?ann)) {
            auto did = ast::def_id_of_def(cx.dm.get(ann.id));
            for (vec[def_id] layer in cx.bl) {
                for (def_id black in layer) {
                    if (did == black) {
                        cx.tcx.sess.span_err
                            (ex.span, str::connect(pt.node.idents, "::") +
                             " is being aliased and may not be assigned to");
                    }
                }
            }
        }
        case (_) {}
    }
}

fn check_rooted(&ctx cx, &@ast::expr ex, bool autoderef)
    -> option::t[def_id] {
    auto root = expr_root(cx, ex, autoderef);
    if (has_unsafe_box(root.ds)) {
        cx.tcx.sess.span_err
            (ex.span, "can not create alias to improperly anchored value");
    }
    alt (root.ex.node) {
        case (ast::expr_path(_, ?ann)) {
            ret some(ast::def_id_of_def(cx.dm.get(ann.id)));
        }
        case (_) {
            ret none[def_id];
        }
    }
}

fn expr_root(&ctx cx, @ast::expr ex, bool autoderef)
    -> rec(@ast::expr ex, vec[deref] ds) {
    let vec[deref] ds = [];
    if (autoderef) {
        auto auto_unbox = maybe_auto_unbox(cx, ex);
        if (auto_unbox.done) {
            vec::push(ds, rec(mut=auto_unbox.mut, t=unbox));
        }
    }
    while (true) {
        alt ({ex.node}) {
            case (ast::expr_field(?base, ?ident, _)) {
                auto auto_unbox = maybe_auto_unbox(cx, base);
                alt (auto_unbox.t) {
                    case (ty::ty_tup(?fields)) {
                        auto fnm = ty::field_num(cx.tcx.sess, ex.span, ident);
                        auto mt = fields.(fnm).mut != ast::imm;
                        vec::push(ds, rec(mut=mt, t=field(ident)));
                    }
                    case (ty::ty_rec(?fields)) {
                        for (ty::field fld in fields) {
                            if (str::eq(ident, fld.ident)) {
                                auto mt = fld.mt.mut != ast::imm;
                                vec::push(ds, rec(mut=mt, t=field(ident)));
                                break;
                            }
                        }
                    }
                    case (ty::ty_obj(_)) {
                        vec::push(ds, rec(mut=false, t=field(ident)));
                    }
                }
                if (auto_unbox.done) {
                    vec::push(ds, rec(mut=auto_unbox.mut, t=unbox));
                }
                ex = base;
            }
            case (ast::expr_index(?base, _, _)) {
                auto auto_unbox = maybe_auto_unbox(cx, base);
                alt (auto_unbox.t) {
                    case (ty::ty_vec(?mt)) {
                        vec::push(ds, rec(mut=mt.mut != ast::imm, t=index));
                    }
                }
                if (auto_unbox.done) {
                    vec::push(ds, rec(mut=auto_unbox.mut, t=unbox));
                }
                ex = base;
            }
            case (ast::expr_unary(?op, ?base, _)) {
                if (op == ast::deref) {
                    alt (ty::struct(*cx.tcx, ty::expr_ty(*cx.tcx, base))) {
                        case (ty::ty_box(?mt)) {
                            vec::push(ds, rec(mut=mt.mut!=ast::imm, t=unbox));
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
    vec::reverse(ds);
    ret rec(ex = ex, ds = ds);
}

fn maybe_auto_unbox(&ctx cx, &@ast::expr ex)
    -> rec(ty::sty t, bool done, bool mut) {
    auto tp = ty::struct(*cx.tcx, ty::expr_ty(*cx.tcx, ex));
    alt (tp) {
        case (ty::ty_box(?mt)) {
            ret rec(t=ty::struct(*cx.tcx, mt.ty),
                    done=true, mut=mt.mut != ast::imm);
        }
        case (_) { ret rec(t=tp, done=false, mut=false); }
    }
}

fn has_unsafe_box(&vec[deref] ds) -> bool {
    auto saw_mut = false;
    for (deref d in ds) {
        if (d.mut) { saw_mut = true; }
        if (d.t == unbox) {
            // Directly aliasing the content of a mutable box is never okay,
            // and any box living under mutable connection may be severed from
            // its root and freed.
            if (saw_mut) { ret true; }
        }
    }
    ret false;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
