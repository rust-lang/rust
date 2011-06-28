
import front::ast;
import front::ast::ident;
import front::ast::node_id;
import front::ast::def_id;
import util::common::span;
import visit::vt;
import std::vec;
import std::str;
import std::option;
import std::option::some;
import std::option::none;
import std::option::is_none;


// This is not an alias-analyser (though it would merit from becoming one, or
// at getting input from one, to be more precise). It is a pass that checks
// whether aliases are used in a safe way. Beyond that, though it doesn't have
// a lot to do with aliases, it also checks whether assignments are valid
// (using an lval, which is actually mutable), since it already has all the
// information needed to do that (and the typechecker, which would be a
// logical place for such a check, doesn't).
tag valid { valid; overwritten(span, ast::path); val_taken(span, ast::path); }

type restrict =
    @rec(vec[node_id] root_vars,
         node_id block_defnum,
         vec[node_id] bindings,
         vec[ty::t] tys,
         vec[uint] depends_on,
         mutable valid ok);

type scope = vec[restrict];

tag local_info { arg(ast::mode); objfield(ast::mutability); }

type ctx = rec(@ty::ctxt tcx,
               std::map::hashmap[node_id, local_info] local_map);

fn check_crate(@ty::ctxt tcx, &@ast::crate crate) {
    auto cx =
        @rec(tcx=tcx,

             // Stores information about object fields and function
             // arguments that's otherwise not easily available.
             local_map=util::common::new_int_hash());
    auto v =
        @rec(visit_fn=bind visit_fn(cx, _, _, _, _, _, _, _),
             visit_item=bind visit_item(cx, _, _, _),
             visit_expr=bind visit_expr(cx, _, _, _)
             with *visit::default_visitor[scope]());
    visit::visit_crate(*crate, [], visit::vtor(v));
}

fn visit_fn(@ctx cx, &ast::_fn f, &vec[ast::ty_param] tp, &span sp,
            &ident name, ast::node_id id, &scope sc, &vt[scope] v) {
    visit::visit_fn_decl(f.decl, sc, v);
    for (ast::arg arg_ in f.decl.inputs) {
        cx.local_map.insert(arg_.id, arg(arg_.mode));
    }
    vt(v).visit_block(f.body, [], v);
}

fn visit_item(@ctx cx, &@ast::item i, &scope sc, &vt[scope] v) {
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

fn visit_expr(@ctx cx, &@ast::expr ex, &scope sc, &vt[scope] v) {
    auto handled = true;
    alt (ex.node) {
        case (ast::expr_call(?f, ?args)) {
            check_call(*cx, f, args, sc);
            handled = false;
        }
        case (ast::expr_be(?cl)) {
            check_tail_call(*cx, cl);
            visit::visit_expr(cl, sc, v);
        }
        case (ast::expr_alt(?input, ?arms)) {
            check_alt(*cx, input, arms, sc, v);
        }
        case (ast::expr_put(?val)) {
            alt (val) {
                case (some(?ex)) {
                    auto root = expr_root(*cx, ex, false);
                    if (mut_field(root.ds)) {
                        cx.tcx.sess.span_fatal(ex.span,
                                             "result of put must be" +
                                                 " immutably rooted");
                    }
                    visit_expr(cx, ex, sc, v);
                }
                case (_) { }
            }
        }
        case (ast::expr_for_each(?decl, ?call, ?block)) {
            check_for_each(*cx, decl, call, block, sc, v);
        }
        case (ast::expr_for(?decl, ?seq, ?block)) {
            check_for(*cx, decl, seq, block, sc, v);
        }
        case (ast::expr_path(?pt)) {
            check_var(*cx, ex, pt, ex.id, false, sc);
            handled = false;
        }
        case (ast::expr_move(?dest, ?src)) {
            check_assign(cx, dest, src, sc, v);
        }
        case (ast::expr_assign(?dest, ?src)) {
            check_assign(cx, dest, src, sc, v);
        }
        case (ast::expr_assign_op(_, ?dest, ?src)) {
            check_assign(cx, dest, src, sc, v);
        }
        case (_) { handled = false; }
    }
    if (!handled) { visit::visit_expr(ex, sc, v); }
}

fn check_call(&ctx cx, &@ast::expr f, &vec[@ast::expr] args, &scope sc) ->
   rec(vec[node_id] root_vars, vec[ty::t] unsafe_ts) {
    auto fty = ty::expr_ty(*cx.tcx, f);
    auto arg_ts = fty_args(cx, fty);
    let vec[node_id] roots = [];
    let vec[tup(uint, node_id)] mut_roots = [];
    let vec[ty::t] unsafe_ts = [];
    let vec[uint] unsafe_t_offsets = [];
    auto i = 0u;
    for (ty::arg arg_t in arg_ts) {
        if (arg_t.mode != ty::mo_val) {
            auto arg = args.(i);
            auto root = expr_root(cx, arg, false);
            if (arg_t.mode == ty::mo_alias(true)) {
                alt (path_def_id(cx, arg)) {
                    case (some(?did)) {
                        vec::push(mut_roots, tup(i, did._1));
                    }
                    case (_) {
                        if (!mut_field(root.ds)) {
                            auto m =
                                "passing a temporary value or \
                                 immutable field by mutable alias";
                            cx.tcx.sess.span_fatal(arg.span, m);
                        }
                    }
                }
            }
            alt (path_def_id(cx, root.ex)) {
                case (some(?did)) { vec::push(roots, did._1); }
                case (_) { }
            }
            alt (inner_mut(root.ds)) {
                case (some(?t)) {
                    vec::push(unsafe_ts, t);
                    vec::push(unsafe_t_offsets, i);
                }
                case (_) { }
            }
        }
        i += 1u;
    }
    if (vec::len(unsafe_ts) > 0u) {
        alt (f.node) {
            case (ast::expr_path(_)) {
                if (def_is_local(cx.tcx.def_map.get(f.id), true)) {
                    cx.tcx.sess.span_fatal(f.span,
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
                cx.tcx.sess.span_fatal(args.(i).span,
                                     #fmt("argument %u may alias with \
                     argument %u, which is not immutably rooted",
                                          i, offset));
            }
            i += 1u;
        }
    }
    // Ensure we're not passing a root by mutable alias.

    for (tup(uint, node_id) root in mut_roots) {
        auto mut_alias_to_root = vec::count(root._1, roots) > 1u;
        for (restrict r in sc) {
            if (vec::member(root._1, r.root_vars)) {
                mut_alias_to_root = true;
            }
        }
        if (mut_alias_to_root) {
            cx.tcx.sess.span_fatal(args.(root._0).span,
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
                                cx.tcx.sess.span_fatal(args.(i).span,
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
                cx.tcx.sess.span_fatal(args.(i).span,
                                      "can not pass a local value by \
                                     alias to a tail call");
            }
        }
        i += 1u;
    }
}

fn check_alt(&ctx cx, &@ast::expr input, &vec[ast::arm] arms, &scope sc,
             &vt[scope] v) {
    visit::visit_expr(input, sc, v);
    auto root = expr_root(cx, input, true);
    auto roots =
        alt (path_def_id(cx, root.ex)) {
            case (some(?did)) { [did._1] }
            case (_) { [] }
        };
    let vec[ty::t] forbidden_tp =
        alt (inner_mut(root.ds)) { case (some(?t)) { [t] } case (_) { [] } };
    for (ast::arm a in arms) {
        auto dnums = arm_defnums(a);
        auto new_sc = sc;
        if (vec::len(dnums) > 0u) {
            new_sc =
                sc +
                    [@rec(root_vars=roots,
                          block_defnum=dnums.(0),
                          bindings=dnums,
                          tys=forbidden_tp,
                          depends_on=deps(sc, roots),
                          mutable ok=valid)];
        }
        visit::visit_arm(a, new_sc, v);
    }
}

fn arm_defnums(&ast::arm arm) -> vec[node_id] {
    auto dnums = [];
    fn walk_pat(&mutable vec[node_id] found, &@ast::pat p) {
        alt (p.node) {
            case (ast::pat_bind(_, ?id)) { vec::push(found, id); }
            case (ast::pat_tag(_, ?children, _)) {
                for (@ast::pat child in children) { walk_pat(found, child); }
            }
            case (_) { }
        }
    }
    walk_pat(dnums, arm.pat);
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
                     bindings=[defnum],
                     tys=data.unsafe_ts,
                     depends_on=deps(sc, data.root_vars),
                     mutable ok=valid);
            visit::visit_block(block, sc + [new_sc], v);
        }
    }
}

fn check_for(&ctx cx, &@ast::local local, &@ast::expr seq, &ast::block block,
             &scope sc, &vt[scope] v) {
    visit::visit_expr(seq, sc, v);
    auto defnum = local.node.id;
    auto root = expr_root(cx, seq, false);
    auto root_def =
        alt (path_def_id(cx, root.ex)) {
            case (some(?did)) { [did._1] }
            case (_) { [] }
        };
    auto unsafe =
        alt (inner_mut(root.ds)) { case (some(?t)) { [t] } case (_) { [] } };
    // If this is a mutable vector, don't allow it to be touched.

    auto seq_t = ty::expr_ty(*cx.tcx, seq);
    alt (ty::struct(*cx.tcx, seq_t)) {
        case (ty::ty_vec(?mt)) {
            if (mt.mut != ast::imm) { unsafe = [seq_t]; }
        }
        case (ty::ty_str) { /* no-op */ }
        case (ty::ty_ivec(?mt)) {
            if (mt.mut != ast::imm) { unsafe = [seq_t]; }
        }
        case (ty::ty_istr) { /* no-op */ }
        case (_) {
            cx.tcx.sess.span_unimpl(seq.span, "unknown seq type " +
                                    pretty::ppaux::ty_to_str(*cx.tcx, seq_t));
        }
    }
    auto new_sc =
        @rec(root_vars=root_def,
             block_defnum=defnum,
             bindings=[defnum],
             tys=unsafe,
             depends_on=deps(sc, root_def),
             mutable ok=valid);
    visit::visit_block(block, sc + [new_sc], v);
}

fn check_var(&ctx cx, &@ast::expr ex, &ast::path p, ast::node_id id,
             bool assign, &scope sc) {
    auto def = cx.tcx.def_map.get(id);
    if (!def_is_local(def, true)) { ret; }
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
        } else if (vec::member(my_defnum, r.bindings)) {
            test_scope(cx, sc, r, p);
        }
    }
}


// FIXME does not catch assigning to immutable object fields yet
fn check_assign(&@ctx cx, &@ast::expr dest, &@ast::expr src, &scope sc,
                &vt[scope] v) {
    visit_expr(cx, src, sc, v);
    alt (dest.node) {
        case (ast::expr_path(?p)) {
            auto dnum = ast::def_id_of_def(cx.tcx.def_map.get(dest.id))._1;
            if (is_immutable_alias(cx, sc, dnum)) {
                cx.tcx.sess.span_fatal(dest.span,
                                     "assigning to immutable alias");
            } else if (is_immutable_objfield(cx, dnum)) {
                cx.tcx.sess.span_fatal(dest.span,
                                     "assigning to immutable obj field");
            }
            auto var_t = ty::expr_ty(*cx.tcx, dest);
            for (restrict r in sc) {
                if (vec::member(dnum, r.root_vars)) {
                    r.ok = overwritten(dest.span, p);
                }
            }
            check_var(*cx, dest, p, dest.id, true, sc);
        }
        case (_) {
            auto root = expr_root(*cx, dest, false);
            if (vec::len(root.ds) == 0u) {
                cx.tcx.sess.span_fatal(dest.span, "assignment to non-lvalue");
            } else if (!root.ds.(0).mut) {
                auto name =
                    alt (root.ds.(0).kind) {
                        case (unbox) { "box" }
                        case (field) { "field" }
                        case (index) { "vec content" }
                    };
                cx.tcx.sess.span_fatal(dest.span,
                                     "assignment to immutable " + name);
            }
            visit_expr(cx, dest, sc, v);
        }
    }
}

fn is_immutable_alias(&@ctx cx, &scope sc, node_id dnum) -> bool {
    alt (cx.local_map.find(dnum)) {
        case (some(arg(ast::alias(false)))) { ret true; }
        case (_) { }
    }
    for (restrict r in sc) {
        if (vec::member(dnum, r.bindings)) { ret true; }
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
        cx.tcx.sess.span_fatal(msg._0,
                             msg._1 + " will invalidate alias " +
                                 ast::path_name(p) + ", which is still used");
    }
}

fn deps(&scope sc, vec[node_id] roots) -> vec[uint] {
    auto i = 0u;
    auto result = [];
    for (restrict r in sc) {
        for (node_id dn in roots) {
            if (vec::member(dn, r.bindings)) { vec::push(result, i); }
        }
        i += 1u;
    }
    ret result;
}

tag deref_t { unbox; field; index; }

type deref = rec(bool mut, deref_t kind, ty::t outer_t);


// Finds the root (the thing that is dereferenced) for the given expr, and a
// vec of dereferences that were used on this root. Note that, in this vec,
// the inner derefs come in front, so foo.bar.baz becomes rec(ex=foo,
// ds=[field(baz),field(bar)])
fn expr_root(&ctx cx, @ast::expr ex, bool autoderef) ->
   rec(@ast::expr ex, vec[deref] ds) {
    fn maybe_auto_unbox(&ctx cx, &ty::t t) ->
       rec(ty::t t, option::t[deref] d) {
        alt (ty::struct(*cx.tcx, t)) {
            case (ty::ty_box(?mt)) {
                ret rec(t=mt.ty,
                        d=some(rec(mut=mt.mut != ast::imm,
                                   kind=unbox,
                                   outer_t=t)));
            }
            case (_) { ret rec(t=t, d=none); }
        }
    }
    fn maybe_push_auto_unbox(&option::t[deref] d, &mutable vec[deref] ds) {
        alt (d) { case (some(?d)) { vec::push(ds, d); } case (none) { } }
    }
    let vec[deref] ds = [];
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
                vec::push(ds, rec(mut=mut, kind=field, outer_t=auto_unbox.t));
                maybe_push_auto_unbox(auto_unbox.d, ds);
                ex = base;
            }
            case (ast::expr_index(?base, _)) {
                auto auto_unbox =
                    maybe_auto_unbox(cx, ty::expr_ty(*cx.tcx, base));
                alt (ty::struct(*cx.tcx, auto_unbox.t)) {
                    case (ty::ty_vec(?mt)) {
                        vec::push(ds,
                                  rec(mut=mt.mut != ast::imm,
                                      kind=index,
                                      outer_t=auto_unbox.t));
                    }
                    case (ty::ty_ivec(?mt)) {
                        vec::push(ds,
                                  rec(mut=mt.mut != ast::imm,
                                      kind=index,
                                      outer_t=auto_unbox.t));
                    }
                }
                maybe_push_auto_unbox(auto_unbox.d, ds);
                ex = base;
            }
            case (ast::expr_unary(?op, ?base)) {
                if (op == ast::deref) {
                    auto base_t = ty::expr_ty(*cx.tcx, base);
                    alt (ty::struct(*cx.tcx, base_t)) {
                        case (ty::ty_box(?mt)) {
                            vec::push(ds, rec(mut=mt.mut != ast::imm,
                                              kind=unbox,
                                              outer_t=base_t));
                        }
                        case (ty::ty_res(_, ?inner)) {
                            vec::push(ds, rec(mut=false,
                                              kind=unbox,
                                              outer_t=base_t));
                        }
                    }
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
    ret rec(ex=ex, ds=ds);
}

fn mut_field(&vec[deref] ds) -> bool {
    for (deref d in ds) { if (d.mut) { ret true; } }
    ret false;
}

fn inner_mut(&vec[deref] ds) -> option::t[ty::t] {
    for (deref d in ds) { if (d.mut) { ret some(d.outer_t); } }
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
            case (
                 // These may contain anything.
                 ty::ty_fn(_, _, _, _, _)) {
                ret true;
            }
            case (ty::ty_obj(_)) { ret true; }
            case (
                 // A type param may include everything, but can only be
                 // treated as opaque downstream, and is thus safe unless we
                 // saw mutable fields, in which case the whole thing can be
                 // overwritten.
                 ty::ty_param(_)) {
                ret mut;
            }
            case (_) { ret false; }
        }
    }
    ret helper(*cx.tcx, needle, haystack, mut);
}

fn def_is_local(&ast::def d, bool objfields_count) -> bool {
    ret alt (d) {
            case (ast::def_local(_)) { true }
            case (ast::def_arg(_)) { true }
            case (ast::def_obj_field(_)) { objfields_count }
            case (ast::def_binding(_)) { true }
            case (_) { false }
        };
}

fn fty_args(&ctx cx, ty::t fty) -> vec[ty::arg] {
    ret alt (ty::struct(*cx.tcx, fty)) {
            case (ty::ty_fn(_, ?args, _, _, _)) { args }
            case (ty::ty_native_fn(_, ?args, _)) { args }
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
