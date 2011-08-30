
import syntax::ast;
import syntax::ast_util;
import ast::ident;
import ast::fn_ident;
import ast::node_id;
import ast::def_id;
import syntax::codemap::span;
import syntax::visit;
import visit::vt;
import std::vec;
import std::str;
import std::istr;
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
    @{root_vars: [node_id],
      local_id: uint,
      bindings: [node_id],
      tys: [ty::t],
      depends_on: [uint],
      mutable ok: valid};

type scope = @[restrict];

tag local_info {
    arg(ast::mode);
    objfield(ast::mutability);
    local(uint);
}

type mut_map = std::map::hashmap<node_id, ()>;
type ctx = {tcx: ty::ctxt,
            local_map: std::map::hashmap<node_id, local_info>,
            mutable next_local: uint,
            mut_map: mut_map};

fn check_crate(tcx: ty::ctxt, crate: &@ast::crate) -> mut_map {
    // Stores information about object fields and function
    // arguments that's otherwise not easily available.
    let cx = @{tcx: tcx,
               local_map: std::map::new_int_hash(),
               mutable next_local: 0u,
               mut_map: std::map::new_int_hash()};
    let v = @{visit_fn: bind visit_fn(cx, _, _, _, _, _, _, _),
              visit_item: bind visit_item(cx, _, _, _),
              visit_expr: bind visit_expr(cx, _, _, _),
              visit_decl: bind visit_decl(cx, _, _, _)
              with *visit::default_visitor::<scope>()};
    visit::visit_crate(*crate, @[], visit::mk_vt(v));
    tcx.sess.abort_if_errors();
    ret cx.mut_map;
}

fn visit_fn(cx: &@ctx, f: &ast::_fn, _tp: &[ast::ty_param], _sp: &span,
            _name: &fn_ident, id: ast::node_id, sc: &scope, v: &vt<scope>) {
    visit::visit_fn_decl(f.decl, sc, v);
    for arg_: ast::arg in f.decl.inputs {
        cx.local_map.insert(arg_.id, arg(arg_.mode));
    }
    let scope =
        alt f.proto {

          // Blocks need to obey any restrictions from the enclosing scope.
          ast::proto_block. {
            sc
          }

          // Closures need to prohibit writing to any of the upvars.
          // This doesn't seem like a particularly clean way to do this.
          ast::proto_closure. {
            let dnums = [];
            for each nid in freevars::get_freevar_defs(cx.tcx, id).keys() {
                dnums += [nid];
            };
            @[
              // I'm not sure if there is anything sensical to put here
              @{root_vars: [],
                local_id: cx.next_local,
                bindings: dnums,
                tys: [],
                depends_on: [],
                mutable ok: valid}]
          }

          // Non capturing functions start out fresh.
          _ {
            @[]
          }
        };

    v.visit_block(f.body, scope, v);
}

fn visit_item(cx: &@ctx, i: &@ast::item, sc: &scope, v: &vt<scope>) {
    alt i.node {
      ast::item_obj(o, _, _) {
        for f: ast::obj_field in o.fields {
            cx.local_map.insert(f.id, objfield(f.mut));
        }
      }
      _ { }
    }
    visit::visit_item(i, sc, v);
}

fn visit_expr(cx: &@ctx, ex: &@ast::expr, sc: &scope, v: &vt<scope>) {
    let handled = true;
    alt ex.node {
      ast::expr_call(f, args) {
        check_call(*cx, f, args, sc);
        handled = false;
      }
      ast::expr_be(cl) {
        check_tail_call(*cx, cl);
        visit::visit_expr(cl, sc, v);
      }
      ast::expr_alt(input, arms) { check_alt(*cx, input, arms, sc, v); }
      ast::expr_put(val) {
        alt val {
          some(ex) {
            let root = expr_root(*cx, ex, false);
            if mut_field(root.ds) {
                cx.tcx.sess.span_err(ex.span,
                                     ~"result of put must be" +
                                         ~" immutably rooted");
            }
            visit_expr(cx, ex, sc, v);
          }
          _ { }
        }
      }
      ast::expr_for_each(decl, call, blk) {
        check_for_each(*cx, decl, call, blk, sc, v);
      }
      ast::expr_for(decl, seq, blk) { check_for(*cx, decl, seq, blk, sc, v); }
      ast::expr_path(pt) {
        check_var(*cx, ex, pt, ex.id, false, sc);
        handled = false;
      }
      ast::expr_swap(lhs, rhs) {
        check_lval(cx, lhs, sc, v);
        check_lval(cx, rhs, sc, v);
        handled = false;
      }
      ast::expr_move(dest, src) {
        check_assign(cx, dest, src, sc, v);
        check_move_rhs(cx, src, sc, v);
      }
      ast::expr_assign(dest, src) | ast::expr_assign_op(_, dest, src) {
        check_assign(cx, dest, src, sc, v);
      }
      _ { handled = false; }
    }
    if !handled { visit::visit_expr(ex, sc, v); }
}

fn register_locals(cx: &ctx, pat: &@ast::pat) {
    for each pat in ast_util::pat_bindings(pat) {
        cx.local_map.insert(pat.id, local(cx.next_local));
        cx.next_local += 1u;
    }
}

fn visit_decl(cx: &@ctx, d: &@ast::decl, sc: &scope, v: &vt<scope>) {
    visit::visit_decl(d, sc, v);
    alt d.node {
      ast::decl_local(locs) {
        for loc: @ast::local in locs {
            alt loc.node.init {
              some(init) {
                if init.op == ast::init_move {
                    check_move_rhs(cx, init.expr, sc, v);
                }
              }
              none. { }
            }
            register_locals(*cx, loc.node.pat);
        }
      }
      _ { }
    }
}

fn check_call(cx: &ctx, f: &@ast::expr, args: &[@ast::expr], sc: &scope) ->
   {root_vars: [node_id], unsafe_ts: [ty::t]} {
    let fty = ty::expr_ty(cx.tcx, f);
    let arg_ts = fty_args(cx, fty);
    let roots: [node_id] = [];
    let mut_roots: [{arg: uint, node: node_id}] = [];
    let unsafe_ts: [ty::t] = [];
    let unsafe_t_offsets: [uint] = [];
    let i = 0u;
    for arg_t: ty::arg in arg_ts {
        if arg_t.mode != ty::mo_val {
            let arg = args[i];
            let root = expr_root(cx, arg, false);
            if arg_t.mode == ty::mo_alias(true) {
                alt path_def(cx, arg) {
                  some(def) {
                    let dnum = ast_util::def_id_of_def(def).node;
                    if def_is_local(def, true) {
                        if is_immutable_alias(cx, sc, dnum) {
                            cx.tcx.sess.span_err(
                                arg.span,
                                ~"passing an immutable alias \
                                 by mutable alias");
                        } else if is_immutable_objfield(cx, dnum) {
                            cx.tcx.sess.span_err(
                                arg.span,
                                ~"passing an immutable object \
                                 field by mutable alias");
                        }
                        cx.mut_map.insert(dnum, ());
                    } else {
                        cx.tcx.sess.span_err(
                            arg.span,
                            ~"passing a static item by mutable alias");
                    }
                    mut_roots += [{arg: i, node: dnum}];
                  }
                  _ {
                    if !mut_field(root.ds) {
                        let m =
                            ~"passing a temporary value or \
                                 immutable field by mutable alias";
                        cx.tcx.sess.span_err(arg.span, m);
                    }
                  }
                }
            }
            alt path_def_id(cx, root.ex) {
              some(did) { roots += [did.node]; }
              _ { }
            }
            alt inner_mut(root.ds) {
              some(t) { unsafe_ts += [t]; unsafe_t_offsets += [i]; }
              _ { }
            }
        }
        i += 1u;
    }
    if vec::len(unsafe_ts) > 0u {
        alt f.node {
          ast::expr_path(_) {
            if def_is_local(cx.tcx.def_map.get(f.id), true) {
                cx.tcx.sess.span_err(f.span, istr::from_estr(
                                     #fmt["function may alias with \
                         argument %u, which is not immutably rooted",
                                          unsafe_t_offsets[0]]));
            }
          }
          _ { }
        }
    }
    let j = 0u;
    for unsafe: ty::t in unsafe_ts {
        let offset = unsafe_t_offsets[j];
        j += 1u;
        let i = 0u;
        for arg_t: ty::arg in arg_ts {
            let mut_alias = arg_t.mode == ty::mo_alias(true);
            if i != offset &&
                   ty_can_unsafely_include(cx, unsafe, arg_t.ty, mut_alias) {
                cx.tcx.sess.span_err(args[i].span, istr::from_estr(
                                     #fmt["argument %u may alias with \
                     argument %u, which is not immutably rooted",
                                          i, offset]));
            }
            i += 1u;
        }
    }
    // Ensure we're not passing a root by mutable alias.

    for root: {arg: uint, node: node_id} in mut_roots {
        let mut_alias_to_root = false;
        let mut_alias_to_root_count = 0u;
        for r: node_id in roots {
            if root.node == r {
                mut_alias_to_root_count += 1u;
                if mut_alias_to_root_count > 1u {
                    mut_alias_to_root = true;
                    break;
                }
            }
        }


        if mut_alias_to_root {
            cx.tcx.sess.span_err(args[root.arg].span,
                                 ~"passing a mutable alias to a \
                 variable that roots another alias");
        }
    }
    ret {root_vars: roots, unsafe_ts: unsafe_ts};
}

fn check_tail_call(cx: &ctx, call: &@ast::expr) {
    let args;
    let f = alt call.node { ast::expr_call(f, args_) { args = args_; f } };
    let i = 0u;
    for arg_t: ty::arg in fty_args(cx, ty::expr_ty(cx.tcx, f)) {
        if arg_t.mode != ty::mo_val {
            let mut_a = arg_t.mode == ty::mo_alias(true);
            let ok = true;
            alt args[i].node {
              ast::expr_path(_) {
                let def = cx.tcx.def_map.get(args[i].id);
                let dnum = ast_util::def_id_of_def(def).node;
                alt cx.local_map.find(dnum) {
                  some(arg(ast::alias(mut))) {
                    if mut_a && !mut {
                        cx.tcx.sess.span_err(args[i].span,
                                             ~"passing an immutable \
                                     alias by mutable alias");
                    }
                  }
                  _ { ok = !def_is_local(def, false); }
                }
              }
              _ { ok = false; }
            }
            if !ok {
                cx.tcx.sess.span_err(args[i].span,
                                     ~"can not pass a local value by \
                                     alias to a tail call");
            }
        }
        i += 1u;
    }
}

fn check_alt(cx: &ctx, input: &@ast::expr, arms: &[ast::arm], sc: &scope,
             v: &vt<scope>) {
    v.visit_expr(input, sc, v);
    let root = expr_root(cx, input, true);
    let roots =
        alt path_def_id(cx, root.ex) { some(did) { [did.node] } _ { [] } };
    let forbidden_tp: [ty::t] =
        alt inner_mut(root.ds) { some(t) { [t] } _ { [] } };
    for a: ast::arm in arms {
        let dnums = ast_util::pat_binding_ids(a.pats[0]);
        let new_sc = sc;
        if vec::len(dnums) > 0u {
            new_sc = @(*sc + [@{root_vars: roots,
                                local_id: cx.next_local,
                                bindings: dnums,
                                tys: forbidden_tp,
                                depends_on: deps(sc, roots),
                                mutable ok: valid}]);
        }
        register_locals(cx, a.pats[0]);
        visit::visit_arm(a, new_sc, v);
    }
}

fn check_for_each(cx: &ctx, local: &@ast::local, call: &@ast::expr,
                  blk: &ast::blk, sc: &scope, v: &vt<scope>) {
    v.visit_expr(call, sc, v);
    alt call.node {
      ast::expr_call(f, args) {
        let data = check_call(cx, f, args, sc);
        let bindings = ast_util::pat_binding_ids(local.node.pat);
        let new_sc =
            @{root_vars: data.root_vars,
              local_id: cx.next_local,
              bindings: bindings,
              tys: data.unsafe_ts,
              depends_on: deps(sc, data.root_vars),
              mutable ok: valid};
        register_locals(cx, local.node.pat);
        visit::visit_block(blk, @(*sc + [new_sc]), v);
      }
    }
}

fn check_for(cx: &ctx, local: &@ast::local, seq: &@ast::expr, blk: &ast::blk,
             sc: &scope, v: &vt<scope>) {
    v.visit_expr(seq, sc, v);
    let root = expr_root(cx, seq, false);
    let root_def =
        alt path_def_id(cx, root.ex) { some(did) { [did.node] } _ { [] } };
    let unsafe = alt inner_mut(root.ds) { some(t) { [t] } _ { [] } };

    // If this is a mutable vector, don't allow it to be touched.
    let seq_t = ty::expr_ty(cx.tcx, seq);
    alt ty::struct(cx.tcx, seq_t) {
      ty::ty_vec(mt) { if mt.mut != ast::imm { unsafe = [seq_t]; } }
      ty::ty_str. | ty::ty_istr. {/* no-op */ }
      _ {
        cx.tcx.sess.span_unimpl(
            seq.span,
            ~"unknown seq type " +
            util::ppaux::ty_to_str(cx.tcx, seq_t));
      }
    }
    let bindings = ast_util::pat_binding_ids(local.node.pat);
    let new_sc =
        @{root_vars: root_def,
          local_id: cx.next_local,
          bindings: bindings,
          tys: unsafe,
          depends_on: deps(sc, root_def),
          mutable ok: valid};
    register_locals(cx, local.node.pat);
    visit::visit_block(blk, @(*sc + [new_sc]), v);
}

fn check_var(cx: &ctx, ex: &@ast::expr, p: &ast::path, id: ast::node_id,
             assign: bool, sc: &scope) {
    let def = cx.tcx.def_map.get(id);
    if !def_is_local(def, true) { ret; }
    let my_defnum = ast_util::def_id_of_def(def).node;
    let my_local_id = alt cx.local_map.find(my_defnum) {
      some(local(id)) { id }
      _ { 0u }
    };
    let var_t = ty::expr_ty(cx.tcx, ex);
    for r: restrict in *sc {
        // excludes variables introduced since the alias was made
        if my_local_id < r.local_id {
            for t: ty::t in r.tys {
                if ty_can_unsafely_include(cx, t, var_t, assign) {
                    r.ok = val_taken(ex.span, p);
                }
            }
        } else if vec::member(my_defnum, r.bindings) {
            test_scope(cx, sc, r, p);
        }
    }
}

fn check_lval(cx: &@ctx, dest: &@ast::expr, sc: &scope, v: &vt<scope>) {
    alt dest.node {
      ast::expr_path(p) {
        let dnum = ast_util::def_id_of_def(cx.tcx.def_map.get(dest.id)).node;
        cx.mut_map.insert(dnum, ());
        if is_immutable_alias(*cx, sc, dnum) {
            cx.tcx.sess.span_err(dest.span, ~"assigning to immutable alias");
        } else if is_immutable_objfield(*cx, dnum) {
            cx.tcx.sess.span_err(dest.span,
                                 ~"assigning to immutable obj field");
        }
        for r: restrict in *sc {
            if vec::member(dnum, r.root_vars) {
                r.ok = overwritten(dest.span, p);
            }
        }
      }
      _ {
        let root = expr_root(*cx, dest, false);
        if vec::len(*root.ds) == 0u {
            cx.tcx.sess.span_err(dest.span, ~"assignment to non-lvalue");
        } else if !root.ds[0].mut {
            let name =
                alt root.ds[0].kind {
                  unbox. { ~"box" }
                  field. { ~"field" }
                  index. { ~"vec content" }
                };
            cx.tcx.sess.span_err(dest.span,
                                 ~"assignment to immutable " + name);
        }
        visit_expr(cx, dest, sc, v);
      }
    }
}

fn check_move_rhs(cx: &@ctx, src: &@ast::expr, sc: &scope, v: &vt<scope>) {
    alt src.node {
      ast::expr_path(p) {
        alt cx.tcx.def_map.get(src.id) {
          ast::def_obj_field(_) {
            cx.tcx.sess.span_err(src.span,
                                 ~"may not move out of an obj field");
          }
          _ { }
        }
        check_lval(cx, src, sc, v);
      }
      _ {
        let root = expr_root(*cx, src, false);

        // Not a path and no-derefs means this is a temporary.
        if vec::len(*root.ds) != 0u {
            cx.tcx.sess.span_err(src.span, ~"moving out of a data structure");
        }
      }
    }
}

fn check_assign(cx: &@ctx, dest: &@ast::expr, src: &@ast::expr, sc: &scope,
                v: &vt<scope>) {
    visit_expr(cx, src, sc, v);
    check_lval(cx, dest, sc, v);
}


fn is_immutable_alias(cx: &ctx, sc: &scope, dnum: node_id) -> bool {
    alt cx.local_map.find(dnum) {
      some(arg(ast::alias(false))) { ret true; }
      _ { }
    }
    for r: restrict in *sc { if vec::member(dnum, r.bindings) { ret true; } }
    ret false;
}

fn is_immutable_objfield(cx: &ctx, dnum: node_id) -> bool {
    ret cx.local_map.find(dnum) == some(objfield(ast::imm));
}

fn test_scope(cx: &ctx, sc: &scope, r: &restrict, p: &ast::path) {
    let prob = r.ok;
    for dep: uint in r.depends_on {
        if prob != valid { break; }
        prob = sc[dep].ok;
    }
    if prob != valid {
        let msg =
            alt prob {
              overwritten(sp, wpt) {
                {span: sp, msg: ~"overwriting " +
                    ast_util::path_name(wpt)}
              }
              val_taken(sp, vpt) {
                {span: sp,
                 msg: ~"taking the value of " +
                     ast_util::path_name(vpt)}
              }
            };
        cx.tcx.sess.span_err(msg.span,
                             msg.msg + ~" will invalidate alias " +
                             ast_util::path_name(p) +
                             ~", which is still used");
    }
}

fn deps(sc: &scope, roots: &[node_id]) -> [uint] {
    let i = 0u;
    let result = [];
    for r: restrict in *sc {
        for dn: node_id in roots {
            if vec::member(dn, r.bindings) { result += [i]; }
        }
        i += 1u;
    }
    ret result;
}

tag deref_t { unbox; field; index; }

type deref = @{mut: bool, kind: deref_t, outer_t: ty::t};


// Finds the root (the thing that is dereferenced) for the given expr, and a
// vec of dereferences that were used on this root. Note that, in this vec,
// the inner derefs come in front, so foo.bar.baz becomes rec(ex=foo,
// ds=[field(baz),field(bar)])
fn expr_root(cx: &ctx, ex: @ast::expr, autoderef: bool) ->
   {ex: @ast::expr, ds: @[deref]} {
    fn maybe_auto_unbox(cx: &ctx, t: ty::t) -> {t: ty::t, ds: [deref]} {
        let ds = [];
        while true {
            alt ty::struct(cx.tcx, t) {
              ty::ty_box(mt) {
                ds += [@{mut: mt.mut != ast::imm, kind: unbox, outer_t: t}];
                t = mt.ty;
              }
              ty::ty_uniq(mt) {
                ds += [@{mut: false, kind: unbox, outer_t: t}];
              }
              ty::ty_res(_, inner, tps) {
                ds += [@{mut: false, kind: unbox, outer_t: t}];
                t = ty::substitute_type_params(cx.tcx, tps, inner);
              }
              ty::ty_tag(did, tps) {
                let variants = ty::tag_variants(cx.tcx, did);
                if vec::len(variants) != 1u ||
                       vec::len(variants[0].args) != 1u {
                    break;
                }
                ds += [@{mut: false, kind: unbox, outer_t: t}];
                t =
                    ty::substitute_type_params(cx.tcx, tps,
                                               variants[0].args[0]);
              }
              _ { break; }
            }
        }
        ret {t: t, ds: ds};
    }
    let ds: [deref] = [];
    while true {
        alt { ex.node } {
          ast::expr_field(base, ident) {
            let auto_unbox = maybe_auto_unbox(cx, ty::expr_ty(cx.tcx, base));
            let mut = false;
            alt ty::struct(cx.tcx, auto_unbox.t) {
              ty::ty_rec(fields) {
                for fld: ty::field in fields {
                    if istr::eq(ident, fld.ident) {
                        mut = fld.mt.mut != ast::imm;
                        break;
                    }
                }
              }
              ty::ty_obj(_) { }
            }
            ds += [@{mut: mut, kind: field, outer_t: auto_unbox.t}];
            ds += auto_unbox.ds;
            ex = base;
          }
          ast::expr_index(base, _) {
            let auto_unbox = maybe_auto_unbox(cx, ty::expr_ty(cx.tcx, base));
            alt ty::struct(cx.tcx, auto_unbox.t) {
              ty::ty_vec(mt) {
                ds +=
                    [@{mut: mt.mut != ast::imm,
                       kind: index,
                       outer_t: auto_unbox.t}];
              }
            }
            ds += auto_unbox.ds;
            ex = base;
          }
          ast::expr_unary(op, base) {
            if op == ast::deref {
                let base_t = ty::expr_ty(cx.tcx, base);
                let mut = false;
                alt ty::struct(cx.tcx, base_t) {
                  ty::ty_box(mt) { mut = mt.mut != ast::imm; }
                  ty::ty_uniq(_) { }
                  ty::ty_res(_, _, _) { }
                  ty::ty_tag(_, _) { }
                  ty::ty_ptr(mt) { mut = mt.mut != ast::imm; }
                }
                ds += [@{mut: mut, kind: unbox, outer_t: base_t}];
                ex = base;
            } else { break; }
          }
          _ { break; }
        }
    }
    if autoderef {
        let auto_unbox = maybe_auto_unbox(cx, ty::expr_ty(cx.tcx, ex));
        ds += auto_unbox.ds;
    }
    ret {ex: ex, ds: @ds};
}

fn mut_field(ds: &@[deref]) -> bool {
    for d: deref in *ds { if d.mut { ret true; } }
    ret false;
}

fn inner_mut(ds: &@[deref]) -> option::t<ty::t> {
    for d: deref in *ds { if d.mut { ret some(d.outer_t); } }
    ret none;
}

fn path_def(cx: &ctx, ex: &@ast::expr) -> option::t<ast::def> {
    ret alt ex.node {
          ast::expr_path(_) { some(cx.tcx.def_map.get(ex.id)) }
          _ { none }
        }
}

fn path_def_id(cx: &ctx, ex: &@ast::expr) -> option::t<ast::def_id> {
    alt ex.node {
      ast::expr_path(_) {
        ret some(ast_util::def_id_of_def(cx.tcx.def_map.get(ex.id)));
      }
      _ { ret none; }
    }
}

fn ty_can_unsafely_include(cx: &ctx, needle: ty::t, haystack: ty::t,
                           mut: bool) -> bool {
    fn get_mut(cur: bool, mt: &ty::mt) -> bool {
        ret cur || mt.mut != ast::imm;
    }
    fn helper(tcx: &ty::ctxt, needle: ty::t, haystack: ty::t, mut: bool) ->
       bool {
        if needle == haystack { ret true; }
        alt ty::struct(tcx, haystack) {
          ty::ty_tag(_, ts) {
            for t: ty::t in ts {
                if helper(tcx, needle, t, mut) { ret true; }
            }
            ret false;
          }
          ty::ty_box(mt) | ty::ty_ptr(mt) {
            ret helper(tcx, needle, mt.ty, get_mut(mut, mt));
          }
          ty::ty_uniq(t) { ret helper(tcx, needle, t, false); }
          ty::ty_rec(fields) {
            for f: ty::field in fields {
                if helper(tcx, needle, f.mt.ty, get_mut(mut, f.mt)) {
                    ret true;
                }
            }
            ret false;
          }
          ty::ty_tup(ts) {
            for t in ts { if helper(tcx, needle, t, mut) { ret true; } }
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
          ty::ty_param(_, _) {
            ret mut;
          }
          _ { ret false; }
        }
    }
    ret helper(cx.tcx, needle, haystack, mut);
}

fn def_is_local(d: &ast::def, objfields_count: bool) -> bool {
    ret alt d {
          ast::def_local(_) | ast::def_arg(_) | ast::def_binding(_) { true }
          ast::def_obj_field(_) { objfields_count }
          _ { false }
        };
}

fn fty_args(cx: &ctx, fty: ty::t) -> [ty::arg] {
    ret alt ty::struct(cx.tcx, ty::type_autoderef(cx.tcx, fty)) {
          ty::ty_fn(_, args, _, _, _) | ty::ty_native_fn(_, args, _) { args }
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
