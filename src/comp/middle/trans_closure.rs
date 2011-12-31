import syntax::ast;
import syntax::ast_util;
import lib::llvm::llvm;
import llvm::{ValueRef, TypeRef};
import trans_common::*;
import trans_build::*;
import trans::*;
import middle::freevars::{get_freevars, freevar_info};
import option::{some, none};
import back::abi;
import syntax::codemap::span;
import back::link::{
    mangle_internal_name_by_path,
    mangle_internal_name_by_path_and_seq};
import trans::{
    trans_shared_malloc,
    type_of_inner,
    size_of,
    node_id_type,
    INIT,
    trans_shared_free,
    drop_ty,
    new_sub_block_ctxt,
    load_if_immediate,
    dest
};

// ___Good to know (tm)__________________________________________________
//
// The layout of a closure environment in memory is
// roughly as follows:
//
// struct closure_box {
//    unsigned ref_count; // only used for sharid environments
//    struct closure {
//      type_desc *tydesc;         // descriptor for the "struct closure" type
//      type_desc *bound_tdescs[]; // bound descriptors
//      struct {
//          upvar1_t upvar1;
//          ...
//          upvarN_t upvarN;
//      } bound_data;
//   };
// };
//
// NB: this struct is defined in the code in trans_common::T_closure()
// and mk_closure_ty() below.  The former defines the LLVM version and
// the latter the Rust equivalent.  It occurs to me that these could
// perhaps be unified, but currently they are not.
//
// Note that the closure carries a type descriptor that describes
// itself.  Trippy.  This is needed because the precise types of the
// closed over data are lost in the closure type (`fn(T)->U`), so if
// we need to take/drop, we must know what data is in the upvars and
// so forth.
//
// The allocation strategy for this closure depends on the closure
// type.  For a sendfn, the closure (and the referenced type
// descriptors) will be allocated in the exchange heap.  For a fn, the
// closure is allocated in the task heap and is reference counted.
// For a block, the closure is allocated on the stack.  Note that in
// all cases we allocate space for a ref count just to make our lives
// easier when upcasting to block(T)->U, in the shape code, and so
// forth.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tag environment_value {
    // Evaluate expr and store result in env (used for bind).
    env_expr(@ast::expr);

    // Copy the value from this llvm ValueRef into the environment.
    env_copy(ValueRef, ty::t, lval_kind);

    // Move the value from this llvm ValueRef into the environment.
    env_move(ValueRef, ty::t, lval_kind);

    // Access by reference (used for blocks).
    env_ref(ValueRef, ty::t, lval_kind);
}

// Given a closure ty, emits a corresponding tuple ty
fn mk_closure_ty(tcx: ty::ctxt,
                 ck: ty::closure_kind,
                 ty_params: [fn_ty_param],
                 bound_data_ty: ty::t)
    -> ty::t {
    let tydesc_ty = alt ck {
      ty::closure_block. | ty::closure_shared. { ty::mk_type(tcx) }
      ty::closure_send. { ty::mk_send_type(tcx) }
    };
    let param_ptrs = [];
    for tp in ty_params {
        param_ptrs += [tydesc_ty];
        option::may(tp.dicts) {|dicts|
            for dict in dicts { param_ptrs += [tydesc_ty]; }
        }
    }
    ty::mk_tup(tcx, [tydesc_ty, ty::mk_tup(tcx, param_ptrs), bound_data_ty])
}

fn shared_opaque_closure_box_ty(tcx: ty::ctxt) -> ty::t {
    let opaque_closure_ty = ty::mk_opaque_closure(tcx);
    ret ty::mk_imm_box(tcx, opaque_closure_ty);
}

fn send_opaque_closure_box_ty(tcx: ty::ctxt) -> ty::t {
    let opaque_closure_ty = ty::mk_opaque_closure(tcx);
    let tup_ty = ty::mk_tup(tcx, [ty::mk_int(tcx), opaque_closure_ty]);
    ret ty::mk_uniq(tcx, {ty: tup_ty, mut: ast::imm});
}

type closure_result = {
    llbox: ValueRef,  // llvalue of boxed environment
    box_ty: ty::t,    // type of boxed environment
    bcx: @block_ctxt  // final bcx
};

// Given a block context and a list of tydescs and values to bind
// construct a closure out of them. If copying is true, it is a
// heap allocated closure that copies the upvars into environment.
// Otherwise, it is stack allocated and copies pointers to the upvars.
fn store_environment(
    bcx: @block_ctxt, lltyparams: [fn_ty_param],
    bound_values: [environment_value],
    ck: ty::closure_kind)
    -> closure_result {

    fn dummy_environment_box(bcx: @block_ctxt, r: result)
        -> (@block_ctxt, ValueRef, ValueRef) {
        // Prevent glue from trying to free this.
        let ccx = bcx_ccx(bcx);
        let ref_cnt = GEPi(bcx, r.val, [0, abi::box_rc_field_refcnt]);
        Store(r.bcx, C_int(ccx, 2), ref_cnt);
        let closure = GEPi(r.bcx, r.val, [0, abi::box_rc_field_body]);
        (r.bcx, closure, r.val)
    }

    fn maybe_clone_tydesc(bcx: @block_ctxt,
                          ck: ty::closure_kind,
                          td: ValueRef) -> ValueRef {
        ret alt ck {
          ty::closure_block. | ty::closure_shared. {
            td
          }
          ty::closure_send. {
            Call(bcx, bcx_ccx(bcx).upcalls.create_shared_type_desc, [td])
          }
        };
    }

    //let ccx = bcx_ccx(bcx);
    let tcx = bcx_tcx(bcx);

    // First, synthesize a tuple type containing the types of all the
    // bound expressions.
    // bindings_ty = [bound_ty1, bound_ty2, ...]
    let bound_tys = [];
    for bv in bound_values {
        bound_tys += [alt bv {
            env_copy(_, t, _) { t }
            env_move(_, t, _) { t }
            env_ref(_, t, _) { t }
            env_expr(e) { ty::expr_ty(tcx, e) }
        }];
    }
    let bound_data_ty = ty::mk_tup(tcx, bound_tys);
    let closure_ty =
        mk_closure_ty(tcx, ck, lltyparams, bound_data_ty);

    let temp_cleanups = [];

    // Allocate a box that can hold something closure-sized.
    //
    // For now, no matter what kind of closure we have, we always allocate
    // space for a ref cnt in the closure.  If the closure is a block or
    // unique closure, this ref count isn't really used: we initialize it to 2
    // so that it will never drop to zero.  This is a hack and could go away
    // but then we'd have to modify the code to do the right thing when
    // casting from a shared closure to a block.
    let (bcx, closure, box) = alt ck {
      ty::closure_shared. {
        let r = trans::trans_malloc_boxed(bcx, closure_ty);
        add_clean_free(bcx, r.box, false);
        temp_cleanups += [r.box];
        (r.bcx, r.body, r.box)
      }
      ty::closure_send. {
        // Dummy up a box in the exchange heap.
        let tup_ty = ty::mk_tup(tcx, [ty::mk_int(tcx), closure_ty]);
        let box_ty = ty::mk_uniq(tcx, {ty: tup_ty, mut: ast::imm});
        check trans_uniq::type_is_unique_box(bcx, box_ty);
        let r = trans_uniq::alloc_uniq(bcx, box_ty);
        add_clean_free(bcx, r.val, true);
        temp_cleanups += [r.val];
        dummy_environment_box(bcx, r)
      }
      ty::closure_block. {
        // Dummy up a box on the stack,
        let ty = ty::mk_tup(tcx, [ty::mk_int(tcx), closure_ty]);
        let r = trans::alloc_ty(bcx, ty);
        dummy_environment_box(bcx, r)
      }
    };

    // Store bindings tydesc.
    alt ck {
      ty::closure_shared. | ty::closure_send. {
        let bound_tydesc = GEPi(bcx, closure, [0, abi::closure_elt_tydesc]);
        let ti = none;

        // NDM I believe this is the correct value,
        // but using it exposes bugs and limitations
        // in the shape code.  Therefore, I am using
        // tps_normal, which is what we used before.
        //
        // let tps = tps_fn(vec::len(lltyparams));

        let tps = tps_normal;
        let {result:closure_td, _} =
            trans::get_tydesc(bcx, closure_ty, true, tps, ti);
        trans::lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, ti);
        trans::lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, ti);
        bcx = closure_td.bcx;
        let td = maybe_clone_tydesc(bcx, ck, closure_td.val);
        Store(bcx, td, bound_tydesc);
      }
      ty::closure_block. { /* skip this for blocks, not really relevant */ }
    }

    check type_is_tup_like(bcx, closure_ty);
    let box_ty = ty::mk_imm_box(bcx_tcx(bcx), closure_ty);

    // If necessary, copy tydescs describing type parameters into the
    // appropriate slot in the closure.
    let {bcx:bcx, val:ty_params_slot} =
        GEP_tup_like_1(bcx, closure_ty, closure,
                       [0, abi::closure_elt_ty_params]);
    let off = 0;

    for tp in lltyparams {
        let cloned_td = maybe_clone_tydesc(bcx, ck, tp.desc);
        Store(bcx, cloned_td, GEPi(bcx, ty_params_slot, [0, off]));
        off += 1;
        option::may(tp.dicts, {|dicts|
            for dict in dicts {
                let cast = PointerCast(bcx, dict, val_ty(cloned_td));
                Store(bcx, cast, GEPi(bcx, ty_params_slot, [0, off]));
                off += 1;
            }
        });
    }

    // Copy expr values into boxed bindings.
    // Silly check
    vec::iteri(bound_values) { |i, bv|
        let bound = trans::GEP_tup_like_1(bcx, box_ty, box,
                                          [0, abi::box_rc_field_body,
                                           abi::closure_elt_bindings,
                                           i as int]);
        bcx = bound.bcx;
        alt bv {
          env_expr(e) {
            bcx = trans::trans_expr_save_in(bcx, e, bound.val);
            add_clean_temp_mem(bcx, bound.val, bound_tys[i]);
            temp_cleanups += [bound.val];
          }
          env_copy(val, ty, owned.) {
            let val1 = load_if_immediate(bcx, val, ty);
            bcx = trans::copy_val(bcx, INIT, bound.val, val1, ty);
          }
          env_copy(val, ty, owned_imm.) {
            bcx = trans::copy_val(bcx, INIT, bound.val, val, ty);
          }
          env_copy(_, _, temporary.) {
            fail "Cannot capture temporary upvar";
          }
          env_move(val, ty, kind) {
            let src = {bcx:bcx, val:val, kind:kind};
            bcx = move_val(bcx, INIT, bound.val, src, ty);
          }
          env_ref(val, ty, owned.) {
            Store(bcx, val, bound.val);
          }
          env_ref(val, ty, owned_imm.) {
            let addr = do_spill_noroot(bcx, val);
            Store(bcx, addr, bound.val);
          }
          env_ref(_, _, temporary.) {
            fail "Cannot capture temporary upvar";
          }
        }
    }
    for cleanup in temp_cleanups { revoke_clean(bcx, cleanup); }

    ret {llbox: box, box_ty: box_ty, bcx: bcx};
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for store_environment.
fn build_closure(bcx0: @block_ctxt,
                 cap_vars: [capture::capture_var],
                 ck: ty::closure_kind)
    -> closure_result {
    // If we need to, package up the iterator body to call
    let env_vals = [];
    let bcx = bcx0;
    let tcx = bcx_tcx(bcx);

    // Package up the captured upvars
    vec::iter(cap_vars) { |cap_var|
        let lv = trans_local_var(bcx, cap_var.def);
        let nid = ast_util::def_id_of_def(cap_var.def).node;
        let ty = ty::node_id_to_monotype(tcx, nid);
        alt cap_var.mode {
          capture::cap_ref. {
            assert ck == ty::closure_block;
            ty = ty::mk_mut_ptr(tcx, ty);
            env_vals += [env_ref(lv.val, ty, lv.kind)];
          }
          capture::cap_copy. {
            env_vals += [env_copy(lv.val, ty, lv.kind)];
          }
          capture::cap_move. {
            env_vals += [env_move(lv.val, ty, lv.kind)];
          }
          capture::cap_drop. {
            bcx = drop_ty(bcx, lv.val, ty);
          }
        }
    }
    ret store_environment(bcx, copy bcx.fcx.lltyparams, env_vals, ck);
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
fn load_environment(enclosing_cx: @block_ctxt,
                    fcx: @fn_ctxt,
                    boxed_closure_ty: ty::t,
                    cap_vars: [capture::capture_var],
                    ck: ty::closure_kind) {
    let bcx = new_raw_block_ctxt(fcx, fcx.llloadenv);

    let ccx = bcx_ccx(bcx);
    let sp = bcx.sp;
    check (type_has_static_size(ccx, boxed_closure_ty));
    let llty = type_of(ccx, sp, boxed_closure_ty);
    let llclosure = PointerCast(bcx, fcx.llenv, llty);

    // Populate the type parameters from the environment. We need to
    // do this first because the tydescs are needed to index into
    // the bindings if they are dynamically sized.
    let lltydescs = GEPi(bcx, llclosure,
                         [0, abi::box_rc_field_body,
                          abi::closure_elt_ty_params]);
    let off = 0;
    for tp in copy enclosing_cx.fcx.lltyparams {
        let tydesc = Load(bcx, GEPi(bcx, lltydescs, [0, off]));
        off += 1;
        let dicts = option::map(tp.dicts, {|dicts|
            let rslt = [];
            for dict in dicts {
                let dict = Load(bcx, GEPi(bcx, lltydescs, [0, off]));
                rslt += [PointerCast(bcx, dict, T_ptr(T_dict()))];
                off += 1;
            }
            rslt
        });
        fcx.lltyparams += [{desc: tydesc, dicts: dicts}];
    }

    // Populate the upvars from the environment.
    let path = [0, abi::box_rc_field_body, abi::closure_elt_bindings];
    let i = 0u;
    vec::iter(cap_vars) { |cap_var|
        alt cap_var.mode {
          capture::cap_drop. { /* ignore */ }
          _ {
            check type_is_tup_like(bcx, boxed_closure_ty);
            let upvarptr = GEP_tup_like(
                bcx, boxed_closure_ty, llclosure, path + [i as int]);
            bcx = upvarptr.bcx;
            let llupvarptr = upvarptr.val;
            alt ck {
              ty::closure_block. { llupvarptr = Load(bcx, llupvarptr); }
              ty::closure_send. | ty::closure_shared. { }
            }
            let def_id = ast_util::def_id_of_def(cap_var.def);
            fcx.llupvars.insert(def_id.node, llupvarptr);
            i += 1u;
          }
        }
    }
}

fn trans_expr_fn(bcx: @block_ctxt,
                 proto: ast::proto,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 sp: span,
                 id: ast::node_id,
                 cap_clause: ast::capture_clause,
                 dest: dest) -> @block_ctxt {
    if dest == ignore { ret bcx; }
    let ccx = bcx_ccx(bcx), bcx = bcx;
    let fty = node_id_type(ccx, id);
    let llfnty = type_of_fn_from_ty(ccx, sp, fty, []);
    let sub_cx = extend_path(bcx.fcx.lcx, ccx.names.next("anon"));
    let s = mangle_internal_name_by_path(ccx, sub_cx.path);
    let llfn = decl_internal_cdecl_fn(ccx.llmod, s, llfnty);
    register_fn(ccx, sp, sub_cx.path, "anon fn", [], id);

    let trans_closure_env = lambda(ck: ty::closure_kind) -> ValueRef {
        let cap_vars = capture::compute_capture_vars(
            ccx.tcx, id, proto, cap_clause);
        let {llbox, box_ty, bcx} = build_closure(bcx, cap_vars, ck);
        trans_closure(sub_cx, sp, decl, body, llfn, no_self, [], id, {|fcx|
            load_environment(bcx, fcx, box_ty, cap_vars, ck);
        });
        llbox
    };

    let closure = alt proto {
      ast::proto_block. { trans_closure_env(ty::closure_block) }
      ast::proto_shared(_) { trans_closure_env(ty::closure_shared) }
      ast::proto_send. { trans_closure_env(ty::closure_send) }
      ast::proto_bare. {
        let closure = C_null(T_opaque_boxed_closure_ptr(ccx));
        trans_closure(sub_cx, sp, decl, body, llfn, no_self, [],
                      id, {|_fcx|});
        closure
      }
    };
    fill_fn_pair(bcx, get_dest_addr(dest), llfn, closure);
    ret bcx;
}

fn trans_bind(cx: @block_ctxt, f: @ast::expr, args: [option::t<@ast::expr>],
              id: ast::node_id, dest: dest) -> @block_ctxt {
    let f_res = trans_callee(cx, f);
    ret trans_bind_1(cx, ty::expr_ty(bcx_tcx(cx), f), f_res, args,
                     ty::node_id_to_type(bcx_tcx(cx), id), dest);
}

fn trans_bind_1(cx: @block_ctxt, outgoing_fty: ty::t,
                f_res: lval_maybe_callee,
                args: [option::t<@ast::expr>], pair_ty: ty::t,
                dest: dest) -> @block_ctxt {
    let bound: [@ast::expr] = [];
    for argopt: option::t<@ast::expr> in args {
        alt argopt { none. { } some(e) { bound += [e]; } }
    }
    let bcx = f_res.bcx;
    if dest == ignore {
        for ex in bound { bcx = trans_expr(bcx, ex, ignore); }
        ret bcx;
    }

    // Figure out which tydescs we need to pass, if any.
    let (outgoing_fty_real, lltydescs, param_bounds) = alt f_res.generic {
      none. { (outgoing_fty, [], @[]) }
      some(ginfo) {
        let tds = [], orig = 0u;
        vec::iter2(ginfo.tydescs, *ginfo.param_bounds) {|td, bounds|
            tds += [td];
            for bound in *bounds {
                alt bound {
                  ty::bound_iface(_) {
                    let dict = trans_impl::get_dict(
                        bcx, option::get(ginfo.origins)[orig]);
                    tds += [PointerCast(bcx, dict.val, val_ty(td))];
                    orig += 1u;
                    bcx = dict.bcx;
                  }
                  _ {}
                }
            }
        }
        lazily_emit_all_generic_info_tydesc_glues(cx, ginfo);
        (ginfo.item_type, tds, ginfo.param_bounds)
      }
    };

    if vec::len(bound) == 0u && vec::len(lltydescs) == 0u {
        // Trivial 'binding': just return the closure
        let lv = lval_maybe_callee_to_lval(f_res, pair_ty);
        bcx = lv.bcx;
        ret memmove_ty(bcx, get_dest_addr(dest), lv.val, pair_ty);
    }
    let closure = alt f_res.env {
      null_env. { none }
      _ { let (_, cl) = maybe_add_env(cx, f_res); some(cl) }
    };

    // FIXME: should follow from a precondition on trans_bind_1
    let ccx = bcx_ccx(cx);
    check (type_has_static_size(ccx, outgoing_fty));

    // Arrange for the bound function to live in the first binding spot
    // if the function is not statically known.
    let (env_vals, target_res) = alt closure {
      some(cl) {
        // Cast the function we are binding to be the type that the
        // closure will expect it to have. The type the closure knows
        // about has the type parameters substituted with the real types.
        let sp = cx.sp;
        let llclosurety = T_ptr(type_of(ccx, sp, outgoing_fty));
        let src_loc = PointerCast(bcx, cl, llclosurety);
        ([env_copy(src_loc, pair_ty, owned)], none)
      }
      none. { ([], some(f_res.val)) }
    };

    // Actually construct the closure
    let {llbox, box_ty, bcx} = store_environment(
        bcx, vec::map(lltydescs, {|d| {desc: d, dicts: none}}),
        env_vals + vec::map(bound, {|x| env_expr(x)}),
        ty::closure_shared);

    // Make thunk
    let llthunk =
        trans_bind_thunk(cx.fcx.lcx, cx.sp, pair_ty, outgoing_fty_real, args,
                         box_ty, *param_bounds, target_res);

    // Fill the function pair
    fill_fn_pair(bcx, get_dest_addr(dest), llthunk.val, llbox);
    ret bcx;
}

fn make_fn_glue(
    cx: @block_ctxt,
    v: ValueRef,
    t: ty::t,
    glue_fn: fn(@block_ctxt, v: ValueRef, t: ty::t) -> @block_ctxt)
    -> @block_ctxt {
    let bcx = cx;
    let tcx = bcx_tcx(cx);

    let fn_env = lambda(blk: block(@block_ctxt, ValueRef) -> @block_ctxt)
        -> @block_ctxt {
        let box_cell_v = GEPi(cx, v, [0, abi::fn_field_box]);
        let box_ptr_v = Load(cx, box_cell_v);
        let inner_cx = new_sub_block_ctxt(cx, "iter box");
        let next_cx = new_sub_block_ctxt(cx, "next");
        let null_test = IsNull(cx, box_ptr_v);
        CondBr(cx, null_test, next_cx.llbb, inner_cx.llbb);
        inner_cx = blk(inner_cx, box_cell_v);
        Br(inner_cx, next_cx.llbb);
        ret next_cx;
    };

    ret alt ty::struct(tcx, t) {
      ty::ty_native_fn(_, _) | ty::ty_fn({proto: ast::proto_bare., _}) {
        bcx
      }
      ty::ty_fn({proto: ast::proto_block., _}) {
        bcx
      }
      ty::ty_fn({proto: ast::proto_send., _}) {
        fn_env({ |bcx, box_cell_v|
            let box_ty = trans_closure::send_opaque_closure_box_ty(tcx);
            glue_fn(bcx, box_cell_v, box_ty)
        })
      }
      ty::ty_fn({proto: ast::proto_shared(_), _}) {
        fn_env({ |bcx, box_cell_v|
            let box_ty = trans_closure::shared_opaque_closure_box_ty(tcx);
            glue_fn(bcx, box_cell_v, box_ty)
        })
      }
      _ { fail "make_fn_glue invoked on non-function type" }
    };
}

fn call_opaque_closure_glue(bcx: @block_ctxt,
                            v: ValueRef,     // ptr to an opaque closure
                            field: int) -> @block_ctxt {
    let ccx = bcx_ccx(bcx);
    let v = PointerCast(bcx, v, T_ptr(T_opaque_closure(ccx)));
    let tydescptr = GEPi(bcx, v, [0, abi::closure_elt_tydesc]);
    let tydesc = Load(bcx, tydescptr);
    let ti = none;
    call_tydesc_glue_full(bcx, v, tydesc, field, ti);
    ret bcx;
}

// pth is cx.path
fn trans_bind_thunk(cx: @local_ctxt,
                    sp: span,
                    incoming_fty: ty::t,
                    outgoing_fty: ty::t,
                    args: [option::t<@ast::expr>],
                    boxed_closure_ty: ty::t,
                    param_bounds: [ty::param_bounds],
                    target_fn: option::t<ValueRef>)
    -> {val: ValueRef, ty: TypeRef} {
    // If we supported constraints on record fields, we could make the
    // constraints for this function:
    /*
    : returns_non_ty_var(ccx, outgoing_fty),
      type_has_static_size(ccx, incoming_fty) ->
    */
    // but since we don't, we have to do the checks at the beginning.
    let ccx = cx.ccx;
    check type_has_static_size(ccx, incoming_fty);

    // Here we're not necessarily constructing a thunk in the sense of
    // "function with no arguments".  The result of compiling 'bind f(foo,
    // bar, baz)' would be a thunk that, when called, applies f to those
    // arguments and returns the result.  But we're stretching the meaning of
    // the word "thunk" here to also mean the result of compiling, say, 'bind
    // f(foo, _, baz)', or any other bind expression that binds f and leaves
    // some (or all) of the arguments unbound.

    // Here, 'incoming_fty' is the type of the entire bind expression, while
    // 'outgoing_fty' is the type of the function that is having some of its
    // arguments bound.  If f is a function that takes three arguments of type
    // int and returns int, and we're translating, say, 'bind f(3, _, 5)',
    // then outgoing_fty is the type of f, which is (int, int, int) -> int,
    // and incoming_fty is the type of 'bind f(3, _, 5)', which is int -> int.

    // Once translated, the entire bind expression will be the call f(foo,
    // bar, baz) wrapped in a (so-called) thunk that takes 'bar' as its
    // argument and that has bindings of 'foo' to 3 and 'baz' to 5 and a
    // pointer to 'f' all saved in its environment.  So, our job is to
    // construct and return that thunk.

    // Give the thunk a name, type, and value.
    let s: str = mangle_internal_name_by_path_and_seq(ccx, cx.path, "thunk");
    let llthunk_ty: TypeRef = get_pair_fn_ty(type_of(ccx, sp, incoming_fty));
    let llthunk: ValueRef = decl_internal_cdecl_fn(ccx.llmod, s, llthunk_ty);

    // Create a new function context and block context for the thunk, and hold
    // onto a pointer to the first block in the function for later use.
    let fcx = new_fn_ctxt(cx, sp, llthunk);
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;
    // Since we might need to construct derived tydescs that depend on
    // our bound tydescs, we need to load tydescs out of the environment
    // before derived tydescs are constructed. To do this, we load them
    // in the load_env block.
    let l_bcx = new_raw_block_ctxt(fcx, fcx.llloadenv);

    // The 'llenv' that will arrive in the thunk we're creating is an
    // environment that will contain the values of its arguments and a pointer
    // to the original function.  So, let's create one of those:

    // The llenv pointer needs to be the correct size.  That size is
    // 'boxed_closure_ty', which was determined by trans_bind.
    check (type_has_static_size(ccx, boxed_closure_ty));
    let llclosure_ptr_ty = type_of(ccx, sp, boxed_closure_ty);
    let llclosure = PointerCast(l_bcx, fcx.llenv, llclosure_ptr_ty);

    // "target", in this context, means the function that's having some of its
    // arguments bound and that will be called inside the thunk we're
    // creating.  (In our running example, target is the function f.)  Pick
    // out the pointer to the target function from the environment. The
    // target function lives in the first binding spot.
    let (lltargetfn, lltargetenv, starting_idx) = alt target_fn {
      some(fptr) {
        (fptr, llvm::LLVMGetUndef(T_opaque_boxed_closure_ptr(ccx)), 0)
      }
      none. {
        // Silly check
        check type_is_tup_like(bcx, boxed_closure_ty);
        let {bcx: cx, val: pair} =
            GEP_tup_like(bcx, boxed_closure_ty, llclosure,
                         [0, abi::box_rc_field_body,
                          abi::closure_elt_bindings, 0]);
        let lltargetenv =
            Load(cx, GEPi(cx, pair, [0, abi::fn_field_box]));
        let lltargetfn = Load
            (cx, GEPi(cx, pair, [0, abi::fn_field_code]));
        bcx = cx;
        (lltargetfn, lltargetenv, 1)
      }
    };

    // And then, pick out the target function's own environment.  That's what
    // we'll use as the environment the thunk gets.

    // Get f's return type, which will also be the return type of the entire
    // bind expression.
    let outgoing_ret_ty = ty::ty_fn_ret(cx.ccx.tcx, outgoing_fty);

    // Get the types of the arguments to f.
    let outgoing_args = ty::ty_fn_args(cx.ccx.tcx, outgoing_fty);

    // The 'llretptr' that will arrive in the thunk we're creating also needs
    // to be the correct type.  Cast it to f's return type, if necessary.
    let llretptr = fcx.llretptr;
    let ccx = cx.ccx;
    if ty::type_contains_params(ccx.tcx, outgoing_ret_ty) {
        check non_ty_var(ccx, outgoing_ret_ty);
        let llretty = type_of_inner(ccx, sp, outgoing_ret_ty);
        llretptr = PointerCast(bcx, llretptr, T_ptr(llretty));
    }

    // Set up the three implicit arguments to the thunk.
    let llargs: [ValueRef] = [llretptr, lltargetenv];

    // Copy in the type parameters.
    check type_is_tup_like(l_bcx, boxed_closure_ty);
    let {bcx: l_bcx, val: param_record} =
        GEP_tup_like(l_bcx, boxed_closure_ty, llclosure,
                     [0, abi::box_rc_field_body, abi::closure_elt_ty_params]);
    let off = 0;
    for param in param_bounds {
        let dsc = Load(l_bcx, GEPi(l_bcx, param_record, [0, off])),
            dicts = none;
        llargs += [dsc];
        off += 1;
        for bound in *param {
            alt bound {
              ty::bound_iface(_) {
                let dict = Load(l_bcx, GEPi(l_bcx, param_record, [0, off]));
                dict = PointerCast(l_bcx, dict, T_ptr(T_dict()));
                llargs += [dict];
                off += 1;
                dicts = some(alt dicts {
                  none. { [dict] }
                  some(ds) { ds + [dict] }
                });
              }
              _ {}
            }
        }
        fcx.lltyparams += [{desc: dsc, dicts: dicts}];
    }

    let a: uint = 2u; // retptr, env come first
    let b: int = starting_idx;
    let outgoing_arg_index: uint = 0u;
    let llout_arg_tys: [TypeRef] =
        type_of_explicit_args(cx.ccx, sp, outgoing_args);
    for arg: option::t<@ast::expr> in args {
        let out_arg = outgoing_args[outgoing_arg_index];
        let llout_arg_ty = llout_arg_tys[outgoing_arg_index];
        alt arg {
          // Arg provided at binding time; thunk copies it from
          // closure.
          some(e) {
            // Silly check
            check type_is_tup_like(bcx, boxed_closure_ty);
            let bound_arg =
                GEP_tup_like(bcx, boxed_closure_ty, llclosure,
                             [0, abi::box_rc_field_body,
                              abi::closure_elt_bindings, b]);
            bcx = bound_arg.bcx;
            let val = bound_arg.val;
            if out_arg.mode == ast::by_val { val = Load(bcx, val); }
            if out_arg.mode == ast::by_copy {
                let {bcx: cx, val: alloc} = alloc_ty(bcx, out_arg.ty);
                bcx = memmove_ty(cx, alloc, val, out_arg.ty);
                bcx = take_ty(bcx, alloc, out_arg.ty);
                val = alloc;
            }
            // If the type is parameterized, then we need to cast the
            // type we actually have to the parameterized out type.
            if ty::type_contains_params(cx.ccx.tcx, out_arg.ty) {
                val = PointerCast(bcx, val, llout_arg_ty);
            }
            llargs += [val];
            b += 1;
          }

          // Arg will be provided when the thunk is invoked.
          none. {
            let arg: ValueRef = llvm::LLVMGetParam(llthunk, a);
            if ty::type_contains_params(cx.ccx.tcx, out_arg.ty) {
                arg = PointerCast(bcx, arg, llout_arg_ty);
            }
            llargs += [arg];
            a += 1u;
          }
        }
        outgoing_arg_index += 1u;
    }

    // Cast the outgoing function to the appropriate type.
    // This is necessary because the type of the function that we have
    // in the closure does not know how many type descriptors the function
    // needs to take.
    let ccx = bcx_ccx(bcx);

    let lltargetty =
        type_of_fn_from_ty(ccx, sp, outgoing_fty, param_bounds);
    lltargetfn = PointerCast(bcx, lltargetfn, T_ptr(lltargetty));
    Call(bcx, lltargetfn, llargs);
    build_return(bcx);
    finish_fn(fcx, lltop);
    ret {val: llthunk, ty: llthunk_ty};
}
