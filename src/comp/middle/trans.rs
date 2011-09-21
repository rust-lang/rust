// trans.rs: Translate the completed AST to the LLVM IR.
//
// Some functions here, such as trans_block and trans_expr, return a value --
// the result of the translation to LLVM -- while others, such as trans_fn,
// trans_obj, and trans_item, are called only for the side effect of adding a
// particular definition to the LLVM IR output we're producing.
//
// Hopefully useful general knowledge about trans:
//
//   * There's no way to find out the ty::t type of a ValueRef.  Doing so
//     would be "trying to get the eggs out of an omelette" (credit:
//     pcwalton).  You can, instead, find out its TypeRef by calling val_ty,
//     but many TypeRefs correspond to one ty::t; for instance, tup(int, int,
//     int) and rec(x=int, y=int, z=int) will have the same TypeRef.
import std::{int, str, uint, map, option, fs, time, vec};
import std::map::hashmap;
import std::map::{new_int_hash, new_str_hash};
import std::option::{some, none};
import driver::session;
import middle::{ty, gc};
import middle::freevars::*;
import back::{link, x86, abi, upcall};
import syntax::{ast, ast_util};
import syntax::visit;
import syntax::codemap::span;
import syntax::print::pprust::{expr_to_str, path_to_str};
import visit::vt;
import util::common;
import util::common::*;
import lib::llvm::{llvm, target_data, type_names,
                   mk_target_data, mk_type_names};
import lib::llvm::llvm::{ModuleRef, ValueRef, TypeRef, TypeHandleRef,
                         BuilderRef, BasicBlockRef};
import lib::llvm::{Bool, True, False};
import link::{mangle_internal_name_by_type_only,
              mangle_internal_name_by_seq,
              mangle_internal_name_by_path,
              mangle_internal_name_by_path_and_seq,
              mangle_exported_name};
import metadata::{creader, csearch, cstore};
import util::ppaux::{ty_to_str, ty_to_short_str};

import trans_common::*;
import trans_build::*;

import trans_objects::{trans_anon_obj, trans_obj};
import tvec = trans_vec;

fn type_of(cx: @crate_ctxt, sp: span, t: ty::t) : type_has_static_size(cx, t)
   -> TypeRef {
    // Should follow from type_has_static_size -- argh.
    // FIXME (requires Issue #586)
    check non_ty_var(cx, t);
    type_of_inner(cx, sp, t)
}

fn type_of_explicit_args(cx: @crate_ctxt, sp: span, inputs: [ty::arg]) ->
   [TypeRef] {
    let atys = [];
    for arg in inputs {
        let arg_ty = arg.ty;
        // FIXME: would be nice to have a constraint on arg
        // that would obviate the need for this check
        check non_ty_var(cx, arg_ty);
        atys += [T_ptr(type_of_inner(cx, sp, arg_ty))];
    }
    ret atys;
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn type_of_fn(cx: @crate_ctxt, sp: span, proto: ast::proto,
              is_method: bool, ret_ref: bool, inputs: [ty::arg],
              output: ty::t, ty_param_count: uint)
   : non_ty_var(cx, output) -> TypeRef {
    let atys: [TypeRef] = [];

    // Arg 0: Output pointer.
    let out_ty = T_ptr(type_of_inner(cx, sp, output));
    atys += [ret_ref ? T_ptr(out_ty) : out_ty];

    // Arg 1: task pointer.
    atys += [T_taskptr(*cx)];

    // Arg 2: Env (closure-bindings / self-obj)
    if is_method {
        atys += [T_ptr(cx.rust_object_type)];
    } else { atys += [T_opaque_closure_ptr(*cx)]; }

    // Args >3: ty params, if not acquired via capture...
    if !is_method {
        let i = 0u;
        while i < ty_param_count { atys += [T_ptr(cx.tydesc_type)]; i += 1u; }
    }
    if proto == ast::proto_iter {
        // If it's an iter, the 'output' type of the iter is actually the
        // *input* type of the function we're given as our iter-block
        // argument.
        let iter_body_ty = ty::mk_iter_body_fn(cx.tcx, output);
        // FIXME: this check could be avoided pretty easily if we had
        // postconditions
        // (or better yet, just use a constraiend type that expresses
        // non-ty-var things)
        check non_ty_var(cx, iter_body_ty);
        atys += [type_of_inner(cx, sp, iter_body_ty)];
    }
    // ... then explicit args.
    atys += type_of_explicit_args(cx, sp, inputs);
    ret T_fn(atys, llvm::LLVMVoidType());
}

// Given a function type and a count of ty params, construct an llvm type
fn type_of_fn_from_ty(cx: @crate_ctxt, sp: span, fty: ty::t,
                      ty_param_count: uint)
    : returns_non_ty_var(cx, fty) -> TypeRef {
    let by_ref = ast_util::ret_by_ref(ty::ty_fn_ret_style(cx.tcx, fty));
    // FIXME: Check should be unnecessary, b/c it's implied
    // by returns_non_ty_var(t). Make that a postcondition
    // (see Issue #586)
    let ret_ty = ty::ty_fn_ret(cx.tcx, fty);
    check non_ty_var(cx, ret_ty);
    ret type_of_fn(cx, sp, ty::ty_fn_proto(cx.tcx, fty),
                   false, by_ref, ty::ty_fn_args(cx.tcx, fty),
                   ret_ty, ty_param_count);
}

fn type_of_inner(cx: @crate_ctxt, sp: span, t: ty::t)
    : non_ty_var(cx, t) -> TypeRef {
    // Check the cache.

    if cx.lltypes.contains_key(t) { ret cx.lltypes.get(t); }
    let llty =
    alt ty::struct(cx.tcx, t) {
      ty::ty_native(_) { T_ptr(T_i8()) }
      ty::ty_nil. { T_nil() }
      ty::ty_bot. {
        T_nil() /* ...I guess? */
      }
      ty::ty_bool. { T_bool() }
      ty::ty_int. { T_int() }
      ty::ty_float. { T_float() }
      ty::ty_uint. { T_int() }
      ty::ty_machine(tm) {
        alt tm {
          ast::ty_i8. | ast::ty_u8. { T_i8() }
          ast::ty_i16. | ast::ty_u16. { T_i16() }
          ast::ty_i32. | ast::ty_u32. { T_i32() }
          ast::ty_i64. | ast::ty_u64. { T_i64() }
          ast::ty_f32. { T_f32() }
          ast::ty_f64. { T_f64() }
        }
      }
      ty::ty_char. { T_char() }
      ty::ty_str. { T_ptr(T_vec(T_i8())) }
      ty::ty_tag(did, _) { type_of_tag(cx, sp, did, t) }
      ty::ty_box(mt) {
        let mt_ty = mt.ty;
        check non_ty_var(cx, mt_ty);
        T_ptr(T_box(type_of_inner(cx, sp, mt_ty))) }
      ty::ty_uniq(t) {
        check non_ty_var(cx, t);
        T_ptr(type_of_inner(cx, sp, t)) }
      ty::ty_vec(mt) {
        let mt_ty = mt.ty;
        if ty::type_has_dynamic_size(cx.tcx, mt_ty) {
            T_ptr(T_opaque_vec())
        } else {
            // should be unnecessary
            check non_ty_var(cx, mt_ty);
            T_ptr(T_vec(type_of_inner(cx, sp, mt_ty))) }
      }
      ty::ty_ptr(mt) {
        let mt_ty = mt.ty;
        check non_ty_var(cx, mt_ty);
        T_ptr(type_of_inner(cx, sp, mt_ty)) }
      ty::ty_rec(fields) {
        let tys: [TypeRef] = [];
        for f: ty::field in fields {
            let mt_ty = f.mt.ty;
            check non_ty_var(cx, mt_ty);
            tys += [type_of_inner(cx, sp, mt_ty)];
        }
        T_struct(tys)
      }
      ty::ty_fn(_, _, _, _, _) {
        // FIXME: could be a constraint on ty_fn
        check returns_non_ty_var(cx, t);
        T_fn_pair(*cx, type_of_fn_from_ty(cx, sp, t, 0u))
      }
      ty::ty_native_fn(abi, args, out) {
        let nft = native_fn_wrapper_type(cx, sp, 0u, t);
        T_fn_pair(*cx, nft)
      }
      ty::ty_obj(meths) { cx.rust_object_type }
      ty::ty_res(_, sub, tps) {
        let sub1 = ty::substitute_type_params(cx.tcx, tps, sub);
        check non_ty_var(cx, sub1);
        ret T_struct([T_i32(), type_of_inner(cx, sp, sub1)]);
      }
      ty::ty_var(_) {
        // Should be unreachable b/c of precondition.
        // FIXME: would be nice to have a way of expressing this
        // through postconditions, and then making it sound to omit
        // cases in the alt
        std::util::unreachable()
      }
      ty::ty_param(_, _) { T_typaram(cx.tn) }
      ty::ty_type. { T_ptr(cx.tydesc_type) }
      ty::ty_tup(elts) {
        let tys = [];
        for elt in elts {
            check non_ty_var(cx, elt);
            tys += [type_of_inner(cx, sp, elt)];
        }
        T_struct(tys)
      }
    };
    cx.lltypes.insert(t, llty);
    ret llty;
}

fn type_of_tag(cx: @crate_ctxt, sp: span, did: ast::def_id, t: ty::t)
    -> TypeRef {
    let degen = std::vec::len(ty::tag_variants(cx.tcx, did)) == 1u;
    if check type_has_static_size(cx, t) {
        let size = static_size_of_tag(cx, sp, t);
        if !degen { ret T_tag(cx.tn, size); }
        // LLVM does not like 0-size arrays, apparently
        if size == 0u { size = 1u; }
        ret T_array(T_i8(), size);
    }
    else {
        if degen { ret T_i8(); } else { ret T_opaque_tag(cx.tn); }
    }
}

fn type_of_ty_param_kinds_and_ty(lcx: @local_ctxt, sp: span,
                                 tpt: ty::ty_param_kinds_and_ty) -> TypeRef {
    let cx = lcx.ccx;
    let t = tpt.ty;
    alt ty::struct(cx.tcx, t) {
      ty::ty_fn(_, _, _, _, _) | ty::ty_native_fn(_, _, _) {
        check returns_non_ty_var(cx, t);
        ret type_of_fn_from_ty(cx, sp, t, std::vec::len(tpt.kinds));
      }
      _ {
        // fall through
      }
    }
    // FIXME: could have a precondition on tpt, but that
    // doesn't work right now because one predicate can't imply
    // another
    check (type_has_static_size(cx, t));
    type_of(cx, sp, t)
}

fn type_of_or_i8(bcx: @block_ctxt, typ: ty::t) -> TypeRef {
    let ccx = bcx_ccx(bcx);
    if check type_has_static_size(ccx, typ) {
        let sp = bcx.sp;
        type_of(ccx, sp, typ)
    } else { T_i8() }
}


// Name sanitation. LLVM will happily accept identifiers with weird names, but
// gas doesn't!
fn sanitize(s: str) -> str {
    let result = "";
    for c: u8 in s {
        if c == '@' as u8 {
            result += "boxed_";
        } else {
            if c == ',' as u8 {
                result += "_";
            } else {
                if c == '{' as u8 || c == '(' as u8 {
                    result += "_of_";
                } else {
                    if c != 10u8 && c != '}' as u8 && c != ')' as u8 &&
                           c != ' ' as u8 && c != '\t' as u8 && c != ';' as u8
                       {
                        let v = [c];
                        result += str::unsafe_from_bytes(v);
                    }
                }
            }
        }
    }
    ret result;
}


fn log_fn_time(ccx: @crate_ctxt, name: str, start: time::timeval,
               end: time::timeval) {
    let elapsed =
        1000 * (end.sec - start.sec as int) +
            ((end.usec as int) - (start.usec as int)) / 1000;
    *ccx.stats.fn_times += [{ident: name, time: elapsed}];
}


fn decl_fn(llmod: ModuleRef, name: str, cc: uint, llty: TypeRef) -> ValueRef {
    let llfn: ValueRef =
        str::as_buf(name, {|buf| llvm::LLVMAddFunction(llmod, buf, llty) });
    llvm::LLVMSetFunctionCallConv(llfn, cc);
    ret llfn;
}

fn decl_cdecl_fn(llmod: ModuleRef, name: str, llty: TypeRef) -> ValueRef {
    ret decl_fn(llmod, name, lib::llvm::LLVMCCallConv, llty);
}

fn decl_fastcall_fn(llmod: ModuleRef, name: str, llty: TypeRef) -> ValueRef {
    let llfn = decl_fn(llmod, name, lib::llvm::LLVMFastCallConv, llty);
    let _: () = str::as_buf("rust", {|buf| llvm::LLVMSetGC(llfn, buf) });
    ret llfn;
}


// Only use this if you are going to actually define the function. It's
// not valid to simply declare a function as internal.
fn decl_internal_fastcall_fn(llmod: ModuleRef, name: str, llty: TypeRef) ->
   ValueRef {
    let llfn = decl_fastcall_fn(llmod, name, llty);
    llvm::LLVMSetLinkage(llfn,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    ret llfn;
}

fn decl_glue(llmod: ModuleRef, cx: crate_ctxt, s: str) -> ValueRef {
    ret decl_cdecl_fn(llmod, s, T_fn([T_taskptr(cx)], T_void()));
}

fn get_extern_fn(externs: hashmap<str, ValueRef>, llmod: ModuleRef, name: str,
                 cc: uint, ty: TypeRef) -> ValueRef {
    if externs.contains_key(name) { ret externs.get(name); }
    let f = decl_fn(llmod, name, cc, ty);
    externs.insert(name, f);
    ret f;
}

fn get_extern_const(externs: hashmap<str, ValueRef>, llmod: ModuleRef,
                    name: str, ty: TypeRef) -> ValueRef {
    if externs.contains_key(name) { ret externs.get(name); }
    let c = str::as_buf(name, {|buf| llvm::LLVMAddGlobal(llmod, ty, buf) });
    externs.insert(name, c);
    ret c;
}

fn get_simple_extern_fn(externs: hashmap<str, ValueRef>, llmod: ModuleRef,
                        name: str, n_args: int) -> ValueRef {
    let inputs = std::vec::init_elt::<TypeRef>(T_int(), n_args as uint);
    let output = T_int();
    let t = T_fn(inputs, output);
    ret get_extern_fn(externs, llmod, name, lib::llvm::LLVMCCallConv, t);
}

fn trans_native_call(cx: @block_ctxt, externs: hashmap<str, ValueRef>,
                     llmod: ModuleRef, name: str, args: [ValueRef]) ->
   ValueRef {
    let n: int = std::vec::len::<ValueRef>(args) as int;
    let llnative: ValueRef = get_simple_extern_fn(externs, llmod, name, n);
    let call_args: [ValueRef] = [];
    for a: ValueRef in args { call_args += [ZExtOrBitCast(cx, a, T_int())]; }
    ret Call(cx, llnative, call_args);
}

fn trans_non_gc_free(cx: @block_ctxt, v: ValueRef) -> @block_ctxt {
    Call(cx, bcx_ccx(cx).upcalls.free,
         [cx.fcx.lltaskptr, PointerCast(cx, v, T_ptr(T_i8())), C_int(0)]);
    ret cx;
}

fn trans_shared_free(cx: @block_ctxt, v: ValueRef) -> @block_ctxt {
    Call(cx, bcx_ccx(cx).upcalls.shared_free,
         [cx.fcx.lltaskptr, PointerCast(cx, v, T_ptr(T_i8()))]);
    ret cx;
}

fn umax(cx: @block_ctxt, a: ValueRef, b: ValueRef) -> ValueRef {
    let cond = ICmp(cx, lib::llvm::LLVMIntULT, a, b);
    ret Select(cx, cond, b, a);
}

fn umin(cx: @block_ctxt, a: ValueRef, b: ValueRef) -> ValueRef {
    let cond = ICmp(cx, lib::llvm::LLVMIntULT, a, b);
    ret Select(cx, cond, a, b);
}

fn align_to(cx: @block_ctxt, off: ValueRef, align: ValueRef) -> ValueRef {
    let mask = Sub(cx, align, C_int(1));
    let bumped = Add(cx, off, mask);
    ret And(cx, bumped, Not(cx, mask));
}


// Returns the real size of the given type for the current target.
fn llsize_of_real(cx: @crate_ctxt, t: TypeRef) -> uint {
    ret llvm::LLVMStoreSizeOfType(cx.td.lltd, t);
}

// Returns the real alignment of the given type for the current target.
fn llalign_of_real(cx: @crate_ctxt, t: TypeRef) -> uint {
    ret llvm::LLVMPreferredAlignmentOfType(cx.td.lltd, t);
}

fn llsize_of(t: TypeRef) -> ValueRef {
    ret llvm::LLVMConstIntCast(lib::llvm::llvm::LLVMSizeOf(t), T_int(),
                               False);
}

fn llalign_of(t: TypeRef) -> ValueRef {
    ret llvm::LLVMConstIntCast(lib::llvm::llvm::LLVMAlignOf(t), T_int(),
                               False);
}

fn size_of(cx: @block_ctxt, t: ty::t) -> result {
    let ccx = bcx_ccx(cx);
    if check type_has_static_size(ccx, t) {
        let sp = cx.sp;
        rslt(cx, llsize_of(type_of(ccx, sp, t)))
    } else { dynamic_size_of(cx, t) }
}

fn align_of(cx: @block_ctxt, t: ty::t) -> result {
    let ccx = bcx_ccx(cx);
    if check type_has_static_size(ccx, t) {
        let sp = cx.sp;
        rslt(cx, llalign_of(type_of(ccx, sp, t)))
    } else { dynamic_align_of(cx, t) }
}

fn alloca(cx: @block_ctxt, t: TypeRef) -> ValueRef {
    ret Alloca(new_raw_block_ctxt(cx.fcx, cx.fcx.llstaticallocas), t);
}

fn dynastack_alloca(cx: @block_ctxt, t: TypeRef, n: ValueRef, ty: ty::t) ->
   ValueRef {
    let bcx = cx;
    let dy_cx = new_raw_block_ctxt(cx.fcx, cx.fcx.lldynamicallocas);
    let lltaskptr = bcx_fcx(bcx).lltaskptr;
    alt bcx_fcx(cx).llobstacktoken {
      none. {
        bcx_fcx(cx).llobstacktoken =
            some(mk_obstack_token(bcx_ccx(cx), cx.fcx, lltaskptr));
      }
      some(_) {/* no-op */ }
    }

    let dynastack_alloc = bcx_ccx(bcx).upcalls.dynastack_alloc;
    let llsz = Mul(dy_cx, C_uint(llsize_of_real(bcx_ccx(bcx), t)), n);

    let ti = none;
    let lltydesc = get_tydesc(cx, ty, false, tps_normal, ti).result.val;

    let llresult = Call(dy_cx, dynastack_alloc, [lltaskptr, llsz, lltydesc]);
    ret PointerCast(dy_cx, llresult, T_ptr(t));
}

fn mk_obstack_token(ccx: @crate_ctxt, fcx: @fn_ctxt, lltaskptr: ValueRef) ->
   ValueRef {
    let cx = new_raw_block_ctxt(fcx, fcx.lldynamicallocas);
    ret Call(cx, ccx.upcalls.dynastack_mark, [lltaskptr]);
}


// Creates a simpler, size-equivalent type. The resulting type is guaranteed
// to have (a) the same size as the type that was passed in; (b) to be non-
// recursive. This is done by replacing all boxes in a type with boxed unit
// types.
fn simplify_type(ccx: @crate_ctxt, typ: ty::t) -> ty::t {
    fn simplifier(ccx: @crate_ctxt, typ: ty::t) -> ty::t {
        alt ty::struct(ccx.tcx, typ) {
          ty::ty_box(_) { ret ty::mk_imm_box(ccx.tcx, ty::mk_nil(ccx.tcx)); }
          ty::ty_uniq(_) { ret ty::mk_uniq(ccx.tcx, ty::mk_nil(ccx.tcx)); }
          ty::ty_fn(_, _, _, _, _) {
            ret ty::mk_tup(ccx.tcx,
                           [ty::mk_imm_box(ccx.tcx, ty::mk_nil(ccx.tcx)),
                            ty::mk_imm_box(ccx.tcx, ty::mk_nil(ccx.tcx))]);
          }
          ty::ty_obj(_) {
            ret ty::mk_tup(ccx.tcx,
                           [ty::mk_imm_box(ccx.tcx, ty::mk_nil(ccx.tcx)),
                            ty::mk_imm_box(ccx.tcx, ty::mk_nil(ccx.tcx))]);
          }
          ty::ty_res(_, sub, tps) {
            let sub1 = ty::substitute_type_params(ccx.tcx, tps, sub);
            ret ty::mk_tup(ccx.tcx,
                           [ty::mk_int(ccx.tcx), simplify_type(ccx, sub1)]);
          }
          _ { ret typ; }
        }
    }
    ret ty::fold_ty(ccx.tcx, ty::fm_general(bind simplifier(ccx, _)), typ);
}


// Computes the size of the data part of a non-dynamically-sized tag.
fn static_size_of_tag(cx: @crate_ctxt, sp: span, t: ty::t)
    : type_has_static_size(cx, t) -> uint {
    if cx.tag_sizes.contains_key(t) { ret cx.tag_sizes.get(t); }
    alt ty::struct(cx.tcx, t) {
      ty::ty_tag(tid, subtys) {
        // Compute max(variant sizes).

        let max_size = 0u;
        let variants = ty::tag_variants(cx.tcx, tid);
        for variant: ty::variant_info in variants {
            let tup_ty = simplify_type(cx, ty::mk_tup(cx.tcx, variant.args));
            // Perform any type parameter substitutions.

            tup_ty = ty::substitute_type_params(cx.tcx, subtys, tup_ty);
            // Here we possibly do a recursive call.

            // FIXME: Avoid this check. Since the parent has static
            // size, any field must as well. There should be a way to
            // express that with constrained types.
            check (type_has_static_size(cx, tup_ty));
            let this_size = llsize_of_real(cx, type_of(cx, sp, tup_ty));
            if max_size < this_size { max_size = this_size; }
        }
        cx.tag_sizes.insert(t, max_size);
        ret max_size;
      }
      _ {
        cx.tcx.sess.span_fatal(sp, "non-tag passed to static_size_of_tag()");
      }
    }
}

fn dynamic_size_of(cx: @block_ctxt, t: ty::t) -> result {
    fn align_elements(cx: @block_ctxt, elts: [ty::t]) -> result {
        //
        // C padding rules:
        //
        //
        //   - Pad after each element so that next element is aligned.
        //   - Pad after final structure member so that whole structure
        //     is aligned to max alignment of interior.
        //

        let off = C_int(0);
        let max_align = C_int(1);
        let bcx = cx;
        for e: ty::t in elts {
            let elt_align = align_of(bcx, e);
            bcx = elt_align.bcx;
            let elt_size = size_of(bcx, e);
            bcx = elt_size.bcx;
            let aligned_off = align_to(bcx, off, elt_align.val);
            off = Add(bcx, aligned_off, elt_size.val);
            max_align = umax(bcx, max_align, elt_align.val);
        }
        off = align_to(bcx, off, max_align);
        ret rslt(bcx, off);
    }
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_param(p, _) {
        let szptr = field_of_tydesc(cx, t, false, abi::tydesc_field_size);
        ret rslt(szptr.bcx, Load(szptr.bcx, szptr.val));
      }
      ty::ty_rec(flds) {
        let tys: [ty::t] = [];
        for f: ty::field in flds { tys += [f.mt.ty]; }
        ret align_elements(cx, tys);
      }
      ty::ty_tup(elts) {
        let tys = [];
        for tp in elts { tys += [tp]; }
        ret align_elements(cx, tys);
      }
      ty::ty_tag(tid, tps) {
        let bcx = cx;
        // Compute max(variant sizes).

        let max_size: ValueRef = alloca(bcx, T_int());
        Store(bcx, C_int(0), max_size);
        let variants = ty::tag_variants(bcx_tcx(bcx), tid);
        for variant: ty::variant_info in variants {
            // Perform type substitution on the raw argument types.

            let raw_tys: [ty::t] = variant.args;
            let tys: [ty::t] = [];
            for raw_ty: ty::t in raw_tys {
                let t = ty::substitute_type_params(bcx_tcx(cx), tps, raw_ty);
                tys += [t];
            }
            let rslt = align_elements(bcx, tys);
            bcx = rslt.bcx;
            let this_size = rslt.val;
            let old_max_size = Load(bcx, max_size);
            Store(bcx, umax(bcx, this_size, old_max_size), max_size);
        }
        let max_size_val = Load(bcx, max_size);
        let total_size =
            if std::vec::len(variants) != 1u {
                Add(bcx, max_size_val, llsize_of(T_int()))
            } else { max_size_val };
        ret rslt(bcx, total_size);
      }
    }
}

fn dynamic_align_of(cx: @block_ctxt, t: ty::t) -> result {
// FIXME: Typestate constraint that shows this alt is
// exhaustive
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_param(p, _) {
        let aptr = field_of_tydesc(cx, t, false, abi::tydesc_field_align);
        ret rslt(aptr.bcx, Load(aptr.bcx, aptr.val));
      }
      ty::ty_rec(flds) {
        let a = C_int(1);
        let bcx = cx;
        for f: ty::field in flds {
            let align = align_of(bcx, f.mt.ty);
            bcx = align.bcx;
            a = umax(bcx, a, align.val);
        }
        ret rslt(bcx, a);
      }
      ty::ty_tag(_, _) {
        ret rslt(cx, C_int(1)); // FIXME: stub
      }
      ty::ty_tup(elts) {
        let a = C_int(1);
        let bcx = cx;
        for e in elts {
            let align = align_of(bcx, e);
            bcx = align.bcx;
            a = umax(bcx, a, align.val);
        }
        ret rslt(bcx, a);
      }
    }
}

// Simple wrapper around GEP that takes an array of ints and wraps them
// in C_int()
fn GEPi(cx: @block_ctxt, base: ValueRef, ixs: [int]) -> ValueRef {
    let v: [ValueRef] = [];
    for i: int in ixs { v += [C_int(i)]; }
    ret InBoundsGEP(cx, base, v);
}

// Increment a pointer by a given amount and then cast it to be a pointer
// to a given type.
fn bump_ptr(bcx: @block_ctxt, t: ty::t, base: ValueRef, sz: ValueRef) ->
   ValueRef {
    let raw = PointerCast(bcx, base, T_ptr(T_i8()));
    let bumped = GEP(bcx, raw, [sz]);
    let ccx = bcx_ccx(bcx);
    if check type_has_static_size(ccx, t) {
        let sp = bcx.sp;
        let typ = T_ptr(type_of(ccx, sp, t));
        PointerCast(bcx, bumped, typ)
    } else { bumped }
}

// Replacement for the LLVM 'GEP' instruction when field-indexing into a
// tuple-like structure (tup, rec) with a static index. This one is driven off
// ty::struct and knows what to do when it runs into a ty_param stuck in the
// middle of the thing it's GEP'ing into. Much like size_of and align_of,
// above.
fn GEP_tup_like(cx: @block_ctxt, t: ty::t, base: ValueRef, ixs: [int])
    : type_is_tup_like(cx, t) -> result {
    // It might be a static-known type. Handle this.
    if !ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        ret rslt(cx, GEPi(cx, base, ixs));
    }
    // It is a dynamic-containing type that, if we convert directly to an LLVM
    // TypeRef, will be all wrong; there's no proper LLVM type to represent
    // it, and the lowering function will stick in i8* values for each
    // ty_param, which is not right; the ty_params are all of some dynamic
    // size.
    //
    // What we must do instead is sadder. We must look through the indices
    // manually and split the input type into a prefix and a target. We then
    // measure the prefix size, bump the input pointer by that amount, and
    // cast to a pointer-to-target type.

    // Given a type, an index vector and an element number N in that vector,
    // calculate index X and the type that results by taking the first X-1
    // elements of the type and splitting the Xth off. Return the prefix as
    // well as the innermost Xth type.

    fn split_type(ccx: @crate_ctxt, t: ty::t, ixs: [int], n: uint) ->
       {prefix: [ty::t], target: ty::t} {
        let len: uint = std::vec::len::<int>(ixs);
        // We don't support 0-index or 1-index GEPs: The former is nonsense
        // and the latter would only be meaningful if we supported non-0
        // values for the 0th index (we don't).

        assert (len > 1u);
        if n == 0u {
            // Since we're starting from a value that's a pointer to a
            // *single* structure, the first index (in GEP-ese) should just be
            // 0, to yield the pointee.

            assert (ixs[n] == 0);
            ret split_type(ccx, t, ixs, n + 1u);
        }
        assert (n < len);
        let ix: int = ixs[n];
        let prefix: [ty::t] = [];
        let i: int = 0;
        while i < ix {
            prefix += [ty::get_element_type(ccx.tcx, t, i as uint)];
            i += 1;
        }
        let selected = ty::get_element_type(ccx.tcx, t, i as uint);
        if n == len - 1u {
            // We are at the innermost index.

            ret {prefix: prefix, target: selected};
        } else {
            // Not the innermost index; call self recursively to dig deeper.
            // Once we get an inner result, append it current prefix and
            // return to caller.

            let inner = split_type(ccx, selected, ixs, n + 1u);
            prefix += inner.prefix;
            ret {prefix: prefix with inner};
        }
    }
    // We make a fake prefix tuple-type here; luckily for measuring sizes
    // the tuple parens are associative so it doesn't matter that we've
    // flattened the incoming structure.

    let s = split_type(bcx_ccx(cx), t, ixs, 0u);

    let args = [];
    for typ: ty::t in s.prefix { args += [typ]; }
    let prefix_ty = ty::mk_tup(bcx_tcx(cx), args);

    let bcx = cx;
    let sz = size_of(bcx, prefix_ty);
    ret rslt(sz.bcx, bump_ptr(sz.bcx, s.target, base, sz.val));
}


// Replacement for the LLVM 'GEP' instruction when field indexing into a tag.
// This function uses GEP_tup_like() above and automatically performs casts as
// appropriate. @llblobptr is the data part of a tag value; its actual type is
// meaningless, as it will be cast away.
fn GEP_tag(cx: @block_ctxt, llblobptr: ValueRef, tag_id: ast::def_id,
           variant_id: ast::def_id, ty_substs: [ty::t],
           ix: uint) : valid_variant_index(ix, cx, tag_id, variant_id) ->
   result {
    let variant = ty::tag_variant_with_id(bcx_tcx(cx), tag_id, variant_id);
    // Synthesize a tuple type so that GEP_tup_like() can work its magic.
    // Separately, store the type of the element we're interested in.

    let arg_tys = variant.args;

    let true_arg_tys: [ty::t] = [];
    for aty: ty::t in arg_tys {
        let arg_ty = ty::substitute_type_params(bcx_tcx(cx), ty_substs, aty);
        true_arg_tys += [arg_ty];
    }

    // We know that ix < len(variant.args) -- so
    // it's safe to do this. (Would be nice to have
    // typestate guarantee that a dynamic bounds check
    // error can't happen here, but that's in the future.)
    let elem_ty = true_arg_tys[ix];

    let tup_ty = ty::mk_tup(bcx_tcx(cx), true_arg_tys);
    // Cast the blob pointer to the appropriate type, if we need to (i.e. if
    // the blob pointer isn't dynamically sized).

    let llunionptr: ValueRef;
    let sp = cx.sp;
    let ccx = bcx_ccx(cx);
    if check type_has_static_size(ccx, tup_ty) {
        let llty = type_of(ccx, sp, tup_ty);
        llunionptr = TruncOrBitCast(cx, llblobptr, T_ptr(llty));
    } else { llunionptr = llblobptr; }

    // Do the GEP_tup_like().
    // Silly check -- postcondition on mk_tup?
    check type_is_tup_like(cx, tup_ty);
    let rs = GEP_tup_like(cx, tup_ty, llunionptr, [0, ix as int]);
    // Cast the result to the appropriate type, if necessary.

    let rs_ccx = bcx_ccx(rs.bcx);
    let val =
        if check type_has_static_size(rs_ccx, elem_ty) {
            let llelemty = type_of(rs_ccx, sp, elem_ty);
            PointerCast(rs.bcx, rs.val, T_ptr(llelemty))
        } else { rs.val };

    ret rslt(rs.bcx, val);
}

// trans_shared_malloc: expects a type indicating which pointer type we want
// and a size indicating how much space we want malloc'd.
fn trans_shared_malloc(cx: @block_ctxt, llptr_ty: TypeRef, llsize: ValueRef)
   -> result {
    // FIXME: need a table to collect tydesc globals.

    let tydesc = C_null(T_ptr(bcx_ccx(cx).tydesc_type));
    let rval =
        Call(cx, bcx_ccx(cx).upcalls.shared_malloc,
             [cx.fcx.lltaskptr, llsize, tydesc]);
    ret rslt(cx, PointerCast(cx, rval, llptr_ty));
}

// trans_malloc_boxed_raw: expects an unboxed type and returns a pointer to
// enough space for something of that type, along with space for a reference
// count; in other words, it allocates a box for something of that type.
fn trans_malloc_boxed_raw(cx: @block_ctxt, t: ty::t) -> result {
    let bcx = cx;

    // Synthesize a fake box type structurally so we have something
    // to measure the size of.

    // We synthesize two types here because we want both the type of the
    // pointer and the pointee.  boxed_body is the type that we measure the
    // size of; box_ptr is the type that's converted to a TypeRef and used as
    // the pointer cast target in trans_raw_malloc.

    // The mk_int here is the space being
    // reserved for the refcount.
    let boxed_body = ty::mk_tup(bcx_tcx(bcx), [ty::mk_int(bcx_tcx(cx)), t]);
    let box_ptr = ty::mk_imm_box(bcx_tcx(bcx), t);
    let r = size_of(cx, boxed_body);
    let llsz = r.val; bcx = r.bcx;

    // Grab the TypeRef type of box_ptr, because that's what trans_raw_malloc
    // wants.
    // FIXME: Could avoid this check with a postcondition on mk_imm_box?
    // (requires Issue #586)
    let ccx = bcx_ccx(bcx);
    let sp = bcx.sp;
    check (type_has_static_size(ccx, box_ptr));
    let llty = type_of(ccx, sp, box_ptr);

    let ti = none;
    let tydesc_result = get_tydesc(bcx, t, true, tps_normal, ti);
    let lltydesc = tydesc_result.result.val; bcx = tydesc_result.result.bcx;

    let rval = Call(cx, ccx.upcalls.malloc,
                    [cx.fcx.lltaskptr, llsz, lltydesc]);
    ret rslt(cx, PointerCast(cx, rval, llty));
}

// trans_malloc_boxed: usefully wraps trans_malloc_box_raw; allocates a box,
// initializes the reference count to 1, and pulls out the body and rc
fn trans_malloc_boxed(cx: @block_ctxt, t: ty::t) ->
   {bcx: @block_ctxt, box: ValueRef, body: ValueRef} {
    let res = trans_malloc_boxed_raw(cx, t);
    let box = res.val;
    let rc = GEPi(res.bcx, box, [0, abi::box_rc_field_refcnt]);
    Store(res.bcx, C_int(1), rc);
    let body = GEPi(res.bcx, box, [0, abi::box_rc_field_body]);
    ret {bcx: res.bcx, box: res.val, body: body};
}

// Type descriptor and type glue stuff

// Given a type and a field index into its corresponding type descriptor,
// returns an LLVM ValueRef of that field from the tydesc, generating the
// tydesc if necessary.
fn field_of_tydesc(cx: @block_ctxt, t: ty::t, escapes: bool, field: int) ->
   result {
    let ti = none::<@tydesc_info>;
    let tydesc = get_tydesc(cx, t, escapes, tps_normal, ti).result;
    ret rslt(tydesc.bcx,
             GEP(tydesc.bcx, tydesc.val, [C_int(0), C_int(field)]));
}


// Given a type containing ty params, build a vector containing a ValueRef for
// each of the ty params it uses (from the current frame) and a vector of the
// indices of the ty params present in the type. This is used solely for
// constructing derived tydescs.
fn linearize_ty_params(cx: @block_ctxt, t: ty::t) ->
   {params: [uint], descs: [ValueRef]} {
    let param_vals: [ValueRef] = [];
    let param_defs: [uint] = [];
    type rr =
        {cx: @block_ctxt, mutable vals: [ValueRef], mutable defs: [uint]};

    fn linearizer(r: @rr, t: ty::t) {
        alt ty::struct(bcx_tcx(r.cx), t) {
          ty::ty_param(pid, _) {
            let seen: bool = false;
            for d: uint in r.defs { if d == pid { seen = true; } }
            if !seen { r.vals += [r.cx.fcx.lltydescs[pid]]; r.defs += [pid]; }
          }
          _ { }
        }
    }
    let x = @{cx: cx, mutable vals: param_vals, mutable defs: param_defs};
    let f = bind linearizer(x, _);
    ty::walk_ty(bcx_tcx(cx), f, t);
    ret {params: x.defs, descs: x.vals};
}

fn trans_stack_local_derived_tydesc(cx: @block_ctxt, llsz: ValueRef,
                                    llalign: ValueRef, llroottydesc: ValueRef,
                                    llfirstparam: ValueRef, n_params: uint,
                                    obj_params: uint) -> ValueRef {
    let llmyroottydesc = alloca(cx, bcx_ccx(cx).tydesc_type);

    // By convention, desc 0 is the root descriptor.
    llroottydesc = Load(cx, llroottydesc);
    Store(cx, llroottydesc, llmyroottydesc);

    // Store a pointer to the rest of the descriptors.
    store_inbounds(cx, llfirstparam, llmyroottydesc,
                   [C_int(0), C_int(abi::tydesc_field_first_param)]);
    store_inbounds(cx, C_uint(n_params), llmyroottydesc,
                   [C_int(0), C_int(abi::tydesc_field_n_params)]);
    store_inbounds(cx, llsz, llmyroottydesc,
                   [C_int(0), C_int(abi::tydesc_field_size)]);
    store_inbounds(cx, llalign, llmyroottydesc,
                   [C_int(0), C_int(abi::tydesc_field_align)]);
    store_inbounds(cx, C_uint(obj_params), llmyroottydesc,
                   [C_int(0), C_int(abi::tydesc_field_obj_params)]);
    ret llmyroottydesc;
}

// Objects and closures store their type parameters differently (in the object
// or closure itself rather than in the type descriptor).
tag ty_param_storage { tps_normal; tps_obj(uint); tps_fn(uint); }

fn get_derived_tydesc(cx: @block_ctxt, t: ty::t, escapes: bool,
                      storage: ty_param_storage,
                      &static_ti: option::t<@tydesc_info>) -> result {
    alt cx.fcx.derived_tydescs.find(t) {
      some(info) {


        // If the tydesc escapes in this context, the cached derived
        // tydesc also has to be one that was marked as escaping.
        if !(escapes && !info.escapes) && storage == tps_normal {
            ret rslt(cx, info.lltydesc);
        }
      }
      none. {/* fall through */ }
    }

    bcx_ccx(cx).stats.n_derived_tydescs += 1u;
    let bcx = new_raw_block_ctxt(cx.fcx, cx.fcx.llderivedtydescs);
    let tys = linearize_ty_params(bcx, t);
    let root_ti = get_static_tydesc(bcx, t, tys.params);
    static_ti = some::<@tydesc_info>(root_ti);
    lazily_emit_all_tydesc_glue(cx, static_ti);
    let root = root_ti.tydesc;
    let sz = size_of(bcx, t);
    bcx = sz.bcx;
    let align = align_of(bcx, t);
    bcx = align.bcx;

    // Store the captured type descriptors in an alloca if the caller isn't
    // promising to do so itself.
    let n_params = ty::count_ty_params(bcx_tcx(bcx), t);

    assert (n_params == std::vec::len::<uint>(tys.params));
    assert (n_params == std::vec::len::<ValueRef>(tys.descs));

    let llparamtydescs =
        alloca(bcx, T_array(T_ptr(bcx_ccx(bcx).tydesc_type), n_params + 1u));
    let i = 0;

    // If the type descriptor escapes, we need to add in the root as
    // the first parameter, because upcall_get_type_desc() expects it.
    if escapes {
        Store(bcx, root, GEPi(bcx, llparamtydescs, [0, 0]));
        i += 1;
    }

    for td: ValueRef in tys.descs {
        Store(bcx, td, GEPi(bcx, llparamtydescs, [0, i]));
        i += 1;
    }

    let llfirstparam =
        PointerCast(bcx, llparamtydescs,
                    T_ptr(T_ptr(bcx_ccx(bcx).tydesc_type)));

    // The top bit indicates whether this type descriptor describes an object
    // (0) or a function (1).
    let obj_params;
    alt storage {
      tps_normal. { obj_params = 0u; }
      tps_obj(np) { obj_params = np; }
      tps_fn(np) { obj_params = 0x80000000u | np; }
    }

    let v;
    if escapes {
        let td_val =
            Call(bcx, bcx_ccx(bcx).upcalls.get_type_desc,
                 [bcx.fcx.lltaskptr, C_null(T_ptr(T_nil())), sz.val,
                  align.val, C_uint(1u + n_params), llfirstparam,
                  C_uint(obj_params)]);
        v = td_val;
    } else {
        v =
            trans_stack_local_derived_tydesc(bcx, sz.val, align.val, root,
                                             llfirstparam, n_params,
                                             obj_params);
    }
    bcx.fcx.derived_tydescs.insert(t, {lltydesc: v, escapes: escapes});
    ret rslt(cx, v);
}

type get_tydesc_result = {kind: tydesc_kind, result: result};

fn get_tydesc(cx: @block_ctxt, orig_t: ty::t, escapes: bool,
              storage: ty_param_storage, &static_ti: option::t<@tydesc_info>)
   -> get_tydesc_result {

    let t = ty::strip_cname(bcx_tcx(cx), orig_t);

    // Is the supplied type a type param? If so, return the passed-in tydesc.
    alt ty::type_param(bcx_tcx(cx), t) {
      some(id) {
        if id < vec::len(cx.fcx.lltydescs) {
            ret {kind: tk_param, result: rslt(cx, cx.fcx.lltydescs[id])};
        } else {
            bcx_tcx(cx).sess.span_bug(cx.sp,
                                      "Unbound typaram in get_tydesc: " +
                                          "orig_t = " +
                                          ty_to_str(bcx_tcx(cx), orig_t) +
                                          " ty_param = " +
                                          std::uint::str(id));
        }
      }
      none. {/* fall through */ }
    }

    // Does it contain a type param? If so, generate a derived tydesc.
    if ty::type_contains_params(bcx_tcx(cx), t) {
        ret {kind: tk_derived,
             result: get_derived_tydesc(cx, t, escapes, storage, static_ti)};
    }
    // Otherwise, generate a tydesc if necessary, and return it.
    let info = get_static_tydesc(cx, t, []);
    static_ti = some::<@tydesc_info>(info);
    ret {kind: tk_static, result: rslt(cx, info.tydesc)};
}

fn get_static_tydesc(cx: @block_ctxt, orig_t: ty::t, ty_params: [uint]) ->
   @tydesc_info {
    let t = ty::strip_cname(bcx_tcx(cx), orig_t);


    alt bcx_ccx(cx).tydescs.find(t) {
      some(info) { ret info; }
      none. {
        bcx_ccx(cx).stats.n_static_tydescs += 1u;
        let info = declare_tydesc(cx.fcx.lcx, cx.sp, t, ty_params);
        bcx_ccx(cx).tydescs.insert(t, info);
        ret info;
      }
    }
}

fn set_no_inline(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f,
                              lib::llvm::LLVMNoInlineAttribute as
                                  lib::llvm::llvm::Attribute);
}

// Tell LLVM to emit the information necessary to unwind the stack for the
// function f.
fn set_uwtable(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f,
                              lib::llvm::LLVMUWTableAttribute as
                                  lib::llvm::llvm::Attribute);
}

fn set_always_inline(f: ValueRef) {
    llvm::LLVMAddFunctionAttr(f,
                              lib::llvm::LLVMAlwaysInlineAttribute as
                                  lib::llvm::llvm::Attribute);
}

fn set_glue_inlining(cx: @local_ctxt, f: ValueRef, t: ty::t) {
    if ty::type_is_structural(cx.ccx.tcx, t) {
        set_no_inline(f);
    } else { set_always_inline(f); }
}


// Generates the declaration for (but doesn't emit) a type descriptor.
fn declare_tydesc(cx: @local_ctxt, sp: span, t: ty::t, ty_params: [uint]) ->
   @tydesc_info {
    log "+++ declare_tydesc " + ty_to_str(cx.ccx.tcx, t);
    let ccx = cx.ccx;
    let llsize;
    let llalign;
    if check type_has_static_size(ccx, t) {
        let llty = type_of(ccx, sp, t);
        llsize = llsize_of(llty);
        llalign = llalign_of(llty);
    } else {
        // These will be overwritten as the derived tydesc is generated, so
        // we create placeholder values.

        llsize = C_int(0);
        llalign = C_int(0);
    }
    let name;
    if cx.ccx.sess.get_opts().debuginfo {
        name = mangle_internal_name_by_type_only(cx.ccx, t, "tydesc");
        name = sanitize(name);
    } else { name = mangle_internal_name_by_seq(cx.ccx, "tydesc"); }
    let gvar =
        str::as_buf(name,
                    {|buf|
                        llvm::LLVMAddGlobal(ccx.llmod, ccx.tydesc_type, buf)
                    });
    let info =
        @{ty: t,
          tydesc: gvar,
          size: llsize,
          align: llalign,
          mutable take_glue: none::<ValueRef>,
          mutable drop_glue: none::<ValueRef>,
          mutable free_glue: none::<ValueRef>,
          mutable cmp_glue: none::<ValueRef>,
          ty_params: ty_params};
    log "--- declare_tydesc " + ty_to_str(cx.ccx.tcx, t);
    ret info;
}

tag glue_helper {
    default_helper(fn(@block_ctxt, ValueRef, ty::t));
    copy_helper(fn(@block_ctxt, ValueRef, ValueRef, ty::t));
}

fn declare_generic_glue(cx: @local_ctxt, t: ty::t, llfnty: TypeRef, name: str)
   -> ValueRef {
    let name = name;
    let fn_nm;
    if cx.ccx.sess.get_opts().debuginfo {
        fn_nm = mangle_internal_name_by_type_only(cx.ccx, t, "glue_" + name);
        fn_nm = sanitize(fn_nm);
    } else { fn_nm = mangle_internal_name_by_seq(cx.ccx, "glue_" + name); }
    let llfn = decl_cdecl_fn(cx.ccx.llmod, fn_nm, llfnty);
    set_glue_inlining(cx, llfn, t);
    ret llfn;
}

// FIXME: was this causing the leak?
fn make_generic_glue_inner(cx: @local_ctxt, sp: span, t: ty::t,
                           llfn: ValueRef, helper: glue_helper,
                           ty_params: [uint]) -> ValueRef {
    let fcx = new_fn_ctxt(cx, sp, llfn);
    llvm::LLVMSetLinkage(llfn,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    cx.ccx.stats.n_glues_created += 1u;
    // Any nontrivial glue is with values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.

    let ccx = cx.ccx;
    let llty =
        if check type_has_static_size(ccx, t) {
            T_ptr(type_of(ccx, sp, t))
        } else { T_ptr(T_i8()) };

    let ty_param_count = std::vec::len::<uint>(ty_params);
    let lltyparams = llvm::LLVMGetParam(llfn, 3u);
    let copy_args_bcx = new_raw_block_ctxt(fcx, fcx.llcopyargs);
    let lltydescs = [mutable];
    let p = 0u;
    while p < ty_param_count {
        let llparam = GEP(copy_args_bcx, lltyparams, [C_int(p as int)]);
        llparam = Load(copy_args_bcx, llparam);
        std::vec::grow_set(lltydescs, ty_params[p], 0 as ValueRef, llparam);
        p += 1u;
    }

    // TODO: Implement some kind of freeze operation in the standard library.
    let lltydescs_frozen = [];
    for lltydesc: ValueRef in lltydescs { lltydescs_frozen += [lltydesc]; }
    fcx.lltydescs = lltydescs_frozen;

    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;
    let llrawptr0 = llvm::LLVMGetParam(llfn, 4u);
    let llval0 = BitCast(bcx, llrawptr0, llty);
    alt helper {
      default_helper(helper) { helper(bcx, llval0, t); }
      copy_helper(helper) {
        let llrawptr1 = llvm::LLVMGetParam(llfn, 5u);
        let llval1 = BitCast(bcx, llrawptr1, llty);
        helper(bcx, llval0, llval1, t);
      }
    }
    finish_fn(fcx, lltop);
    ret llfn;
}

fn make_generic_glue(cx: @local_ctxt, sp: span, t: ty::t, llfn: ValueRef,
                     helper: glue_helper, ty_params: [uint], name: str) ->
   ValueRef {
    if !cx.ccx.sess.get_opts().stats {
        ret make_generic_glue_inner(cx, sp, t, llfn, helper, ty_params);
    }

    let start = time::get_time();
    let llval = make_generic_glue_inner(cx, sp, t, llfn, helper, ty_params);
    let end = time::get_time();
    log_fn_time(cx.ccx, "glue " + name + " " + ty_to_short_str(cx.ccx.tcx, t),
                start, end);
    ret llval;
}

fn emit_tydescs(ccx: @crate_ctxt) {
    for each pair: @{key: ty::t, val: @tydesc_info} in ccx.tydescs.items() {
        let glue_fn_ty = T_ptr(T_glue_fn(*ccx));
        let cmp_fn_ty = T_ptr(T_cmp_glue_fn(*ccx));
        let ti = pair.val;
        let take_glue =
            alt ti.take_glue {
              none. { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let drop_glue =
            alt ti.drop_glue {
              none. { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let free_glue =
            alt ti.free_glue {
              none. { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let cmp_glue =
            alt ti.cmp_glue {
              none. { ccx.stats.n_null_glues += 1u; C_null(cmp_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };

        let shape = shape::shape_of(ccx, pair.key, ti.ty_params);
        let shape_tables =
            llvm::LLVMConstPointerCast(ccx.shape_cx.llshapetables,
                                       T_ptr(T_i8()));

        let tydesc =
            C_named_struct(ccx.tydesc_type,
                           [C_null(T_ptr(T_ptr(ccx.tydesc_type))),
                            ti.size, // size
                            ti.align, // align
                            take_glue, // take_glue
                            drop_glue, // drop_glue
                            free_glue, // free_glue
                            C_null(T_ptr(T_i8())), // unused
                            C_null(glue_fn_ty), // sever_glue
                            C_null(glue_fn_ty), // mark_glue
                            C_null(glue_fn_ty), // is_stateful
                            cmp_glue, // cmp_glue
                            C_shape(ccx, shape), // shape
                            shape_tables, // shape_tables
                            C_int(0), // n_params
                            C_int(0)]); // n_obj_params

        let gvar = ti.tydesc;
        llvm::LLVMSetInitializer(gvar, tydesc);
        llvm::LLVMSetGlobalConstant(gvar, True);
        llvm::LLVMSetLinkage(gvar,
                             lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    }
}

fn make_take_glue(cx: @block_ctxt, v: ValueRef, t: ty::t) {
    let bcx = cx;
    // NB: v is an *alias* of type t here, not a direct value.
    if ty::type_is_boxed(bcx_tcx(bcx), t) {
        bcx = incr_refcnt_of_boxed(bcx, Load(bcx, v));
    } else if ty::type_is_structural(bcx_tcx(bcx), t) {
        bcx = iter_structural_ty(bcx, v, t, take_ty);
    } else if ty::type_is_vec(bcx_tcx(bcx), t) {
        bcx = tvec::duplicate(bcx, v);
        bcx = tvec::iter_vec(bcx, v, t, take_ty);
    }

    build_return(bcx);
}

fn incr_refcnt_of_boxed(cx: @block_ctxt, box_ptr: ValueRef) -> @block_ctxt {
    let rc_ptr =
        GEP(cx, box_ptr, [C_int(0), C_int(abi::box_rc_field_refcnt)]);
    let rc = Load(cx, rc_ptr);
    rc = Add(cx, rc, C_int(1));
    Store(cx, rc, rc_ptr);
    ret cx;
}

fn make_free_glue(bcx: @block_ctxt, v0: ValueRef, t: ty::t) {
    // NB: v is an *alias* of type t here, not a direct value.
    let bcx =
        alt ty::struct(bcx_tcx(bcx), t) {
          ty::ty_box(body_mt) {
            let v = Load(bcx, v0);
            let body = GEP(bcx, v, [C_int(0), C_int(abi::box_rc_field_body)]);
            let bcx = drop_ty(bcx, body, body_mt.ty);
            if !bcx_ccx(bcx).sess.get_opts().do_gc {
                trans_non_gc_free(bcx, v)
            } else { bcx }
          }
          ty::ty_uniq(_) { fail "free uniq unimplemented"; }
          ty::ty_obj(_) {
            // Call through the obj's own fields-drop glue first.
            // Then free the body.
            let box_cell =
                GEP(bcx, v0, [C_int(0), C_int(abi::obj_field_box)]);
            let b = Load(bcx, box_cell);
            let ccx = bcx_ccx(bcx);
            let llbox_ty = T_opaque_obj_ptr(*ccx);
            b = PointerCast(bcx, b, llbox_ty);
            let body = GEP(bcx, b, [C_int(0), C_int(abi::box_rc_field_body)]);
            let tydescptr =
                GEP(bcx, body, [C_int(0), C_int(abi::obj_body_elt_tydesc)]);
            let tydesc = Load(bcx, tydescptr);
            let ti = none;
            call_tydesc_glue_full(bcx, body, tydesc,
                                  abi::tydesc_field_drop_glue, ti);
            if !bcx_ccx(bcx).sess.get_opts().do_gc {
                trans_non_gc_free(bcx, b)
            } else { bcx }
          }
          ty::ty_fn(_, _, _, _, _) {
            // Call through the closure's own fields-drop glue first.
            // Then free the body.
            let box_cell = GEP(bcx, v0, [C_int(0), C_int(abi::fn_field_box)]);
            let v = Load(bcx, box_cell);
            let body = GEP(bcx, v, [C_int(0), C_int(abi::box_rc_field_body)]);
            let bindings =
                GEP(bcx, body, [C_int(0), C_int(abi::closure_elt_bindings)]);
            let tydescptr =
                GEP(bcx, body, [C_int(0), C_int(abi::closure_elt_tydesc)]);
            let ti = none;
            call_tydesc_glue_full(bcx, bindings, Load(bcx, tydescptr),
                                  abi::tydesc_field_drop_glue, ti);
            if !bcx_ccx(bcx).sess.get_opts().do_gc {
                trans_non_gc_free(bcx, v)
            } else { bcx }
          }
          _ { bcx }
        };

    build_return(bcx);
}

fn make_drop_glue(bcx: @block_ctxt, v0: ValueRef, t: ty::t) {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let ccx = bcx_ccx(bcx);
    let bcx =
        alt ty::struct(ccx.tcx, t) {
          ty::ty_vec(_) { tvec::make_drop_glue(bcx, v0, t) }
          ty::ty_str. { tvec::make_drop_glue(bcx, v0, t) }
          ty::ty_box(_) { decr_refcnt_maybe_free(bcx, v0, v0, t) }
          ty::ty_uniq(_) {
            let vptr = Load(bcx, v0);
            let bcx = trans_shared_free(bcx, vptr);
            Store(bcx, C_null(val_ty(vptr)), v0);
            bcx
          }
          ty::ty_obj(_) {
            let box_cell =
                GEP(bcx, v0, [C_int(0), C_int(abi::obj_field_box)]);
            decr_refcnt_maybe_free(bcx, box_cell, v0, t)
          }
          ty::ty_res(did, inner, tps) {
            trans_res_drop(bcx, v0, did, inner, tps)
          }
          ty::ty_fn(_, _, _, _, _) {
            let box_cell = GEP(bcx, v0, [C_int(0), C_int(abi::fn_field_box)]);
            decr_refcnt_maybe_free(bcx, box_cell, v0, t)
          }
          _ {
            if ty::type_has_pointers(ccx.tcx, t) &&
                   ty::type_is_structural(ccx.tcx, t) {
                iter_structural_ty(bcx, v0, t, drop_ty)
            } else { bcx }
          }
        };
    build_return(bcx);
}

fn trans_res_drop(cx: @block_ctxt, rs: ValueRef, did: ast::def_id,
                  inner_t: ty::t, tps: [ty::t]) -> @block_ctxt {
    let ccx = bcx_ccx(cx);
    let inner_t_s = ty::substitute_type_params(ccx.tcx, tps, inner_t);
    let tup_ty = ty::mk_tup(ccx.tcx, [ty::mk_int(ccx.tcx), inner_t_s]);
    let drop_cx = new_sub_block_ctxt(cx, "drop res");
    let next_cx = new_sub_block_ctxt(cx, "next");

    // Silly check
    check type_is_tup_like(cx, tup_ty);
    let drop_flag = GEP_tup_like(cx, tup_ty, rs, [0, 0]);
    cx = drop_flag.bcx;
    let null_test = IsNull(cx, Load(cx, drop_flag.val));
    CondBr(cx, null_test, next_cx.llbb, drop_cx.llbb);
    cx = drop_cx;

    check type_is_tup_like(cx, tup_ty);
    let val = GEP_tup_like(cx, tup_ty, rs, [0, 1]);
    cx = val.bcx;
    // Find and call the actual destructor.
    let dtor_addr = trans_common::get_res_dtor(ccx, cx.sp, did, inner_t);
    let args = [cx.fcx.llretptr, cx.fcx.lltaskptr, null_env_ptr(cx)];
    for tp: ty::t in tps {
        let ti: option::t<@tydesc_info> = none;
        let td = get_tydesc(cx, tp, false, tps_normal, ti).result;
        args += [td.val];
        cx = td.bcx;
    }
    // Kludge to work around the fact that we know the precise type of the
    // value here, but the dtor expects a type that still has opaque pointers
    // for type variables.
    let val_llty = lib::llvm::fn_ty_param_tys
        (llvm::LLVMGetElementType
         (llvm::LLVMTypeOf(dtor_addr)))[std::vec::len(args)];
    let val_cast = BitCast(cx, val.val, val_llty);
    FastCall(cx, dtor_addr, args + [val_cast]);

    cx = drop_ty(cx, val.val, inner_t_s);
    Store(cx, C_int(0), drop_flag.val);
    Br(cx, next_cx.llbb);
    ret next_cx;
}

fn decr_refcnt_maybe_free(cx: @block_ctxt, box_ptr_alias: ValueRef,
                          full_alias: ValueRef, t: ty::t) -> @block_ctxt {
    let ccx = bcx_ccx(cx);
    let rc_adj_cx = new_sub_block_ctxt(cx, "rc--");
    let free_cx = new_sub_block_ctxt(cx, "free");
    let next_cx = new_sub_block_ctxt(cx, "next");
    let box_ptr = Load(cx, box_ptr_alias);
    let llbox_ty = T_opaque_obj_ptr(*ccx);
    box_ptr = PointerCast(cx, box_ptr, llbox_ty);
    let null_test = IsNull(cx, box_ptr);
    CondBr(cx, null_test, next_cx.llbb, rc_adj_cx.llbb);
    let rc_ptr =
        GEP(rc_adj_cx, box_ptr, [C_int(0), C_int(abi::box_rc_field_refcnt)]);
    let rc = Load(rc_adj_cx, rc_ptr);
    rc = Sub(rc_adj_cx, rc, C_int(1));
    Store(rc_adj_cx, rc, rc_ptr);
    let zero_test = ICmp(rc_adj_cx, lib::llvm::LLVMIntEQ, C_int(0), rc);
    CondBr(rc_adj_cx, zero_test, free_cx.llbb, next_cx.llbb);
    let free_cx = free_ty(free_cx, full_alias, t);
    Br(free_cx, next_cx.llbb);
    ret next_cx;
}


// Structural comparison: a rather involved form of glue.
fn maybe_name_value(cx: @crate_ctxt, v: ValueRef, s: str) {
    if cx.sess.get_opts().save_temps {
        let _: () = str::as_buf(s, {|buf| llvm::LLVMSetValueName(v, buf) });
    }
}


// Used only for creating scalar comparison glue.
tag scalar_type { nil_type; signed_int; unsigned_int; floating_point; }


fn compare_scalar_types(cx: @block_ctxt, lhs: ValueRef, rhs: ValueRef,
                        t: ty::t, llop: ValueRef) -> result {
    let f = bind compare_scalar_values(cx, lhs, rhs, _, llop);

    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_nil. { ret f(nil_type); }
      ty::ty_bool. | ty::ty_uint. | ty::ty_ptr(_) | ty::ty_char. {
        ret f(unsigned_int);
      }
      ty::ty_int. { ret f(signed_int); }
      ty::ty_float. { ret f(floating_point); }
      ty::ty_machine(_) {
        if ty::type_is_fp(bcx_tcx(cx), t) {
            // Floating point machine types
            ret f(floating_point);
        } else if ty::type_is_signed(bcx_tcx(cx), t) {
            // Signed, integral machine types
            ret f(signed_int);
        } else {
            // Unsigned, integral machine types
            ret f(unsigned_int);
        }
      }
      ty::ty_type. {
        trans_fail(cx, none, "attempt to compare values of type type");

        // This is a bit lame, because we return a dummy block to the
        // caller that's actually unreachable, but I don't think it
        // matters.
        ret rslt(new_sub_block_ctxt(cx, "after_fail_dummy"), C_bool(false));
      }
      ty::ty_native(_) {
        trans_fail(cx, none::<span>,
                   "attempt to compare values of type native");
        ret rslt(new_sub_block_ctxt(cx, "after_fail_dummy"), C_bool(false));
      }
      _ {
        // Should never get here, because t is scalar.
        bcx_ccx(cx).sess.bug("non-scalar type passed to \
                                 compare_scalar_types");
      }
    }
}


// A helper function to do the actual comparison of scalar values.
fn compare_scalar_values(cx: @block_ctxt, lhs: ValueRef, rhs: ValueRef,
                         nt: scalar_type, llop: ValueRef) -> result {
    let eq_cmp;
    let lt_cmp;
    let le_cmp;
    alt nt {
      nil_type. {
        // We don't need to do actual comparisons for nil.
        // () == () holds but () < () does not.
        eq_cmp = 1u;
        lt_cmp = 0u;
        le_cmp = 1u;
      }
      floating_point. {
        eq_cmp = lib::llvm::LLVMRealUEQ;
        lt_cmp = lib::llvm::LLVMRealULT;
        le_cmp = lib::llvm::LLVMRealULE;
      }
      signed_int. {
        eq_cmp = lib::llvm::LLVMIntEQ;
        lt_cmp = lib::llvm::LLVMIntSLT;
        le_cmp = lib::llvm::LLVMIntSLE;
      }
      unsigned_int. {
        eq_cmp = lib::llvm::LLVMIntEQ;
        lt_cmp = lib::llvm::LLVMIntULT;
        le_cmp = lib::llvm::LLVMIntULE;
      }
    }
    // FIXME: This wouldn't be necessary if we could bind methods off of
    // objects and therefore abstract over FCmp and ICmp (issue #435).  Then
    // we could just write, e.g., "cmp_fn = bind FCmp(cx, _, _, _);" in
    // the above, and "auto eq_result = cmp_fn(eq_cmp, lhs, rhs);" in the
    // below.

    fn generic_cmp(cx: @block_ctxt, nt: scalar_type, op: uint, lhs: ValueRef,
                   rhs: ValueRef) -> ValueRef {
        let r: ValueRef;
        if nt == nil_type {
            r = C_bool(op != 0u);
        } else if nt == floating_point {
            r = FCmp(cx, op, lhs, rhs);
        } else { r = ICmp(cx, op, lhs, rhs); }
        ret r;
    }
    let last_cx = new_sub_block_ctxt(cx, "last");
    let eq_cx = new_sub_block_ctxt(cx, "eq");
    let eq_result = generic_cmp(eq_cx, nt, eq_cmp, lhs, rhs);
    Br(eq_cx, last_cx.llbb);
    let lt_cx = new_sub_block_ctxt(cx, "lt");
    let lt_result = generic_cmp(lt_cx, nt, lt_cmp, lhs, rhs);
    Br(lt_cx, last_cx.llbb);
    let le_cx = new_sub_block_ctxt(cx, "le");
    let le_result = generic_cmp(le_cx, nt, le_cmp, lhs, rhs);
    Br(le_cx, last_cx.llbb);
    let unreach_cx = new_sub_block_ctxt(cx, "unreach");
    Unreachable(unreach_cx);
    let llswitch = Switch(cx, llop, unreach_cx.llbb, 3u);
    llvm::LLVMAddCase(llswitch, C_u8(abi::cmp_glue_op_eq), eq_cx.llbb);
    llvm::LLVMAddCase(llswitch, C_u8(abi::cmp_glue_op_lt), lt_cx.llbb);
    llvm::LLVMAddCase(llswitch, C_u8(abi::cmp_glue_op_le), le_cx.llbb);
    let last_result =
        Phi(last_cx, T_i1(), [eq_result, lt_result, le_result],
            [eq_cx.llbb, lt_cx.llbb, le_cx.llbb]);
    ret rslt(last_cx, last_result);
}

type val_pair_fn = fn(@block_ctxt, ValueRef, ValueRef) -> @block_ctxt;
type val_and_ty_fn = fn(@block_ctxt, ValueRef, ty::t) -> @block_ctxt;

fn load_inbounds(cx: @block_ctxt, p: ValueRef, idxs: [ValueRef]) -> ValueRef {
    ret Load(cx, InBoundsGEP(cx, p, idxs));
}

fn store_inbounds(cx: @block_ctxt, v: ValueRef, p: ValueRef,
                  idxs: [ValueRef]) {
    Store(cx, v, InBoundsGEP(cx, p, idxs));
}

// Iterates through the elements of a structural type.
fn iter_structural_ty(cx: @block_ctxt, av: ValueRef, t: ty::t,
                      f: val_and_ty_fn) -> @block_ctxt {
    fn iter_boxpp(cx: @block_ctxt, box_cell: ValueRef, f: val_and_ty_fn) ->
       @block_ctxt {
        let box_ptr = Load(cx, box_cell);
        let tnil = ty::mk_nil(bcx_tcx(cx));
        let tbox = ty::mk_imm_box(bcx_tcx(cx), tnil);
        let inner_cx = new_sub_block_ctxt(cx, "iter box");
        let next_cx = new_sub_block_ctxt(cx, "next");
        let null_test = IsNull(cx, box_ptr);
        CondBr(cx, null_test, next_cx.llbb, inner_cx.llbb);
        let inner_cx = f(inner_cx, box_cell, tbox);
        Br(inner_cx, next_cx.llbb);
        ret next_cx;
    }

    fn iter_variant(cx: @block_ctxt, a_tup: ValueRef,
                    variant: ty::variant_info, tps: [ty::t], tid: ast::def_id,
                    f: val_and_ty_fn) -> @block_ctxt {
        if std::vec::len::<ty::t>(variant.args) == 0u { ret cx; }
        let fn_ty = variant.ctor_ty;
        let ccx = bcx_ccx(cx);
        alt ty::struct(ccx.tcx, fn_ty) {
          ty::ty_fn(_, args, _, _, _) {
            let j = 0u;
            let v_id = variant.id;
            for a: ty::arg in args {
                check (valid_variant_index(j, cx, tid, v_id));
                let rslt = GEP_tag(cx, a_tup, tid, v_id, tps, j);
                let llfldp_a = rslt.val;
                cx = rslt.bcx;
                let ty_subst = ty::substitute_type_params(ccx.tcx, tps, a.ty);
                cx = f(cx, llfldp_a, ty_subst);
                j += 1u;
            }
          }
        }
        ret cx;
    }

    /*
    Typestate constraint that shows the unimpl case doesn't happen?
    */
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_rec(fields) {
        let i: int = 0;
        for fld: ty::field in fields {
            // Silly check
            check type_is_tup_like(cx, t);
            let {bcx: bcx, val: llfld_a} = GEP_tup_like(cx, t, av, [0, i]);
            cx = f(bcx, llfld_a, fld.mt.ty);
            i += 1;
        }
      }
      ty::ty_tup(args) {
        let i = 0;
        for arg in args {
            // Silly check
            check type_is_tup_like(cx, t);
            let {bcx: bcx, val: llfld_a} = GEP_tup_like(cx, t, av, [0, i]);
            cx = f(bcx, llfld_a, arg);
            i += 1;
        }
      }
      ty::ty_res(_, inner, tps) {
        let tcx = bcx_tcx(cx);
        let inner1 = ty::substitute_type_params(tcx, tps, inner);
        let inner_t_s = ty::substitute_type_params(tcx, tps, inner);
        let tup_t = ty::mk_tup(tcx, [ty::mk_int(tcx), inner_t_s]);
        // Silly check
        check type_is_tup_like(cx, tup_t);
        let {bcx: bcx, val: llfld_a} = GEP_tup_like(cx, tup_t, av, [0, 1]);
        ret f(bcx, llfld_a, inner1);
      }
      ty::ty_tag(tid, tps) {
        let variants = ty::tag_variants(bcx_tcx(cx), tid);
        let n_variants = std::vec::len(variants);

        // Cast the tags to types we can GEP into.
        if n_variants == 1u {
            ret iter_variant(cx, av, variants[0], tps, tid, f);
        }

        let lltagty = T_opaque_tag_ptr(bcx_ccx(cx).tn);
        let av_tag = PointerCast(cx, av, lltagty);
        let lldiscrim_a_ptr = GEP(cx, av_tag, [C_int(0), C_int(0)]);
        let llunion_a_ptr = GEP(cx, av_tag, [C_int(0), C_int(1)]);
        let lldiscrim_a = Load(cx, lldiscrim_a_ptr);

        // NB: we must hit the discriminant first so that structural
        // comparison know not to proceed when the discriminants differ.
        cx = f(cx, lldiscrim_a_ptr, ty::mk_int(bcx_tcx(cx)));
        let unr_cx = new_sub_block_ctxt(cx, "tag-iter-unr");
        Unreachable(unr_cx);
        let llswitch = Switch(cx, lldiscrim_a, unr_cx.llbb, n_variants);
        let next_cx = new_sub_block_ctxt(cx, "tag-iter-next");
        let i = 0u;
        for variant: ty::variant_info in variants {
            let variant_cx =
                new_sub_block_ctxt(cx,
                                   "tag-iter-variant-" +
                                       uint::to_str(i, 10u));
            llvm::LLVMAddCase(llswitch, C_int(i as int), variant_cx.llbb);
            variant_cx =
                iter_variant(variant_cx, llunion_a_ptr, variant, tps, tid, f);
            Br(variant_cx, next_cx.llbb);
            i += 1u;
        }
        ret next_cx;
      }
      ty::ty_fn(_, _, _, _, _) | ty::ty_native_fn(_, _, _) {
        let box_cell_a = GEP(cx, av, [C_int(0), C_int(abi::fn_field_box)]);
        ret iter_boxpp(cx, box_cell_a, f);
      }
      ty::ty_obj(_) {
        let box_cell_a = GEP(cx, av, [C_int(0), C_int(abi::obj_field_box)]);
        ret iter_boxpp(cx, box_cell_a, f);
      }
      _ { bcx_ccx(cx).sess.unimpl("type in iter_structural_ty"); }
    }
    ret cx;
}

fn lazily_emit_all_tydesc_glue(cx: @block_ctxt,
                               static_ti: option::t<@tydesc_info>) {
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_take_glue, static_ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_drop_glue, static_ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_free_glue, static_ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_cmp_glue, static_ti);
}

fn lazily_emit_all_generic_info_tydesc_glues(cx: @block_ctxt,
                                             gi: generic_info) {
    for ti: option::t<@tydesc_info> in gi.static_tis {
        lazily_emit_all_tydesc_glue(cx, ti);
    }
}

fn lazily_emit_tydesc_glue(cx: @block_ctxt, field: int,
                           static_ti: option::t<@tydesc_info>) {
    alt static_ti {
      none. { }
      some(ti) {
        if field == abi::tydesc_field_take_glue {
            alt ti.take_glue {
              some(_) { }
              none. {
                log #fmt["+++ lazily_emit_tydesc_glue TAKE %s",
                         ty_to_str(bcx_tcx(cx), ti.ty)];
                let lcx = cx.fcx.lcx;
                let glue_fn =
                    declare_generic_glue(lcx, ti.ty, T_glue_fn(*lcx.ccx),
                                         "take");
                ti.take_glue = some::<ValueRef>(glue_fn);
                make_generic_glue(lcx, cx.sp, ti.ty, glue_fn,
                                  default_helper(make_take_glue),
                                  ti.ty_params, "take");
                log #fmt["--- lazily_emit_tydesc_glue TAKE %s",
                         ty_to_str(bcx_tcx(cx), ti.ty)];
              }
            }
        } else if field == abi::tydesc_field_drop_glue {
            alt ti.drop_glue {
              some(_) { }
              none. {
                log #fmt["+++ lazily_emit_tydesc_glue DROP %s",
                         ty_to_str(bcx_tcx(cx), ti.ty)];
                let lcx = cx.fcx.lcx;
                let glue_fn =
                    declare_generic_glue(lcx, ti.ty, T_glue_fn(*lcx.ccx),
                                         "drop");
                ti.drop_glue = some::<ValueRef>(glue_fn);
                make_generic_glue(lcx, cx.sp, ti.ty, glue_fn,
                                  default_helper(make_drop_glue),
                                  ti.ty_params, "drop");
                log #fmt["--- lazily_emit_tydesc_glue DROP %s",
                         ty_to_str(bcx_tcx(cx), ti.ty)];
              }
            }
        } else if field == abi::tydesc_field_free_glue {
            alt ti.free_glue {
              some(_) { }
              none. {
                log #fmt["+++ lazily_emit_tydesc_glue FREE %s",
                         ty_to_str(bcx_tcx(cx), ti.ty)];
                let lcx = cx.fcx.lcx;
                let glue_fn =
                    declare_generic_glue(lcx, ti.ty, T_glue_fn(*lcx.ccx),
                                         "free");
                ti.free_glue = some::<ValueRef>(glue_fn);
                make_generic_glue(lcx, cx.sp, ti.ty, glue_fn,
                                  default_helper(make_free_glue),
                                  ti.ty_params, "free");
                log #fmt["--- lazily_emit_tydesc_glue FREE %s",
                         ty_to_str(bcx_tcx(cx), ti.ty)];
              }
            }
        } else if field == abi::tydesc_field_cmp_glue {
            alt ti.cmp_glue {
              some(_) { }
              none. {
                log #fmt["+++ lazily_emit_tydesc_glue CMP %s",
                         ty_to_str(bcx_tcx(cx), ti.ty)];
                ti.cmp_glue = some(bcx_ccx(cx).upcalls.cmp_type);
                log #fmt["--- lazily_emit_tydesc_glue CMP %s",
                         ty_to_str(bcx_tcx(cx), ti.ty)];
              }
            }
        }
      }
    }
}

fn call_tydesc_glue_full(cx: @block_ctxt, v: ValueRef, tydesc: ValueRef,
                         field: int, static_ti: option::t<@tydesc_info>) {
    lazily_emit_tydesc_glue(cx, field, static_ti);

    let static_glue_fn = none;
    alt static_ti {
      none. {/* no-op */ }
      some(sti) {
        if field == abi::tydesc_field_take_glue {
            static_glue_fn = sti.take_glue;
        } else if field == abi::tydesc_field_drop_glue {
            static_glue_fn = sti.drop_glue;
        } else if field == abi::tydesc_field_free_glue {
            static_glue_fn = sti.free_glue;
        }
      }
    }

    let llrawptr = PointerCast(cx, v, T_ptr(T_i8()));
    let lltydescs =
        GEP(cx, tydesc, [C_int(0), C_int(abi::tydesc_field_first_param)]);
    lltydescs = Load(cx, lltydescs);

    let llfn;
    alt static_glue_fn {
      none. {
        let llfnptr = GEP(cx, tydesc, [C_int(0), C_int(field)]);
        llfn = Load(cx, llfnptr);
      }
      some(sgf) { llfn = sgf; }
    }

    Call(cx, llfn,
         [C_null(T_ptr(T_nil())), cx.fcx.lltaskptr, C_null(T_ptr(T_nil())),
          lltydescs, llrawptr]);
}

fn call_tydesc_glue(cx: @block_ctxt, v: ValueRef, t: ty::t, field: int) ->
   @block_ctxt {
    let ti: option::t<@tydesc_info> = none::<@tydesc_info>;
    let {bcx: bcx, val: td} = get_tydesc(cx, t, false, tps_normal, ti).result;
    call_tydesc_glue_full(bcx, v, td, field, ti);
    ret bcx;
}

fn call_cmp_glue(cx: @block_ctxt, lhs: ValueRef, rhs: ValueRef, t: ty::t,
                 llop: ValueRef) -> result {
    // We can't use call_tydesc_glue_full() and friends here because compare
    // glue has a special signature.

    let bcx = cx;

    let r = spill_if_immediate(bcx, lhs, t);
    let lllhs = r.val;
    bcx = r.bcx;
    r = spill_if_immediate(bcx, rhs, t);
    let llrhs = r.val;
    bcx = r.bcx;

    let llrawlhsptr = BitCast(bcx, lllhs, T_ptr(T_i8()));
    let llrawrhsptr = BitCast(bcx, llrhs, T_ptr(T_i8()));
    let ti = none::<@tydesc_info>;
    r = get_tydesc(bcx, t, false, tps_normal, ti).result;
    let lltydesc = r.val;
    bcx = r.bcx;
    lazily_emit_tydesc_glue(bcx, abi::tydesc_field_cmp_glue, ti);
    let lltydescs =
        GEP(bcx, lltydesc, [C_int(0), C_int(abi::tydesc_field_first_param)]);
    lltydescs = Load(bcx, lltydescs);

    let llfn;
    alt ti {
      none. {
        let llfnptr =
            GEP(bcx, lltydesc, [C_int(0), C_int(abi::tydesc_field_cmp_glue)]);
        llfn = Load(bcx, llfnptr);
      }
      some(sti) { llfn = option::get(sti.cmp_glue); }
    }

    let llcmpresultptr = alloca(bcx, T_i1());
    let llargs: [ValueRef] =
        [llcmpresultptr, bcx.fcx.lltaskptr, lltydesc, lltydescs, llrawlhsptr,
         llrawrhsptr, llop];
    Call(bcx, llfn, llargs);
    ret rslt(bcx, Load(bcx, llcmpresultptr));
}

// Compares two values. Performs the simple scalar comparison if the types are
// scalar and calls to comparison glue otherwise.
fn compare(cx: @block_ctxt, lhs: ValueRef, rhs: ValueRef, t: ty::t,
           llop: ValueRef) -> result {
    if ty::type_is_scalar(bcx_tcx(cx), t) {
        ret compare_scalar_types(cx, lhs, rhs, t, llop);
    }
    ret call_cmp_glue(cx, lhs, rhs, t, llop);
}

fn take_ty(cx: @block_ctxt, v: ValueRef, t: ty::t) -> @block_ctxt {
    if ty::type_has_pointers(bcx_tcx(cx), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_take_glue);
    }
    ret cx;
}

fn drop_ty(cx: @block_ctxt, v: ValueRef, t: ty::t) -> @block_ctxt {
    if ty::type_needs_drop(bcx_tcx(cx), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_drop_glue);
    }
    ret cx;
}

fn free_ty(cx: @block_ctxt, v: ValueRef, t: ty::t) -> @block_ctxt {
    if ty::type_has_pointers(bcx_tcx(cx), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_free_glue);
    }
    ret cx;
}

fn call_memmove(cx: @block_ctxt, dst: ValueRef, src: ValueRef,
                n_bytes: ValueRef) -> result {
    // FIXME: switch to the 64-bit variant when on such a platform.
    // TODO: Provide LLVM with better alignment information when the alignment
    // is statically known (it must be nothing more than a constant int, or
    // LLVM complains -- not even a constant element of a tydesc works).

    let i = bcx_ccx(cx).intrinsics;
    assert (i.contains_key("llvm.memmove.p0i8.p0i8.i32"));
    let memmove = i.get("llvm.memmove.p0i8.p0i8.i32");
    let src_ptr = PointerCast(cx, src, T_ptr(T_i8()));
    let dst_ptr = PointerCast(cx, dst, T_ptr(T_i8()));
    let size = IntCast(cx, n_bytes, T_i32());
    let align = C_int(1);
    let volatile = C_bool(false);
    ret rslt(cx,
             Call(cx, memmove, [dst_ptr, src_ptr, size, align, volatile]));
}

fn call_bzero(cx: @block_ctxt, dst: ValueRef, n_bytes: ValueRef,
              align_bytes: ValueRef) -> result {
    // FIXME: switch to the 64-bit variant when on such a platform.

    let i = bcx_ccx(cx).intrinsics;
    assert (i.contains_key("llvm.memset.p0i8.i32"));
    let memset = i.get("llvm.memset.p0i8.i32");
    let dst_ptr = PointerCast(cx, dst, T_ptr(T_i8()));
    let size = IntCast(cx, n_bytes, T_i32());
    let align =
        if lib::llvm::llvm::LLVMIsConstant(align_bytes) == True {
            IntCast(cx, align_bytes, T_i32())
        } else { IntCast(cx, C_int(0), T_i32()) };
    let volatile = C_bool(false);
    ret rslt(cx,
             Call(cx, memset, [dst_ptr, C_u8(0u), size, align, volatile]));
}

fn memmove_ty(cx: @block_ctxt, dst: ValueRef, src: ValueRef, t: ty::t) ->
   result {
    let ccx = bcx_ccx(cx);
    if check type_has_static_size(ccx, t) {
        if ty::type_is_structural(bcx_tcx(cx), t) {
            let sp = cx.sp;
            let llsz = llsize_of(type_of(ccx, sp, t));
            ret call_memmove(cx, dst, src, llsz);
        }

        ret rslt(cx, Store(cx, Load(cx, src), dst));
    }

    let llsz = size_of(cx, t);
    ret call_memmove(llsz.bcx, dst, src, llsz.val);
}

tag copy_action { INIT; DROP_EXISTING; }

// These are the types that are passed by pointer.
fn type_is_structural_or_param(tcx: ty::ctxt, t: ty::t) -> bool {
    if ty::type_is_structural(tcx, t) { ret true; }
    alt ty::struct(tcx, t) {
      ty::ty_param(_, _) { ret true; }
      _ { ret false; }
    }
}

fn copy_val(cx: @block_ctxt, action: copy_action, dst: ValueRef,
            src: ValueRef, t: ty::t) -> @block_ctxt {
    if type_is_structural_or_param(bcx_ccx(cx).tcx, t) &&
           action == DROP_EXISTING {
        let do_copy_cx = new_sub_block_ctxt(cx, "do_copy");
        let next_cx = new_sub_block_ctxt(cx, "next");
        let self_assigning =
            ICmp(cx, lib::llvm::LLVMIntNE, PointerCast(cx, dst, val_ty(src)),
                 src);
        CondBr(cx, self_assigning, do_copy_cx.llbb, next_cx.llbb);
        do_copy_cx = copy_val_no_check(do_copy_cx, action, dst, src, t);
        Br(do_copy_cx, next_cx.llbb);
        ret next_cx;
    }
    ret copy_val_no_check(cx, action, dst, src, t);
}

fn copy_val_no_check(cx: @block_ctxt, action: copy_action, dst: ValueRef,
                     src: ValueRef, t: ty::t) -> @block_ctxt {
    let ccx = bcx_ccx(cx);
    if ty::type_is_scalar(ccx.tcx, t) || ty::type_is_native(ccx.tcx, t) {
        Store(cx, src, dst);
        ret cx;
    }
    if ty::type_is_nil(ccx.tcx, t) || ty::type_is_bot(ccx.tcx, t) { ret cx; }
    if ty::type_is_boxed(ccx.tcx, t) {
        let bcx = cx;
        if action == DROP_EXISTING { bcx = drop_ty(cx, dst, t); }
        Store(bcx, src, dst);
        ret take_ty(bcx, dst, t);
    }
    if type_is_structural_or_param(ccx.tcx, t) || ty::type_is_vec(ccx.tcx, t)
       {
        let bcx = cx;
        if action == DROP_EXISTING { bcx = drop_ty(cx, dst, t); }
        bcx = memmove_ty(bcx, dst, src, t).bcx;
        ret take_ty(bcx, dst, t);
    }
    ccx.sess.bug("unexpected type in trans::copy_val_no_check: " +
                     ty_to_str(ccx.tcx, t));
}


// This works like copy_val, except that it deinitializes the source.
// Since it needs to zero out the source, src also needs to be an lval.
// FIXME: We always zero out the source. Ideally we would detect the
// case where a variable is always deinitialized by block exit and thus
// doesn't need to be dropped.
fn move_val(cx: @block_ctxt, action: copy_action, dst: ValueRef,
            src: lval_result, t: ty::t) -> @block_ctxt {
    let src_val = src.val;
    let tcx = bcx_tcx(cx);
    if ty::type_is_scalar(tcx, t) || ty::type_is_native(tcx, t) {
        if src.is_mem { src_val = Load(cx, src_val); }
        Store(cx, src_val, dst);
        ret cx;
    } else if ty::type_is_nil(tcx, t) || ty::type_is_bot(tcx, t) {
        ret cx;
    } else if ty::type_is_boxed(tcx, t) || ty::type_is_unique_box(tcx, t) {
        if src.is_mem { src_val = Load(cx, src_val); }
        if action == DROP_EXISTING { cx = drop_ty(cx, dst, t); }
        Store(cx, src_val, dst);
        if src.is_mem { ret zero_alloca(cx, src.val, t).bcx; }

        // If we're here, it must be a temporary.
        ret revoke_clean(cx, src_val, t);
    } else if ty::type_is_unique(tcx, t) ||
                  type_is_structural_or_param(tcx, t) {
        if action == DROP_EXISTING { cx = drop_ty(cx, dst, t); }
        cx = memmove_ty(cx, dst, src_val, t).bcx;
        if src.is_mem { ret zero_alloca(cx, src_val, t).bcx; }

        // If we're here, it must be a temporary.
        ret revoke_clean(cx, src_val, t);
    }
    /* FIXME: suggests a type constraint */
    bcx_ccx(cx).sess.bug("unexpected type in trans::move_val: " +
                             ty_to_str(tcx, t));
}

fn move_val_if_temp(cx: @block_ctxt, action: copy_action, dst: ValueRef,
                    src: lval_result, t: ty::t) -> @block_ctxt {

    // Lvals in memory are not temporaries. Copy them.
    if src.is_mem {
        ret copy_val(cx, action, dst, load_if_immediate(cx, src.val, t),
                     t);
    }
    ret move_val(cx, action, dst, src, t);
}

fn trans_crate_lit(cx: @crate_ctxt, lit: ast::lit) -> ValueRef {
    alt lit.node {
      ast::lit_int(i) { ret C_int(i); }
      ast::lit_uint(u) { ret C_int(u as int); }
      ast::lit_mach_int(tm, i) {
        // FIXME: the entire handling of mach types falls apart
        // if target int width is larger than host, at the moment;
        // re-do the mach-int types using 'big' when that works.

        let t = T_int();
        let s = True;
        alt tm {
          ast::ty_u8. { t = T_i8(); s = False; }
          ast::ty_u16. { t = T_i16(); s = False; }
          ast::ty_u32. { t = T_i32(); s = False; }
          ast::ty_u64. { t = T_i64(); s = False; }
          ast::ty_i8. { t = T_i8(); }
          ast::ty_i16. { t = T_i16(); }
          ast::ty_i32. { t = T_i32(); }
          ast::ty_i64. { t = T_i64(); }
        }
        ret C_integral(t, i as uint, s);
      }
      ast::lit_float(fs) { ret C_float(fs); }
      ast::lit_mach_float(tm, s) {
        let t = T_float();
        alt tm { ast::ty_f32. { t = T_f32(); } ast::ty_f64. { t = T_f64(); } }
        ret C_floating(s, t);
      }
      ast::lit_char(c) { ret C_integral(T_char(), c as uint, False); }
      ast::lit_bool(b) { ret C_bool(b); }
      ast::lit_nil. { ret C_nil(); }
      ast::lit_str(s) {
        cx.sess.span_unimpl(lit.span, "unique string in this context");
      }
    }
}

fn trans_lit(cx: @block_ctxt, lit: ast::lit) -> result {
    alt lit.node {
      ast::lit_str(s) { ret tvec::trans_str(cx, s); }
      _ { ret rslt(cx, trans_crate_lit(bcx_ccx(cx), lit)); }
    }
}


// Converts an annotation to a type
fn node_id_type(cx: @crate_ctxt, id: ast::node_id) -> ty::t {
    ret ty::node_id_to_monotype(cx.tcx, id);
}

fn node_type(cx: @crate_ctxt, sp: span, id: ast::node_id) -> TypeRef {
    let ty = node_id_type(cx, id);
    // How to make this a precondition?
    // FIXME (again, would require a predicate that implies
    // another predicate)
    check (type_has_static_size(cx, ty));
    type_of(cx, sp, ty)
}

fn trans_unary(cx: @block_ctxt, op: ast::unop, e: @ast::expr,
               id: ast::node_id) -> result {
    let e_ty = ty::expr_ty(bcx_tcx(cx), e);
    alt op {
      ast::not. {
        let sub = trans_expr(cx, e);
        ret rslt(sub.bcx, Not(sub.bcx, sub.val));
      }
      ast::neg. {
        let sub = trans_expr(cx, e);
        if ty::struct(bcx_tcx(cx), e_ty) == ty::ty_float {
            ret rslt(sub.bcx, FNeg(sub.bcx, sub.val));
        } else { ret rslt(sub.bcx, Neg(sub.bcx, sub.val)); }
      }
      ast::box(_) {
        let lv = trans_lval(cx, e);
        let box_ty = node_id_type(bcx_ccx(lv.bcx), id);
        let sub = trans_malloc_boxed(lv.bcx, e_ty);
        let body = sub.body;
        add_clean_temp(cx, sub.box, box_ty);

        // Cast the body type to the type of the value. This is needed to
        // make tags work, since tags have a different LLVM type depending
        // on whether they're boxed or not.
        let sub_ccx = bcx_ccx(sub.bcx);
        if check type_has_static_size(sub_ccx, e_ty) {
            let e_sp = e.span;
            let llety = T_ptr(type_of(sub_ccx, e_sp, e_ty));
            body = PointerCast(sub.bcx, body, llety);
        } else {
        } // FIXME: can remove the else{} once we have
          // a new snapshot


        let bcx = move_val_if_temp(sub.bcx, INIT, body, lv, e_ty);
        ret rslt(bcx, sub.box);
      }
      ast::uniq(_) {
        ret trans_uniq(cx, e, id);
      }
      ast::deref. {
        bcx_ccx(cx).sess.bug("deref expressions should have been \
                                 translated using trans_lval(), not \
                                 trans_unary()");
      }
    }
}

fn trans_compare(cx: @block_ctxt, op: ast::binop, lhs: ValueRef,
                 _lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t) -> result {
    // Determine the operation we need.
    let llop;
    alt op {
      ast::eq. | ast::ne. { llop = C_u8(abi::cmp_glue_op_eq); }
      ast::lt. | ast::ge. { llop = C_u8(abi::cmp_glue_op_lt); }
      ast::le. | ast::gt. { llop = C_u8(abi::cmp_glue_op_le); }
    }

    let rs = compare(cx, lhs, rhs, rhs_t, llop);

    // Invert the result if necessary.
    alt op {
      ast::eq. | ast::lt. | ast::le. { ret rslt(rs.bcx, rs.val); }
      ast::ne. | ast::ge. | ast::gt. {
        ret rslt(rs.bcx, Not(rs.bcx, rs.val));
      }
    }
}

// Important to get types for both lhs and rhs, because one might be _|_
// and the other not.
fn trans_eager_binop(cx: @block_ctxt, op: ast::binop, lhs: ValueRef,
                     lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t) -> result {

    // If either is bottom, it diverges. So no need to do the
    // operation.
    if ty::type_is_bot(bcx_tcx(cx), lhs_t) ||
           ty::type_is_bot(bcx_tcx(cx), rhs_t) {
        ret rslt(cx, Unreachable(cx));
    }

    let is_float = false;
    let intype = lhs_t;
    if ty::type_is_bot(bcx_tcx(cx), intype) { intype = rhs_t; }

    alt ty::struct(bcx_tcx(cx), intype) {
      ty::ty_float. { is_float = true; }
      _ { is_float = false; }
    }
    alt op {
      ast::add. {
        if ty::type_is_sequence(bcx_tcx(cx), intype) {
            ret tvec::trans_add(cx, intype, lhs, rhs);
        }
        if is_float {
            ret rslt(cx, FAdd(cx, lhs, rhs));
        } else { ret rslt(cx, Add(cx, lhs, rhs)); }
      }
      ast::sub. {
        if is_float {
            ret rslt(cx, FSub(cx, lhs, rhs));
        } else { ret rslt(cx, Sub(cx, lhs, rhs)); }
      }
      ast::mul. {
        if is_float {
            ret rslt(cx, FMul(cx, lhs, rhs));
        } else { ret rslt(cx, Mul(cx, lhs, rhs)); }
      }
      ast::div. {
        if is_float { ret rslt(cx, FDiv(cx, lhs, rhs)); }
        if ty::type_is_signed(bcx_tcx(cx), intype) {
            ret rslt(cx, SDiv(cx, lhs, rhs));
        } else { ret rslt(cx, UDiv(cx, lhs, rhs)); }
      }
      ast::rem. {
        if is_float { ret rslt(cx, FRem(cx, lhs, rhs)); }
        if ty::type_is_signed(bcx_tcx(cx), intype) {
            ret rslt(cx, SRem(cx, lhs, rhs));
        } else { ret rslt(cx, URem(cx, lhs, rhs)); }
      }
      ast::bitor. { ret rslt(cx, Or(cx, lhs, rhs)); }
      ast::bitand. { ret rslt(cx, And(cx, lhs, rhs)); }
      ast::bitxor. { ret rslt(cx, Xor(cx, lhs, rhs)); }
      ast::lsl. { ret rslt(cx, Shl(cx, lhs, rhs)); }
      ast::lsr. { ret rslt(cx, LShr(cx, lhs, rhs)); }
      ast::asr. { ret rslt(cx, AShr(cx, lhs, rhs)); }
      _ { ret trans_compare(cx, op, lhs, lhs_t, rhs, rhs_t); }
    }
}

fn autoderef(cx: @block_ctxt, v: ValueRef, t: ty::t) -> result_t {
    let v1: ValueRef = v;
    let t1: ty::t = t;
    let ccx = bcx_ccx(cx);
    let sp = cx.sp;
    while true {
        alt ty::struct(ccx.tcx, t1) {
          ty::ty_box(mt) {
            let body = GEP(cx, v1, [C_int(0), C_int(abi::box_rc_field_body)]);
            t1 = mt.ty;

            // Since we're changing levels of box indirection, we may have
            // to cast this pointer, since statically-sized tag types have
            // different types depending on whether they're behind a box
            // or not.
            if check type_has_static_size(ccx, t1) {
                let llty = type_of(ccx, sp, t1);
                v1 = PointerCast(cx, body, T_ptr(llty));
            } else { v1 = body; }
          }
          ty::ty_uniq(t) { fail "autoderef uniq unimplemented"; }
          ty::ty_res(did, inner, tps) {
            t1 = ty::substitute_type_params(ccx.tcx, tps, inner);
            v1 = GEP(cx, v1, [C_int(0), C_int(1)]);
          }
          ty::ty_tag(did, tps) {
            let variants = ty::tag_variants(ccx.tcx, did);
            if std::vec::len(variants) != 1u ||
                   std::vec::len(variants[0].args) != 1u {
                break;
            }
            t1 =
                ty::substitute_type_params(ccx.tcx, tps, variants[0].args[0]);
            if check type_has_static_size(ccx, t1) {
                v1 = PointerCast(cx, v1, T_ptr(type_of(ccx, sp, t1)));
            } else { } // FIXME: typestate hack
          }
          _ { break; }
        }
        v1 = load_if_immediate(cx, v1, t1);
    }
    ret {bcx: cx, val: v1, ty: t1};
}

fn trans_binary(cx: @block_ctxt, op: ast::binop, a: @ast::expr, b: @ast::expr)
   -> result {

    // First couple cases are lazy:
    alt op {
      ast::and. {
        // Lazy-eval and
        let lhs_res = trans_expr(cx, a);
        let rhs_cx = new_scope_block_ctxt(cx, "rhs");
        let rhs_res = trans_expr(rhs_cx, b);

        let lhs_false_cx = new_scope_block_ctxt(cx, "lhs false");
        let lhs_false_res = rslt(lhs_false_cx, C_bool(false));

        // The following line ensures that any cleanups for rhs
        // are done within the block for rhs. This is necessary
        // because and/or are lazy. So the rhs may never execute,
        // and the cleanups can't be pushed into later code.
        let rhs_bcx = trans_block_cleanups(rhs_res.bcx, rhs_cx);
        CondBr(lhs_res.bcx, lhs_res.val, rhs_cx.llbb, lhs_false_cx.llbb);
        ret join_results(cx, T_bool(),
                         [lhs_false_res, {bcx: rhs_bcx, val: rhs_res.val}]);
      }
      ast::or. {
        // Lazy-eval or
        let lhs_res = trans_expr(cx, a);
        let rhs_cx = new_scope_block_ctxt(cx, "rhs");
        let rhs_res = trans_expr(rhs_cx, b);
        let lhs_true_cx = new_scope_block_ctxt(cx, "lhs true");
        let lhs_true_res = rslt(lhs_true_cx, C_bool(true));

        // see the and case for an explanation
        let rhs_bcx = trans_block_cleanups(rhs_res.bcx, rhs_cx);
        CondBr(lhs_res.bcx, lhs_res.val, lhs_true_cx.llbb, rhs_cx.llbb);
        ret join_results(cx, T_bool(),
                         [lhs_true_res, {bcx: rhs_bcx, val: rhs_res.val}]);
      }
      _ {
        // Remaining cases are eager:
        let lhs = trans_expr(cx, a);
        let rhs = trans_expr(lhs.bcx, b);

        ret trans_eager_binop(rhs.bcx, op, lhs.val,
                              ty::expr_ty(bcx_tcx(cx), a), rhs.val,
                              ty::expr_ty(bcx_tcx(cx), b));
      }
    }
}

fn join_results(parent_cx: @block_ctxt, t: TypeRef, ins: [result]) -> result {
    let live: [result] = [];
    let vals: [ValueRef] = [];
    let bbs: [BasicBlockRef] = [];
    for r: result in ins {
        if !is_terminated(r.bcx) {
            live += [r];
            vals += [r.val];
            bbs += [r.bcx.llbb];
        }
    }
    alt std::vec::len::<result>(live) {
      0u {
        // No incoming edges are live, so we're in dead-code-land.
        // Arbitrarily pick the first dead edge, since the caller
        // is just going to propagate it outward.

        assert (std::vec::len::<result>(ins) >= 1u);
        ret ins[0];
      }
      _ {/* fall through */ }
    }
    // We have >1 incoming edges. Make a join block and br+phi them into it.

    let join_cx = new_sub_block_ctxt(parent_cx, "join");
    for r: result in live { Br(r.bcx, join_cx.llbb); }
    let phi = Phi(join_cx, t, vals, bbs);
    ret rslt(join_cx, phi);
}

fn join_branches(parent_cx: @block_ctxt, ins: [result]) -> @block_ctxt {
    let out = new_sub_block_ctxt(parent_cx, "join");
    for r: result in ins { if !is_terminated(r.bcx) { Br(r.bcx, out.llbb); } }
    ret out;
}

tag out_method { return; save_in(ValueRef); }

fn trans_if(cx: @block_ctxt, cond: @ast::expr, thn: ast::blk,
            els: option::t<@ast::expr>, output: out_method) -> result {
    let cond_res = trans_expr(cx, cond);

    if ty::type_is_bot(bcx_tcx(cx), ty::expr_ty(bcx_tcx(cx), cond)) {

        // No need to generate code for comparison,
        // since the cond diverges.
        if !is_terminated(cx) {
            ret rslt(cx, Unreachable(cx));
        } else { ret cond_res; }
    }

    let then_cx = new_scope_block_ctxt(cx, "then");
    let then_res = trans_block(then_cx, thn, output);
    let else_cx = new_scope_block_ctxt(cx, "else");
    // Synthesize a block here to act as the else block
    // containing an if expression. Needed in order for the
    // else scope to behave like a normal block scope. A tad
    // ugly.
    // Calling trans_block directly instead of trans_expr
    // because trans_expr will create another scope block
    // context for the block, but we've already got the
    // 'else' context
    let else_res =
        alt els {
          some(elexpr) {
            alt elexpr.node {
              ast::expr_if(_, _, _) {
                let elseif_blk = ast_util::block_from_expr(elexpr);
                trans_block(else_cx, elseif_blk, output)
              }
              ast::expr_block(blk) { trans_block(else_cx, blk, output) }
            }
          }
          _ { rslt(else_cx, C_nil()) }
        };
    CondBr(cond_res.bcx, cond_res.val, then_cx.llbb, else_cx.llbb);
    ret rslt(join_branches(cx, [then_res, else_res]), C_nil());
}

fn trans_for(cx: @block_ctxt, local: @ast::local, seq: @ast::expr,
             body: ast::blk) -> result {
    fn inner(bcx: @block_ctxt, local: @ast::local, curr: ValueRef, t: ty::t,
             body: ast::blk, outer_next_cx: @block_ctxt) -> @block_ctxt {
        let next_cx = new_sub_block_ctxt(bcx, "next");
        let scope_cx =
            new_loop_scope_block_ctxt(bcx, option::some(next_cx),
                                      outer_next_cx, "for loop scope");
        Br(bcx, scope_cx.llbb);
        curr = PointerCast(bcx, curr, T_ptr(type_of_or_i8(bcx, t)));
        bcx = trans_alt::bind_irrefutable_pat(scope_cx, local.node.pat, curr,
                                              bcx.fcx.lllocals, false);
        bcx = trans_block(bcx, body, return).bcx;
        if !is_terminated(bcx) {
            Br(bcx, next_cx.llbb);
            // otherwise, this code is unreachable
        }
        ret next_cx;
    }
    let next_cx = new_sub_block_ctxt(cx, "next");
    let seq_ty = ty::expr_ty(bcx_tcx(cx), seq);
    let {bcx: bcx, val: seq} = trans_expr(cx, seq);
    let seq = PointerCast(bcx, seq, T_ptr(T_ptr(T_opaque_vec())));
    let fill = tvec::get_fill(bcx, seq);
    if ty::type_is_str(bcx_tcx(bcx), seq_ty) {
        fill = Sub(bcx, fill, C_int(1));
    }
    let bcx =
        tvec::iter_vec_raw(bcx, seq, seq_ty, fill,
                           bind inner(_, local, _, _, body, next_cx));
    Br(bcx, next_cx.llbb);
    ret rslt(next_cx, C_nil());
}


// Iterator translation

// Given a block context and a list of tydescs and values to bind
// construct a closure out of them. If copying is true, it is a
// heap allocated closure that copies the upvars into environment.
// Otherwise, it is stack allocated and copies pointers to the upvars.
fn build_environment(bcx: @block_ctxt, lltydescs: [ValueRef],
                     bound_tys: [ty::t], bound_vals: [lval_result],
                     copying: bool) ->
   {ptr: ValueRef, ptrty: ty::t, bcx: @block_ctxt} {
    // Synthesize a closure type.

    // First, synthesize a tuple type containing the types of all the
    // bound expressions.
    // bindings_ty = [bound_ty1, bound_ty2, ...]
    let bindings_ty: ty::t = ty::mk_tup(bcx_tcx(bcx), bound_tys);

    // NB: keep this in sync with T_closure_ptr; we're making
    // a ty::t structure that has the same "shape" as the LLVM type
    // it constructs.

    // Make a vector that contains ty_param_count copies of tydesc_ty.
    // (We'll need room for that many tydescs in the closure.)
    let ty_param_count = std::vec::len(lltydescs);
    let tydesc_ty: ty::t = ty::mk_type(bcx_tcx(bcx));
    let captured_tys: [ty::t] = std::vec::init_elt(tydesc_ty, ty_param_count);

    // Get all the types we've got (some of which we synthesized
    // ourselves) into a vector.  The whole things ends up looking
    // like:

    // closure_tys = [tydesc_ty, [bound_ty1, bound_ty2, ...], [tydesc_ty,
    // tydesc_ty, ...]]
    let closure_tys: [ty::t] =
        [tydesc_ty, bindings_ty, ty::mk_tup(bcx_tcx(bcx), captured_tys)];

    // Finally, synthesize a type for that whole vector.
    let closure_ty: ty::t = ty::mk_tup(bcx_tcx(bcx), closure_tys);

    // Allocate a box that can hold something closure-sized.
    let r =
        if copying {
            trans_malloc_boxed(bcx, closure_ty)
        } else {
            // We need to dummy up a box on the stack
            let ty =
                ty::mk_tup(bcx_tcx(bcx),
                           [ty::mk_int(bcx_tcx(bcx)), closure_ty]);
            let r = alloc_ty(bcx, ty);
            let body = GEPi(bcx, r.val, [0, abi::box_rc_field_body]);
            {bcx: r.bcx, box: r.val, body: body}
        };
    bcx = r.bcx;
    let closure = r.body;

    // Store bindings tydesc.
    if copying {
        let bound_tydesc = GEPi(bcx, closure, [0, abi::closure_elt_tydesc]);
        let ti = none;
        let bindings_tydesc =
            get_tydesc(bcx, bindings_ty, true, tps_normal, ti).result;
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, ti);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, ti);
        bcx = bindings_tydesc.bcx;
        Store(bcx, bindings_tydesc.val, bound_tydesc);
    }

    // Copy expr values into boxed bindings.
    let i = 0u;
    // Silly check
    check type_is_tup_like(bcx, closure_ty);
    let bindings =
        GEP_tup_like(bcx, closure_ty, closure,
                     [0, abi::closure_elt_bindings]);
    bcx = bindings.bcx;
    for lv: lval_result in bound_vals {
        // Also a silly check
        check type_is_tup_like(bcx, bindings_ty);
        let bound =
            GEP_tup_like(bcx, bindings_ty, bindings.val, [0, i as int]);
        bcx = bound.bcx;
        if copying {
            bcx = move_val_if_temp(bcx, INIT, bound.val, lv, bound_tys[i]);
        } else { Store(bcx, lv.val, bound.val); }
        i += 1u;
    }

    // If necessary, copy tydescs describing type parameters into the
    // appropriate slot in the closure.
    // Silly check as well
    check type_is_tup_like(bcx, closure_ty);
    let ty_params_slot =
        GEP_tup_like(bcx, closure_ty, closure,
                     [0, abi::closure_elt_ty_params]);
    bcx = ty_params_slot.bcx;
    i = 0u;
    for td: ValueRef in lltydescs {
        let ty_param_slot = GEPi(bcx, ty_params_slot.val, [0, i as int]);
        Store(bcx, td, ty_param_slot);
        i += 1u;
    }

    ret {ptr: r.box, ptrty: closure_ty, bcx: bcx};
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for build_environment.
fn build_closure(cx: @block_ctxt, upvars: @[ast::def], copying: bool) ->
   {ptr: ValueRef, ptrty: ty::t, bcx: @block_ctxt} {
    let closure_vals: [lval_result] = [];
    let closure_tys: [ty::t] = [];
    // If we need to, package up the iterator body to call
    if !copying && !option::is_none(cx.fcx.lliterbody) {
        closure_vals += [lval_mem(cx, option::get(cx.fcx.lliterbody))];
        closure_tys += [option::get(cx.fcx.iterbodyty)];
    }
    // Package up the upvars
    for def in *upvars {
        closure_vals += [trans_local_var(cx, def)];
        let nid = ast_util::def_id_of_def(def).node;
        let ty = ty::node_id_to_monotype(bcx_tcx(cx), nid);
        if !copying { ty = ty::mk_mut_ptr(bcx_tcx(cx), ty); }
        closure_tys += [ty];
    }

    ret build_environment(cx, copy cx.fcx.lltydescs, closure_tys,
                          closure_vals, copying);
}

// Return a pointer to the stored typarams in a closure.
// This is awful. Since the size of the bindings stored in the closure might
// be dynamically sized, we can't skip past them to get to the tydescs until
// we have loaded the tydescs. Thus we use the stored size of the bindings
// in the tydesc for the closure to skip over them. Ugh.
fn find_environment_tydescs(bcx: @block_ctxt, envty: ty::t, closure: ValueRef)
   -> ValueRef {
    ret if !ty::type_has_dynamic_size(bcx_tcx(bcx), envty) {

            // If we can find the typarams statically, do it
            GEPi(bcx, closure,
                 [0, abi::box_rc_field_body, abi::closure_elt_ty_params])
        } else {
            // Ugh. We need to load the size of the bindings out of the
            // closure's tydesc and use that to skip over the bindings.
            let descsty =
                ty::get_element_type(bcx_tcx(bcx), envty,
                                     abi::closure_elt_ty_params as uint);
            let llenv = GEPi(bcx, closure, [0, abi::box_rc_field_body]);
            // Load the tydesc and find the size of the body
            let lldesc =
                Load(bcx, GEPi(bcx, llenv, [0, abi::closure_elt_tydesc]));
            let llsz =
                Load(bcx, GEPi(bcx, lldesc, [0, abi::tydesc_field_size]));

            // Get the bindings pointer and add the size to it
            let llbinds = GEPi(bcx, llenv, [0, abi::closure_elt_bindings]);
            bump_ptr(bcx, descsty, llbinds, llsz)
        }
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
fn load_environment(enclosing_cx: @block_ctxt, fcx: @fn_ctxt, envty: ty::t,
                    upvars: @[ast::def], copying: bool) {
    let bcx = new_raw_block_ctxt(fcx, fcx.llcopyargs);

    let ty = ty::mk_imm_box(bcx_tcx(bcx), envty);

    let ccx = bcx_ccx(bcx);
    let sp = bcx.sp;
    // FIXME: should have postcondition on mk_imm_box,
    // so this check won't be necessary
    check (type_has_static_size(ccx, ty));
    let llty = type_of(ccx, sp, ty);
    let llclosure = PointerCast(bcx, fcx.llenv, llty);

    // Populate the type parameters from the environment. We need to
    // do this first because the tydescs are needed to index into
    // the bindings if they are dynamically sized.
    let tydesc_count = std::vec::len(enclosing_cx.fcx.lltydescs);
    let lltydescs = find_environment_tydescs(bcx, envty, llclosure);
    let i = 0u;
    while i < tydesc_count {
        let lltydescptr = GEPi(bcx, lltydescs, [0, i as int]);
        fcx.lltydescs += [Load(bcx, lltydescptr)];
        i += 1u;
    }

    // Populate the upvars from the environment.
    let path = [0, abi::box_rc_field_body, abi::closure_elt_bindings];
    i = 0u;
    // If this is an aliasing closure/for-each body, we need to load
    // the iterbody.
    if !copying && !option::is_none(enclosing_cx.fcx.lliterbody) {
        // Silly check
        check type_is_tup_like(bcx, ty);
        let iterbodyptr = GEP_tup_like(bcx, ty, llclosure, path + [0]);
        fcx.lliterbody = some(Load(bcx, iterbodyptr.val));
        bcx = iterbodyptr.bcx;
        i += 1u;
    }

    // Load the actual upvars.
    for upvar_def in *upvars {
        // Silly check
        check type_is_tup_like(bcx, ty);
        let upvarptr = GEP_tup_like(bcx, ty, llclosure, path + [i as int]);
        bcx = upvarptr.bcx;
        let llupvarptr = upvarptr.val;
        if !copying { llupvarptr = Load(bcx, llupvarptr); }
        let def_id = ast_util::def_id_of_def(upvar_def);
        fcx.llupvars.insert(def_id.node, llupvarptr);
        i += 1u;
    }
}

fn trans_for_each(cx: @block_ctxt, local: @ast::local, seq: @ast::expr,
                  body: ast::blk) -> result {
    /*
     * The translation is a little .. complex here. Code like:
     *
     *    let ty1 p = ...;
     *
     *    let ty1 q = ...;
     *
     *    foreach (ty v in foo(a,b)) { body(p,q,v) }
     *
     *
     * Turns into a something like so (C/Rust mishmash):
     *
     *    type env = { *ty1 p, *ty2 q, ... };
     *
     *    let env e = { &p, &q, ... };
     *
     *    fn foreach123_body(env* e, ty v) { body(*(e->p),*(e->q),v) }
     *
     *    foo([foreach123_body, env*], a, b);
     *
     */

    // Step 1: Generate code to build an environment containing pointers
    // to all of the upvars
    let lcx = cx.fcx.lcx;
    let ccx = lcx.ccx;

    // FIXME: possibly support alias-mode here?
    let decl_ty = node_id_type(ccx, local.node.id);
    let upvars = get_freevars(ccx.tcx, body.node.id);

    let llenv = build_closure(cx, upvars, false);

    // Step 2: Declare foreach body function.
    let s: str =
        mangle_internal_name_by_path_and_seq(ccx, lcx.path, "foreach");

    // The 'env' arg entering the body function is a fake env member (as in
    // the env-part of the normal rust calling convention) that actually
    // points to a stack allocated env in this frame. We bundle that env
    // pointer along with the foreach-body-fn pointer into a 'normal' fn pair
    // and pass it in as a first class fn-arg to the iterator.
    let iter_body_fn = ty::mk_iter_body_fn(ccx.tcx, decl_ty);
    // FIXME: should be a postcondition on mk_iter_body_fn
    check returns_non_ty_var(ccx, iter_body_fn);
    let iter_body_llty =
        type_of_fn_from_ty(ccx, cx.sp, iter_body_fn, 0u);
    let lliterbody: ValueRef =
        decl_internal_fastcall_fn(ccx.llmod, s, iter_body_llty);
    let fcx = new_fn_ctxt_w_id(lcx, cx.sp, lliterbody, body.node.id,
                               ast::return_val);
    fcx.iterbodyty = cx.fcx.iterbodyty;

    // Generate code to load the environment out of the
    // environment pointer.
    load_environment(cx, fcx, llenv.ptrty, upvars, false);

    let bcx = new_top_block_ctxt(fcx);
    // Add bindings for the loop variable alias.
    bcx =
        trans_alt::bind_irrefutable_pat(bcx, local.node.pat,
                                        llvm::LLVMGetParam(fcx.llfn, 3u),
                                        bcx.fcx.lllocals, false);
    let lltop = bcx.llbb;
    let r = trans_block(bcx, body, return);
    finish_fn(fcx, lltop);

    if !is_terminated(r.bcx) {
        // if terminated is true, no need for the ret-fail
        build_return(r.bcx);
    }

    // Step 3: Call iter passing [lliterbody, llenv], plus other args.
    alt seq.node {
      ast::expr_call(f, args) {
        let pair =
            create_real_fn_pair(cx, iter_body_llty, lliterbody, llenv.ptr);
        let r = trans_call(cx, f, some(pair), args, seq.id);
        ret rslt(r.res.bcx, C_nil());
      }
    }
}

fn trans_while(cx: @block_ctxt, cond: @ast::expr, body: ast::blk) -> result {
    let next_cx = new_sub_block_ctxt(cx, "while next");
    let cond_cx =
        new_loop_scope_block_ctxt(cx, option::none::<@block_ctxt>, next_cx,
                                  "while cond");
    let body_cx = new_scope_block_ctxt(cond_cx, "while loop body");
    let body_res = trans_block(body_cx, body, return);
    let cond_res = trans_expr(cond_cx, cond);
    Br(body_res.bcx, cond_cx.llbb);
    let cond_bcx = trans_block_cleanups(cond_res.bcx, cond_cx);
    CondBr(cond_bcx, cond_res.val, body_cx.llbb, next_cx.llbb);
    Br(cx, cond_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn trans_do_while(cx: @block_ctxt, body: ast::blk, cond: @ast::expr) ->
   result {
    let next_cx = new_sub_block_ctxt(cx, "next");
    let body_cx =
        new_loop_scope_block_ctxt(cx, option::none::<@block_ctxt>, next_cx,
                                  "do-while loop body");
    let body_res = trans_block(body_cx, body, return);
    if is_terminated(body_res.bcx) {
        // This is kind of ridiculous, but no permutations
        // involving body_res or body_cx.val worked.
        let rs = trans_block(cx, body, return);
        if !is_terminated(next_cx) { Unreachable(next_cx); }
        if !is_terminated(body_cx) { Unreachable(body_cx); }
        ret rs;
    }
    let cond_res = trans_expr(body_res.bcx, cond);
    CondBr(cond_res.bcx, cond_res.val, body_cx.llbb, next_cx.llbb);
    Br(cx, body_cx.llbb);
    ret rslt(next_cx, body_res.val);
}

type generic_info =
    {item_type: ty::t,
     static_tis: [option::t<@tydesc_info>],
     tydescs: [ValueRef]};

type lval_result = {bcx: @block_ctxt,
                    val: ValueRef,
                    is_mem: bool};
tag callee_env { some_env(ValueRef); null_env; is_closure; }
type lval_maybe_callee = {bcx: @block_ctxt,
                          val: ValueRef,
                          is_mem: bool,
                          env: callee_env,
                          generic: option::t<generic_info>};

fn null_env_ptr(bcx: @block_ctxt) -> ValueRef {
    C_null(T_opaque_closure_ptr(*bcx_ccx(bcx)))
}

fn lval_mem(bcx: @block_ctxt, val: ValueRef) -> lval_result {
    ret {bcx: bcx, val: val, is_mem: true};
}
fn lval_val(bcx: @block_ctxt, val: ValueRef) -> lval_result {
    ret {bcx: bcx, val: val, is_mem: false};
}

fn lval_no_env(bcx: @block_ctxt, val: ValueRef, is_mem: bool)
    -> lval_maybe_callee {
    ret {bcx: bcx, val: val, is_mem: is_mem, env: is_closure, generic: none};
}

fn trans_external_path(cx: @block_ctxt, did: ast::def_id,
                       tpt: ty::ty_param_kinds_and_ty) -> ValueRef {
    let lcx = cx.fcx.lcx;
    let name = csearch::get_symbol(lcx.ccx.sess.get_cstore(), did);
    ret get_extern_const(lcx.ccx.externs, lcx.ccx.llmod, name,
                         type_of_ty_param_kinds_and_ty(lcx, cx.sp, tpt));
}

fn lval_static_fn(bcx: @block_ctxt, tpt: ty::ty_param_kinds_and_ty,
                  fn_id: ast::def_id, id: ast::node_id) -> lval_maybe_callee {
    let val = if fn_id.crate == ast::local_crate {
        // Internal reference.
        assert (bcx_ccx(bcx).item_ids.contains_key(fn_id.node));
        bcx_ccx(bcx).item_ids.get(fn_id.node)
    } else {
        // External reference.
        trans_external_path(bcx, fn_id, tpt)
    };
    let tys = ty::node_id_to_type_params(bcx_tcx(bcx), id);
    let gen = none;
    if std::vec::len::<ty::t>(tys) != 0u {
        let tydescs = [], tis = [];
        for t in tys {
            // TODO: Doesn't always escape.
            let ti = none;
            let td = get_tydesc(bcx, t, true, tps_normal, ti).result;
            tis += [ti];
            bcx = td.bcx;
            tydescs += [td.val];
        }
        gen = some({item_type: tpt.ty, static_tis: tis, tydescs: tydescs});
    }
    ret {bcx: bcx, val: val, is_mem: true, env: null_env, generic: gen};
}

fn lookup_discriminant(lcx: @local_ctxt, vid: ast::def_id) -> ValueRef {
    alt lcx.ccx.discrims.find(vid.node) {
      none. {
        // It's an external discriminant that we haven't seen yet.
        assert (vid.crate != ast::local_crate);
        let sym = csearch::get_symbol(lcx.ccx.sess.get_cstore(), vid);
        let gvar =
            str::as_buf(sym,
                        {|buf|
                            llvm::LLVMAddGlobal(lcx.ccx.llmod, T_int(), buf)
                        });
        llvm::LLVMSetLinkage(gvar,
                             lib::llvm::LLVMExternalLinkage as llvm::Linkage);
        llvm::LLVMSetGlobalConstant(gvar, True);
        lcx.ccx.discrims.insert(vid.node, gvar);
        ret gvar;
      }
      some(llval) { ret llval; }
    }
}

fn trans_local_var(cx: @block_ctxt, def: ast::def) -> lval_result {
    alt def {
      ast::def_upvar(did, _, _) {
        assert (cx.fcx.llupvars.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.llupvars.get(did.node));
      }
      ast::def_arg(did, _) {
        assert (cx.fcx.llargs.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.llargs.get(did.node));
      }
      ast::def_local(did, _) {
        assert (cx.fcx.lllocals.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.lllocals.get(did.node));
      }
      ast::def_binding(did) {
        assert (cx.fcx.lllocals.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.lllocals.get(did.node));
      }
      ast::def_obj_field(did, _) {
        assert (cx.fcx.llobjfields.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.llobjfields.get(did.node));
      }
      _ {
        bcx_ccx(cx).sess.span_unimpl
            (cx.sp, "unsupported def type in trans_local_def");
      }
    }
}

fn trans_path(cx: @block_ctxt, p: ast::path, id: ast::node_id)
    -> lval_maybe_callee {
    ret trans_var(cx, p.span, bcx_tcx(cx).def_map.get(id), id);
}

fn trans_var(cx: @block_ctxt, sp: span, def: ast::def, id: ast::node_id)
    -> lval_maybe_callee {
    let ccx = bcx_ccx(cx);
    alt def {
      ast::def_fn(did, _) | ast::def_native_fn(did) {
        let tyt = ty::lookup_item_type(ccx.tcx, did);
        ret lval_static_fn(cx, tyt, did, id);
      }
      ast::def_variant(tid, vid) {
        let v_tyt = ty::lookup_item_type(ccx.tcx, vid);
        alt ty::struct(ccx.tcx, v_tyt.ty) {
          ty::ty_fn(_, _, _, _, _) {
            // N-ary variant.
            ret lval_static_fn(cx, v_tyt, vid, id);
          }
          _ {
            // Nullary variant.
            let tag_ty = node_id_type(ccx, id);
            let alloc_result = alloc_ty(cx, tag_ty);
            let lltagblob = alloc_result.val;
            let lltagty = type_of_tag(ccx, sp, tid, tag_ty);
            let bcx = alloc_result.bcx;
            let lltagptr = PointerCast(bcx, lltagblob, T_ptr(lltagty));
            if std::vec::len(ty::tag_variants(ccx.tcx, tid)) != 1u {
                let lldiscrim_gv = lookup_discriminant(bcx.fcx.lcx, vid);
                let lldiscrim = Load(bcx, lldiscrim_gv);
                let lldiscrimptr = GEP(bcx, lltagptr, [C_int(0), C_int(0)]);
                Store(bcx, lldiscrim, lldiscrimptr);
            }
            ret lval_no_env(bcx, lltagptr, false);
          }
        }
      }
      ast::def_const(did) {
        if did.crate == ast::local_crate {
            assert (ccx.consts.contains_key(did.node));
            ret lval_no_env(cx, ccx.consts.get(did.node), true);
        } else {
            let tp = ty::node_id_to_monotype(ccx.tcx, id);
            let k: [ast::kind] = [];
            let val = trans_external_path(cx, did, {kinds: k, ty: tp});
            ret lval_no_env(cx, load_if_immediate(cx, val, tp), false);
        }
      }
      _ {
        let loc = trans_local_var(cx, def);
        ret lval_no_env(loc.bcx, loc.val, loc.is_mem);
      }
    }
}

fn trans_field(cx: @block_ctxt, sp: span, base: @ast::expr,
               field: ast::ident) -> lval_maybe_callee {
    let {bcx, val} = trans_expr(cx, base);
    ret trans_field_inner(bcx, sp, val, ty::expr_ty(bcx_tcx(cx), base),
                          field);
}

fn trans_field_inner(cx: @block_ctxt, sp: span, v: ValueRef, t0: ty::t,
                     field: ast::ident) -> lval_maybe_callee {
    let r = autoderef(cx, v, t0);
    let t = r.ty;
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_rec(fields) {
        let ix: uint = ty::field_idx(bcx_ccx(cx).sess, sp, field, fields);
        let r_bcx = r.bcx;
        // Silly check
        check type_is_tup_like(r_bcx, t);
        let v = GEP_tup_like(r_bcx, t, r.val, [0, ix as int]);
        ret lval_no_env(v.bcx, v.val, true);
      }
      ty::ty_obj(methods) {
        let ix: uint = ty::method_idx(bcx_ccx(cx).sess, sp, field, methods);
        let vtbl = GEP(r.bcx, r.val, [C_int(0), C_int(abi::obj_field_vtbl)]);
        vtbl = Load(r.bcx, vtbl);

        let vtbl_type = T_ptr(T_array(T_ptr(T_nil()), ix + 1u));
        vtbl = PointerCast(cx, vtbl, vtbl_type);

        let v = GEP(r.bcx, vtbl, [C_int(0), C_int(ix as int)]);
        let tcx = bcx_tcx(cx);
        let ccx = bcx_ccx(cx);

        let fn_ty: ty::t = ty::method_ty_to_fn_ty(tcx, methods[ix]);
        let ret_ty = ty::ty_fn_ret(tcx, fn_ty);
        let ret_ref = ast_util::ret_by_ref(ty::ty_fn_ret_style(tcx, fn_ty));
        // FIXME: constrain ty_obj?
        check non_ty_var(ccx, ret_ty);

        let ll_fn_ty =
            type_of_fn(ccx, sp, ty::ty_fn_proto(tcx, fn_ty),
                       true, ret_ref, ty::ty_fn_args(tcx, fn_ty),
                       ret_ty, 0u);
        v = Load(r.bcx, PointerCast(r.bcx, v, T_ptr(T_ptr(ll_fn_ty))));
        ret {bcx: r.bcx, val: v, is_mem: true,
             env: some_env(r.val), generic: none};
      }
      _ { bcx_ccx(cx).sess.unimpl("field variant in trans_field"); }
    }
}

fn trans_index(cx: @block_ctxt, sp: span, base: @ast::expr, idx: @ast::expr,
               id: ast::node_id) -> lval_result {
    // Is this an interior vector?

    let base_ty = ty::expr_ty(bcx_tcx(cx), base);
    let exp = trans_expr(cx, base);
    let lv = autoderef(exp.bcx, exp.val, base_ty);
    let ix = trans_expr(lv.bcx, idx);
    let v = lv.val;
    let bcx = ix.bcx;
    // Cast to an LLVM integer. Rust is less strict than LLVM in this regard.

    let ix_val;
    let ix_size = llsize_of_real(bcx_ccx(cx), val_ty(ix.val));
    let int_size = llsize_of_real(bcx_ccx(cx), T_int());
    if ix_size < int_size {
        ix_val = ZExt(bcx, ix.val, T_int());
    } else if ix_size > int_size {
        ix_val = Trunc(bcx, ix.val, T_int());
    } else { ix_val = ix.val; }
    let unit_ty = node_id_type(bcx_ccx(cx), id);
    let unit_sz = size_of(bcx, unit_ty);
    bcx = unit_sz.bcx;
    maybe_name_value(bcx_ccx(cx), unit_sz.val, "unit_sz");
    let scaled_ix = Mul(bcx, ix_val, unit_sz.val);
    maybe_name_value(bcx_ccx(cx), scaled_ix, "scaled_ix");
    let lim = tvec::get_fill(bcx, v);
    let body = tvec::get_dataptr(bcx, v, type_of_or_i8(bcx, unit_ty));
    let bounds_check = ICmp(bcx, lib::llvm::LLVMIntULT, scaled_ix, lim);
    let fail_cx = new_sub_block_ctxt(bcx, "fail");
    let next_cx = new_sub_block_ctxt(bcx, "next");
    let ncx = bcx_ccx(next_cx);
    CondBr(bcx, bounds_check, next_cx.llbb, fail_cx.llbb);
    // fail: bad bounds check.

    trans_fail(fail_cx, some::<span>(sp), "bounds check");
    let elt =
        if check type_has_static_size(ncx, unit_ty) {
            let elt_1 = GEP(next_cx, body, [ix_val]);
            let llunitty = type_of(ncx, sp, unit_ty);
            PointerCast(next_cx, elt_1, T_ptr(llunitty))
        } else {
            body = PointerCast(next_cx, body, T_ptr(T_i8()));
            GEP(next_cx, body, [scaled_ix])
        };

    ret lval_mem(next_cx, elt);
}

fn trans_callee(cx: @block_ctxt, e: @ast::expr) -> lval_maybe_callee {
    alt e.node {
      ast::expr_path(p) { ret trans_path(cx, p, e.id); }
      ast::expr_field(base, ident) {
        ret trans_field(cx, e.span, base, ident);
      }
      ast::expr_self_method(ident) {
        alt cx.fcx.llself {
          some(pair) {
            ret trans_field_inner(cx, e.span, pair.v, pair.t, ident);
          }
        }
      }
      _ {
        let lv = trans_lval(cx, e);
        ret lval_no_env(lv.bcx, lv.val, lv.is_mem);
      }
    }
}

// The additional bool returned indicates whether it's mem (that is
// represented as an alloca or heap, hence needs a 'load' to be used as an
// immediate).
fn trans_lval(cx: @block_ctxt, e: @ast::expr) -> lval_result {
    alt e.node {
      ast::expr_path(p) {
        let v = trans_path(cx, p, e.id);
        ret lval_maybe_callee_to_lval(v, ty::expr_ty(bcx_tcx(cx), e));
      }
      ast::expr_field(base, ident) {
        let f = trans_field(cx, e.span, base, ident);
        ret lval_maybe_callee_to_lval(f, ty::expr_ty(bcx_tcx(cx), e));
      }
      ast::expr_index(base, idx) {
        ret trans_index(cx, e.span, base, idx, e.id);
      }
      ast::expr_unary(ast::deref., base) {
        let ccx = bcx_ccx(cx);
        let sub = trans_expr(cx, base);
        let t = ty::expr_ty(ccx.tcx, base);
        let val =
            alt ty::struct(ccx.tcx, t) {
              ty::ty_box(_) {
                InBoundsGEP(sub.bcx, sub.val,
                            [C_int(0), C_int(abi::box_rc_field_body)])
              }
              ty::ty_res(_, _, _) {
                InBoundsGEP(sub.bcx, sub.val, [C_int(0), C_int(1)])
              }
              ty::ty_tag(_, _) {
                let ety = ty::expr_ty(ccx.tcx, e);
                let sp = e.span;
                let ellty =
                    if check type_has_static_size(ccx, ety) {
                        T_ptr(type_of(ccx, sp, ety))
                    } else { T_typaram_ptr(ccx.tn) };
                PointerCast(sub.bcx, sub.val, ellty)
              }
              ty::ty_ptr(_) | ty::ty_uniq(_) { sub.val }
            };
        ret lval_mem(sub.bcx, val);
      }
      ast::expr_call(f, args) {
        let {res: {bcx, val}, by_ref} =
            trans_call(cx, f, none, args, e.id);
        if by_ref { ret lval_mem(bcx, val); }
        else { ret lval_val(bcx, val); }
      }
      _ {
        let res = trans_expr(cx, e);
        ret lval_val(res.bcx, res.val);
      }
    }
}

fn maybe_add_env(bcx: @block_ctxt, c: lval_maybe_callee)
    -> (bool, ValueRef) {
    if c.env == is_closure {
        (c.is_mem, c.val)
    } else {
        let env = alt c.env {
          null_env. { null_env_ptr(bcx) }
          some_env(e) { e }
        };
        let llfnty = llvm::LLVMGetElementType(val_ty(c.val));
        (false, create_real_fn_pair(bcx, llfnty, c.val, env))
    }
}

fn lval_maybe_callee_to_lval(c: lval_maybe_callee, ty: ty::t) -> lval_result {
    alt c.generic {
      some(gi) {
        let n_args = std::vec::len(ty::ty_fn_args(bcx_tcx(c.bcx), ty));
        let args = std::vec::init_elt(none::<@ast::expr>, n_args);
        let {bcx, val} = trans_bind_1(c.bcx, ty, c, args, ty);
        ret lval_val(bcx, val);
      }
      none. {
        let (is_mem, val) = maybe_add_env(c.bcx, c);
        ret {bcx: c.bcx, val: val, is_mem: is_mem};
      }
    }
}

fn int_cast(bcx: @block_ctxt, lldsttype: TypeRef, llsrctype: TypeRef,
            llsrc: ValueRef, signed: bool) -> ValueRef {
    let srcsz = llvm::LLVMGetIntTypeWidth(llsrctype);
    let dstsz = llvm::LLVMGetIntTypeWidth(lldsttype);
    ret if dstsz == srcsz {
            BitCast(bcx, llsrc, lldsttype)
        } else if srcsz > dstsz {
            TruncOrBitCast(bcx, llsrc, lldsttype)
        } else if signed {
            SExtOrBitCast(bcx, llsrc, lldsttype)
        } else { ZExtOrBitCast(bcx, llsrc, lldsttype) };
}

fn float_cast(bcx: @block_ctxt, lldsttype: TypeRef, llsrctype: TypeRef,
              llsrc: ValueRef) -> ValueRef {
    let srcsz = lib::llvm::float_width(llsrctype);
    let dstsz = lib::llvm::float_width(lldsttype);
    ret if dstsz > srcsz {
            FPExt(bcx, llsrc, lldsttype)
        } else if srcsz > dstsz {
            FPTrunc(bcx, llsrc, lldsttype)
        } else { llsrc };
}

fn trans_cast(cx: @block_ctxt, e: @ast::expr, id: ast::node_id) -> result {
    let ccx = bcx_ccx(cx);
    let e_res = trans_expr(cx, e);
    let ll_t_in = val_ty(e_res.val);
    let t_in = ty::expr_ty(ccx.tcx, e);
    let t_out = node_id_type(ccx, id);
    // Check should be avoidable because it's a cast.
    // FIXME: Constrain types so as to avoid this check.
    check (type_has_static_size(ccx, t_out));
    let ll_t_out = type_of(ccx, e.span, t_out);

    tag kind { native_; integral; float; other; }
    fn t_kind(tcx: ty::ctxt, t: ty::t) -> kind {
        ret if ty::type_is_fp(tcx, t) {
                float
            } else if ty::type_is_native(tcx, t) {
                native_
            } else if ty::type_is_integral(tcx, t) {
                integral
            } else { other };
    }
    let k_in = t_kind(ccx.tcx, t_in);
    let k_out = t_kind(ccx.tcx, t_out);
    let s_in = k_in == integral && ty::type_is_signed(ccx.tcx, t_in);

    let newval =
        alt {in: k_in, out: k_out} {
          {in: integral., out: integral.} {
            int_cast(e_res.bcx, ll_t_out, ll_t_in, e_res.val, s_in)
          }
          {in: float., out: float.} {
            float_cast(e_res.bcx, ll_t_out, ll_t_in, e_res.val)
          }
          {in: integral., out: float.} {
            if s_in {
                SIToFP(e_res.bcx, e_res.val, ll_t_out)
            } else { UIToFP(e_res.bcx, e_res.val, ll_t_out) }
          }
          {in: float., out: integral.} {
            if ty::type_is_signed(ccx.tcx, t_out) {
                FPToSI(e_res.bcx, e_res.val, ll_t_out)
            } else { FPToUI(e_res.bcx, e_res.val, ll_t_out) }
          }
          {in: integral., out: native_.} {
            IntToPtr(e_res.bcx, e_res.val, ll_t_out)
          }
          {in: native_., out: integral.} {
            PtrToInt(e_res.bcx, e_res.val, ll_t_out)
          }
          {in: native_., out: native_.} {
            PointerCast(e_res.bcx, e_res.val, ll_t_out)
          }
          _ { ccx.sess.bug("Translating unsupported cast.") }
        };
    ret rslt(e_res.bcx, newval);
}

// pth is cx.path
fn trans_bind_thunk(cx: @local_ctxt, sp: span, incoming_fty: ty::t,
                    outgoing_fty: ty::t, args: [option::t<@ast::expr>],
                    env_ty: ty::t, ty_param_count: uint,
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
    let llthunk: ValueRef =
        decl_internal_fastcall_fn(ccx.llmod, s, llthunk_ty);

    // Create a new function context and block context for the thunk, and hold
    // onto a pointer to the first block in the function for later use.
    let fcx = new_fn_ctxt(cx, sp, llthunk);
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;
    // Since we might need to construct derived tydescs that depend on
    // our bound tydescs, we need to load tydescs out of the environment
    // before derived tydescs are constructed. To do this, we load them
    // in the copy_args block.
    let copy_args_bcx = new_raw_block_ctxt(fcx, fcx.llcopyargs);

    // The 'llenv' that will arrive in the thunk we're creating is an
    // environment that will contain the values of its arguments and a pointer
    // to the original function.  So, let's create one of those:

    // The llenv pointer needs to be the correct size.  That size is
    // 'closure_ty', which was determined by trans_bind.
    let closure_ty = ty::mk_imm_box(ccx.tcx, env_ty);
    // FIXME: would be nice to have a postcondition on mk_imm_box
    // (Issue #586)
    check (type_has_static_size(ccx, closure_ty));
    let llclosure_ptr_ty = type_of(ccx, sp, closure_ty);
    let llclosure = PointerCast(copy_args_bcx, fcx.llenv, llclosure_ptr_ty);

    // "target", in this context, means the function that's having some of its
    // arguments bound and that will be called inside the thunk we're
    // creating.  (In our running example, target is the function f.)  Pick
    // out the pointer to the target function from the environment. The
    // target function lives in the first binding spot.
    let (lltargetfn, lltargetenv, starting_idx) = alt target_fn {
      some(fptr) { (fptr, llvm::LLVMGetUndef(T_opaque_closure_ptr(*ccx)), 0) }
      none. {
        // Silly check
        check type_is_tup_like(bcx, closure_ty);
        let {bcx: cx, val: pair} =
            GEP_tup_like(bcx, closure_ty, llclosure,
                         [0, abi::box_rc_field_body,
                          abi::closure_elt_bindings, 0]);
        let lltargetenv =
            Load(cx, GEP(cx, pair, [C_int(0), C_int(abi::fn_field_box)]));
        let lltargetfn = Load
            (cx, GEP(cx, pair, [C_int(0), C_int(abi::fn_field_code)]));
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
    let llargs: [ValueRef] = [llretptr, fcx.lltaskptr, lltargetenv];

    // Copy in the type parameters.
    let i: uint = 0u;
    while i < ty_param_count {
        // Silly check
        check type_is_tup_like(copy_args_bcx, closure_ty);
        let lltyparam_ptr =
            GEP_tup_like(copy_args_bcx, closure_ty, llclosure,
                         [0, abi::box_rc_field_body,
                          abi::closure_elt_ty_params, i as int]);
        copy_args_bcx = lltyparam_ptr.bcx;
        let td = Load(copy_args_bcx, lltyparam_ptr.val);
        llargs += [td];
        fcx.lltydescs += [td];
        i += 1u;
    }

    let a: uint = 3u; // retptr, task ptr, env come first
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
            check type_is_tup_like(bcx, closure_ty);
            let bound_arg =
                GEP_tup_like(bcx, closure_ty, llclosure,
                             [0, abi::box_rc_field_body,
                              abi::closure_elt_bindings, b]);
            bcx = bound_arg.bcx;
            let val = bound_arg.val;
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

    check returns_non_ty_var(ccx, outgoing_fty);
    let lltargetty =
        type_of_fn_from_ty(ccx, sp, outgoing_fty, ty_param_count);
    lltargetfn = PointerCast(bcx, lltargetfn, T_ptr(lltargetty));
    FastCall(bcx, lltargetfn, llargs);
    build_return(bcx);
    finish_fn(fcx, lltop);
    ret {val: llthunk, ty: llthunk_ty};
}

fn trans_bind(cx: @block_ctxt, f: @ast::expr, args: [option::t<@ast::expr>],
              id: ast::node_id) -> result {
    let f_res = trans_callee(cx, f);
    ret trans_bind_1(cx, ty::expr_ty(bcx_tcx(cx), f), f_res, args,
                     ty::node_id_to_type(bcx_tcx(cx), id));
}

fn trans_bind_1(cx: @block_ctxt, outgoing_fty: ty::t,
                f_res: lval_maybe_callee,
                args: [option::t<@ast::expr>], pair_ty: ty::t) -> result {
    let bound: [@ast::expr] = [];
    for argopt: option::t<@ast::expr> in args {
        alt argopt { none. { } some(e) { bound += [e]; } }
    }

    // Figure out which tydescs we need to pass, if any.
    let outgoing_fty_real; // the type with typarams still in it
    let lltydescs: [ValueRef];
    alt f_res.generic {
      none. { outgoing_fty_real = outgoing_fty; lltydescs = []; }
      some(ginfo) {
        lazily_emit_all_generic_info_tydesc_glues(cx, ginfo);
        outgoing_fty_real = ginfo.item_type;
        lltydescs = ginfo.tydescs;
      }
    }

    let ty_param_count = std::vec::len(lltydescs);
    if std::vec::len(bound) == 0u && ty_param_count == 0u {
        // Trivial 'binding': just return the closure
        let lv = lval_maybe_callee_to_lval(f_res, pair_ty);
        ret rslt(lv.bcx, lv.val);
    }
    let bcx = f_res.bcx;
    let (is_mem, closure) = alt f_res.env {
      null_env. { (true, none) }
      _ { let (mem, cl) = maybe_add_env(cx, f_res); (mem, some(cl)) }
    };

    // FIXME: should follow from a precondition on trans_bind_1
    let ccx = bcx_ccx(cx);
    check (type_has_static_size(ccx, outgoing_fty));

    // Arrange for the bound function to live in the first binding spot
    // if the function is not statically known.
    let (bound_tys, bound_vals, target_res) = alt closure {
      some(cl) {
        // Cast the function we are binding to be the type that the
        // closure will expect it to have. The type the closure knows
        // about has the type parameters substituted with the real types.
        let sp = cx.sp;
        let llclosurety = T_ptr(type_of(ccx, sp, outgoing_fty));
        let src_loc = PointerCast(bcx, cl, llclosurety);
        let bound_f = {bcx: bcx, val: src_loc, is_mem: is_mem};
        ([outgoing_fty], [bound_f], none)
      }
      none. { ([], [], some(f_res.val)) }
    };

    // Translate the bound expressions.
    for e: @ast::expr in bound {
        let lv = trans_lval(bcx, e);
        bcx = lv.bcx;
        bound_vals += [lv];
        bound_tys += [ty::expr_ty(bcx_tcx(cx), e)];
    }

    // Actually construct the closure
    let closure =
        build_environment(bcx, lltydescs, bound_tys, bound_vals, true);
    bcx = closure.bcx;

    // Make thunk
    let llthunk =
        trans_bind_thunk(cx.fcx.lcx, cx.sp, pair_ty, outgoing_fty_real, args,
                         closure.ptrty, ty_param_count, target_res);

    // Construct the function pair
    let pair_v =
        create_real_fn_pair(bcx, llthunk.ty, llthunk.val, closure.ptr);
    add_clean_temp(cx, pair_v, pair_ty);
    ret rslt(bcx, pair_v);
}

fn trans_arg_expr(cx: @block_ctxt, arg: ty::arg, lldestty0: TypeRef,
                  &to_zero: [{v: ValueRef, t: ty::t}],
                  &to_revoke: [{v: ValueRef, t: ty::t}], e: @ast::expr) ->
   result {
    let ccx = bcx_ccx(cx);
    let e_ty = ty::expr_ty(ccx.tcx, e);
    let is_bot = ty::type_is_bot(ccx.tcx, e_ty);
    let lv = trans_lval(cx, e);
    let bcx = lv.bcx;
    let val = lv.val;
    if is_bot {
        // For values of type _|_, we generate an
        // "undef" value, as such a value should never
        // be inspected. It's important for the value
        // to have type lldestty0 (the callee's expected type).
        val = llvm::LLVMGetUndef(lldestty0);
    } else if arg.mode == ast::by_ref {
        let copied = false;
        if !lv.is_mem && type_is_immediate(ccx, e_ty) {
            val = do_spill_noroot(bcx, val);
            copied = true;
        }
        if ccx.copy_map.contains_key(e.id) && lv.is_mem {
            if !copied {
                let alloc = alloc_ty(bcx, e_ty);
                bcx =
                    copy_val(alloc.bcx, INIT, alloc.val,
                             load_if_immediate(alloc.bcx, val, e_ty), e_ty);
                val = alloc.val;
            } else { bcx = take_ty(bcx, val, e_ty); }
            add_clean(bcx, val, e_ty);
        }
    } else if type_is_immediate(ccx, e_ty) && !lv.is_mem {
        let r = do_spill(bcx, val, e_ty);
        val = r.val;
        bcx = r.bcx;
    }

    if !is_bot && ty::type_contains_params(ccx.tcx, arg.ty) {
        let lldestty = lldestty0;
        val = PointerCast(bcx, val, lldestty);
    }

    // Collect arg for later if it happens to be one we've moving out.
    if arg.mode == ast::by_move {
        if lv.is_mem {
            // Use actual ty, not declared ty -- anything else doesn't make
            // sense if declared ty is a ty param
            to_zero += [{v: lv.val, t: e_ty}];
        } else { to_revoke += [{v: lv.val, t: e_ty}]; }
    }
    ret rslt(bcx, val);
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn trans_args(cx: @block_ctxt, outer_cx: @block_ctxt, llenv: ValueRef,
              gen: option::t<generic_info>,
              lliterbody: option::t<ValueRef>, es: [@ast::expr], fn_ty: ty::t)
   -> {bcx: @block_ctxt,
       outer_cx: @block_ctxt,
       args: [ValueRef],
       retslot: ValueRef,
       to_zero: [{v: ValueRef, t: ty::t}],
       to_revoke: [{v: ValueRef, t: ty::t}]} {

    let args: [ty::arg] = ty::ty_fn_args(bcx_tcx(cx), fn_ty);
    let llargs: [ValueRef] = [];
    let lltydescs: [ValueRef] = [];
    let to_zero = [];
    let to_revoke = [];

    let ccx = bcx_ccx(cx);
    let tcx = ccx.tcx;
    let bcx: @block_ctxt = cx;
    let ret_style = ty::ty_fn_ret_style(tcx, fn_ty);
    let by_ref = ast_util::ret_by_ref(ret_style);
    // Arg 0: Output pointer.

    // FIXME: test case looks like
    // f(1, fail, @42);
    if is_terminated(bcx) {
        // This means an earlier arg was divergent.
        // So this arg can't be evaluated.
        ret {bcx: bcx,
             outer_cx: outer_cx,
             args: [],
             retslot: C_nil(),
             to_zero: to_zero,
             to_revoke: to_revoke};
    }
    let retty = ty::ty_fn_ret(tcx, fn_ty), full_retty = retty;
    alt gen {
      some(g) {
        lazily_emit_all_generic_info_tydesc_glues(cx, g);
        lltydescs = g.tydescs;
        args = ty::ty_fn_args(tcx, g.item_type);
        retty = ty::ty_fn_ret(tcx, g.item_type);
      }
      _ { }
    }
    let llretslot_res = if ty::type_is_nil(tcx, retty) {
        rslt(cx, llvm::LLVMGetUndef(T_ptr(T_nil())))
    } else if by_ref {
        rslt(cx, alloca(cx, T_ptr(type_of_or_i8(bcx, full_retty))))
    } else { alloc_ty(bcx, full_retty) };
    bcx = llretslot_res.bcx;
    let llretslot = llretslot_res.val;
    if ty::type_contains_params(tcx, retty) {
        // It's possible that the callee has some generic-ness somewhere in
        // its return value -- say a method signature within an obj or a fn
        // type deep in a structure -- which the caller has a concrete view
        // of. If so, cast the caller's view of the restlot to the callee's
        // view, for the sake of making a type-compatible call.
        check non_ty_var(ccx, retty);
        let llretty = T_ptr(type_of_inner(ccx, bcx.sp, retty));
        if by_ref { llretty = T_ptr(llretty); }
        llargs += [PointerCast(cx, llretslot, llretty)];
    } else { llargs += [llretslot]; }

    // Arg 1: task pointer.
    llargs += [bcx.fcx.lltaskptr];

    // Arg 2: Env (closure-bindings / self-obj)
    llargs += [llenv];

    // Args >3: ty_params ...
    llargs += lltydescs;

    // ... then possibly an lliterbody argument.
    alt lliterbody {
      none. { }
      some(lli) {
        let lli =
            if ty::type_contains_params(tcx, retty) {
                let body_ty = ty::mk_iter_body_fn(tcx, retty);
                check non_ty_var(ccx, body_ty);
                let body_llty = type_of_inner(ccx, cx.sp, body_ty);
                PointerCast(bcx, lli, T_ptr(body_llty))
            } else { lli };
        llargs += [Load(cx, lli)];
      }
    }

    // ... then explicit args.

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    let arg_tys = type_of_explicit_args(ccx, cx.sp, args);
    let i = 0u;
    for e: @ast::expr in es {
        if is_terminated(bcx) {
            // This means an earlier arg was divergent.
            // So this arg can't be evaluated.
            break;
        }
        let is_referenced = alt ret_style {
          ast::return_ref(_, arg_n) { i + 1u == arg_n }
          _ { false }
        };
        let r = trans_arg_expr(is_referenced ? outer_cx : bcx,
                               args[i], arg_tys[i], to_zero, to_revoke, e);
        if is_referenced { outer_cx = r.bcx; } else { bcx = r.bcx; }
        llargs += [r.val];
        i += 1u;
    }
    ret {bcx: bcx,
         outer_cx: outer_cx,
         args: llargs,
         retslot: llretslot,
         to_zero: to_zero,
         to_revoke: to_revoke};
}

fn trans_call(in_cx: @block_ctxt, f: @ast::expr,
              lliterbody: option::t<ValueRef>, args: [@ast::expr],
              id: ast::node_id) -> {res: result, by_ref: bool} {
    // NB: 'f' isn't necessarily a function; it might be an entire self-call
    // expression because of the hack that allows us to process self-calls
    // with trans_call.
    let fn_expr_ty = ty::expr_ty(bcx_tcx(in_cx), f);
    let by_ref = ast_util::ret_by_ref(ty::ty_fn_ret_style(bcx_tcx(in_cx),
                                                          fn_expr_ty));
    let cx = new_scope_block_ctxt(in_cx, "call");
    let f_res = trans_callee(cx, f);
    let bcx = f_res.bcx;

    let faddr = f_res.val;
    let llenv;
    alt f_res.env {
      null_env. {
        llenv = llvm::LLVMGetUndef(T_opaque_closure_ptr(*bcx_ccx(cx)));
      }
      some_env(e) { llenv = e; }
      is_closure. {
        // It's a closure. Have to fetch the elements
        if f_res.is_mem { faddr = load_if_immediate(bcx, faddr, fn_expr_ty); }
        let pair = faddr;
        faddr = GEP(bcx, pair, [C_int(0), C_int(abi::fn_field_code)]);
        faddr = Load(bcx, faddr);
        let llclosure = GEP(bcx, pair, [C_int(0), C_int(abi::fn_field_box)]);
        llenv = Load(bcx, llclosure);
      }
    }

    let ret_ty = ty::node_id_to_type(bcx_tcx(cx), id);
    let args_res =
        trans_args(bcx, in_cx, llenv, f_res.generic, lliterbody, args,
                   fn_expr_ty);
    Br(args_res.outer_cx, cx.llbb);
    bcx = args_res.bcx;
    let llargs = args_res.args;
    let llretslot = args_res.retslot;

    /*
    log_err "calling: " + val_str(bcx_ccx(cx).tn, faddr);

    for arg: ValueRef in llargs {
        log_err "arg: " + val_str(bcx_ccx(cx).tn, arg);
    }
    */

    /* If the block is terminated,
       then one or more of the args has
       type _|_. Since that means it diverges, the code
       for the call itself is unreachable. */
    let retval = C_nil();
    if !is_terminated(bcx) {
        bcx = invoke_fastcall(bcx, faddr, llargs,
                              args_res.to_zero, args_res.to_revoke).bcx;
        alt lliterbody {
          none. {
            if !ty::type_is_nil(bcx_tcx(cx), ret_ty) {
                if by_ref {
                    retval = Load(bcx, llretslot);
                } else {
                    retval = load_if_immediate(bcx, llretslot, ret_ty);
                    // Retval doesn't correspond to anything really tangible
                    // in the frame, but it's a ref all the same, so we put a
                    // note here to drop it when we're done in this scope.
                    add_clean_temp(in_cx, retval, ret_ty);
                }
            }
          }
          some(_) {
            // If there was an lliterbody, it means we were calling an
            // iter, and we are *not* the party using its 'output' value,
            // we should ignore llretslot.
          }
        }

        // Forget about anything we moved out.
        bcx = zero_and_revoke(bcx, args_res.to_zero, args_res.to_revoke);

        if !by_ref { bcx = trans_block_cleanups(bcx, cx); }
        let next_cx = new_sub_block_ctxt(in_cx, "next");
        Br(bcx, next_cx.llbb);
        bcx = next_cx;
    }
    ret {res: rslt(bcx, retval), by_ref: by_ref};
}

fn zero_and_revoke(bcx: @block_ctxt,
                   to_zero: [{v: ValueRef, t: ty::t}],
                   to_revoke: [{v: ValueRef, t: ty::t}]) -> @block_ctxt {
    let bcx = bcx;
    for {v, t} in to_zero {
        bcx = zero_alloca(bcx, v, t).bcx;
    }
    for {v, t} in to_revoke {
        bcx = revoke_clean(bcx, v, t);
    }
    ret bcx;
}

fn invoke(bcx: @block_ctxt, llfn: ValueRef,
          llargs: [ValueRef]) -> result {
    ret invoke_(bcx, llfn, llargs, [], [], Invoke);
}

fn invoke_fastcall(bcx: @block_ctxt, llfn: ValueRef,
                   llargs: [ValueRef],
                   to_zero: [{v: ValueRef, t: ty::t}],
                   to_revoke: [{v: ValueRef, t: ty::t}]) -> result {
    ret invoke_(bcx, llfn, llargs,
                to_zero, to_revoke,
                FastInvoke);
}

fn invoke_(bcx: @block_ctxt, llfn: ValueRef, llargs: [ValueRef],
           to_zero: [{v: ValueRef, t: ty::t}],
           to_revoke: [{v: ValueRef, t: ty::t}],
           invoker: fn(@block_ctxt, ValueRef, [ValueRef],
                       BasicBlockRef, BasicBlockRef) -> ValueRef) -> result {
    // FIXME: May be worth turning this into a plain call when there are no
    // cleanups to run
    let normal_bcx = new_sub_block_ctxt(bcx, "normal return");
    let retval = invoker(bcx, llfn, llargs,
                         normal_bcx.llbb,
                         get_landing_pad(bcx, to_zero, to_revoke));
    ret rslt(normal_bcx, retval);
}

fn get_landing_pad(bcx: @block_ctxt,
                   to_zero: [{v: ValueRef, t: ty::t}],
                   to_revoke: [{v: ValueRef, t: ty::t}]
                  ) -> BasicBlockRef {
    let have_zero_or_revoke = vec::is_not_empty(to_zero)
        || vec::is_not_empty(to_revoke);
    let scope_bcx = find_scope_for_lpad(bcx, have_zero_or_revoke);
    if scope_bcx.lpad_dirty || have_zero_or_revoke {
        let unwind_bcx = new_sub_block_ctxt(bcx, "unwind");
        let lpadbb = trans_landing_pad(unwind_bcx, to_zero, to_revoke);
        scope_bcx.lpad = some(lpadbb);
        scope_bcx.lpad_dirty = have_zero_or_revoke;
    }
    assert option::is_some(scope_bcx.lpad);
    ret option::get(scope_bcx.lpad);

    fn find_scope_for_lpad(bcx: @block_ctxt,
                           have_zero_or_revoke: bool) -> @block_ctxt {
        let scope_bcx = bcx;
        while true {
            scope_bcx = find_scope_cx(scope_bcx);
            if vec::is_not_empty(scope_bcx.cleanups)
                || have_zero_or_revoke {
                ret scope_bcx;
            } else {
                scope_bcx = alt scope_bcx.parent {
                  parent_some(b) { b }
                  parent_none. {
                    ret scope_bcx;
                  }
                };
            }
        }
        fail;
    }
}

fn trans_landing_pad(bcx: @block_ctxt,
                     to_zero: [{v: ValueRef, t: ty::t}],
                     to_revoke: [{v: ValueRef, t: ty::t}]
                    ) -> BasicBlockRef {
    // The landing pad return type (the type being propagated). Not sure what
    // this represents but it's determined by the personality function and
    // this is what the EH proposal example uses.
    let llretty = T_struct([T_ptr(T_i8()), T_i32()]);
    // The exception handling personality function. This is the C++
    // personality function __gxx_personality_v0, wrapped in our naming
    // convention.
    let personality = bcx_ccx(bcx).upcalls.rust_personality;
    // The only landing pad clause will be 'cleanup'
    let clauses = 1u;
    let llpad = LandingPad(bcx, llretty, personality, clauses);
    // The landing pad result is used both for modifying the landing pad
    // in the C API and as the exception value
    let llretval = llpad;
    // The landing pad block is a cleanup
    SetCleanup(bcx, llpad);

    // FIXME: This seems like a very naive and redundant way to generate the
    // landing pads, as we're re-generating all in-scope cleanups for each
    // function call. Probably good optimization opportunities here.
    let bcx = zero_and_revoke(bcx, to_zero, to_revoke);
    let scope_cx = bcx;
    while true {
        scope_cx = find_scope_cx(scope_cx);
        bcx = trans_block_cleanups(bcx, scope_cx);
        scope_cx = alt scope_cx.parent {
          parent_some(b) { b }
          parent_none. { break; }
        };
    }

    // Continue unwinding
    Resume(bcx, llretval);
    ret bcx.llbb;
}

fn trans_tup(cx: @block_ctxt, elts: [@ast::expr], id: ast::node_id) ->
   result {
    let bcx = cx;
    let t = node_id_type(bcx.fcx.lcx.ccx, id);
    let tup_res = alloc_ty(bcx, t);
    let tup_val = tup_res.val;
    bcx = tup_res.bcx;

    // Like trans_rec, we'll collect the fields of the tuple then build it, so
    // that if we fail in between we don't have to deal with cleaning up a
    // partial tuple
    let tupfields: [(ValueRef, lval_result, ty::t)] = [];
    let i: int = 0;
    for e in elts {
        let e_ty = ty::expr_ty(cx.fcx.lcx.ccx.tcx, e);
        let src = trans_lval(bcx, e);
        bcx = src.bcx;
        // FIXME: constraint on argument?
        check type_is_tup_like(bcx, t);
        let dst_res = GEP_tup_like(bcx, t, tup_val, [0, i]);
        tupfields += [(dst_res.val, src, e_ty)];
        bcx = dst_res.bcx;
        i += 1;
    }

    // Fill in the tuple fields
    for (dst, src, t) in tupfields {
        bcx = move_val_if_temp(bcx, INIT, dst, src, t);
    }

    // Only register the cleanups after the tuple is built
    add_clean_temp(cx, tup_val, t);
    ret rslt(bcx, tup_val);
}

fn trans_rec(cx: @block_ctxt, fields: [ast::field],
             base: option::t<@ast::expr>, id: ast::node_id) -> result {
    let bcx = cx;
    let t = node_id_type(bcx_ccx(bcx), id);
    let rec_res = alloc_ty(bcx, t);
    let rec_val = rec_res.val;
    bcx = rec_res.bcx;
    let i: int = 0;
    let base_val = C_nil();
    alt base {
      none. { }
      some(bexp) {
        let base_res = trans_expr(bcx, bexp);
        bcx = base_res.bcx;
        base_val = base_res.val;
      }
    }
    let ty_fields: [ty::field] = [];
    alt ty::struct(bcx_tcx(cx), t) { ty::ty_rec(flds) { ty_fields = flds; } }

    tag fieldsrc {
        provided(lval_result);
        inherited(ValueRef);
    }
    type fieldval = {
        dst: ValueRef,
        src: fieldsrc,
        ty: ty::t
    };
    let fieldvals: [fieldval] = [];

    // We build the record in two stages so that we don't have to clean up a
    // partial record if we fail: first collect all the values, then construct
    // the record.
    for tf: ty::field in ty_fields {
        let e_ty = tf.mt.ty;
        // FIXME: constraint on argument?
        check type_is_tup_like(bcx, t);
        let dst_res = GEP_tup_like(bcx, t, rec_val, [0, i]);
        bcx = dst_res.bcx;
        let expr_provided = false;
        for f: ast::field in fields {
            if str::eq(f.node.ident, tf.ident) {
                expr_provided = true;
                let lv = trans_lval(bcx, f.node.expr);
                bcx = lv.bcx;
                fieldvals += [{
                    dst: dst_res.val,
                    src: provided(lv),
                    ty: e_ty
                }];
                break;
            }
        }
        if !expr_provided {
            // FIXME: constraint on argument?
            check type_is_tup_like(bcx, t);
            let src_res = GEP_tup_like(bcx, t, base_val, [0, i]);
            bcx = src_res.bcx;
            fieldvals += [{
                dst: dst_res.val,
                src: inherited(src_res.val),
                ty: e_ty
            }];
        }
        i += 1;
    }

    // Now build the record
    for fieldval in fieldvals {
        alt fieldval.src {
          provided(lv) {
            bcx = move_val_if_temp(bcx, INIT, fieldval.dst,
                                   lv, fieldval.ty);
          }
          inherited(val) {
            let val = load_if_immediate(bcx, val, fieldval.ty);
            bcx = copy_val(bcx, INIT, fieldval.dst, val, fieldval.ty);
          }
        }
    }

    add_clean_temp(cx, rec_val, t);
    ret rslt(bcx, rec_val);
}

fn trans_expr(cx: @block_ctxt, e: @ast::expr) -> result {
    trans_expr_out(cx, e, return)
}

fn trans_expr_out(cx: @block_ctxt, e: @ast::expr, output: out_method) ->
   result {
    // Fixme Fill in cx.sp
    alt e.node {
      ast::expr_lit(lit) { ret trans_lit(cx, *lit); }
      ast::expr_binary(op, x, y) { ret trans_binary(cx, op, x, y); }
      ast::expr_if(cond, thn, els) {
        ret with_out_method(bind trans_if(cx, cond, thn, els, _), cx, e.id,
                            output);
      }
      ast::expr_if_check(cond, thn, els) {
        ret with_out_method(bind trans_if(cx, cond, thn, els, _), cx, e.id,
                            output);
      }
      ast::expr_ternary(_, _, _) {
        ret trans_expr_out(cx, ast_util::ternary_to_if(e), output);
      }
      ast::expr_for(decl, seq, body) { ret trans_for(cx, decl, seq, body); }
      ast::expr_for_each(decl, seq, body) {
        ret trans_for_each(cx, decl, seq, body);
      }
      ast::expr_while(cond, body) { ret trans_while(cx, cond, body); }
      ast::expr_do_while(body, cond) { ret trans_do_while(cx, body, cond); }
      ast::expr_alt(expr, arms) {
        ret with_out_method(bind trans_alt::trans_alt(cx, expr, arms, _), cx,
                            e.id, output);
      }
      ast::expr_fn(f) {
        let ccx = bcx_ccx(cx);
        let fty = node_id_type(ccx, e.id);
        check returns_non_ty_var(ccx, fty);
        let llfnty: TypeRef =
            type_of_fn_from_ty(ccx, e.span, fty, 0u);
        let sub_cx = extend_path(cx.fcx.lcx, ccx.names.next("anon"));
        let s = mangle_internal_name_by_path(ccx, sub_cx.path);
        let llfn = decl_internal_fastcall_fn(ccx.llmod, s, llfnty);

        let fn_res =
            trans_closure(some(cx), some(llfnty), sub_cx, e.span, f, llfn,
                          none, [], e.id);
        let fn_pair =
            alt fn_res {
              some(fn_pair) { fn_pair }
              none. {
                {fn_pair: create_real_fn_pair(cx, llfnty, llfn,
                                              null_env_ptr(cx)),
                 bcx: cx}
              }
            };
        ret rslt(fn_pair.bcx, fn_pair.fn_pair);
      }
      ast::expr_block(blk) {
        let sub_cx = new_scope_block_ctxt(cx, "block-expr body");
        let next_cx = new_sub_block_ctxt(cx, "next");
        let sub =
            with_out_method(bind trans_block(sub_cx, blk, _), cx, e.id,
                            output);
        Br(cx, sub_cx.llbb);
        if !is_terminated(sub.bcx) { Br(sub.bcx, next_cx.llbb); }
        ret rslt(next_cx, sub.val);
      }
      ast::expr_copy(a) {
        let e_ty = ty::expr_ty(bcx_tcx(cx), a);
        let lv = trans_lval(cx, a);
        let bcx = lv.bcx;
        if !lv.is_mem { ret {bcx: lv.bcx, val: lv.val}; }
        let r = if type_is_immediate(bcx_ccx(cx), e_ty) {
            rslt(bcx, Load(bcx, lv.val))
        } else {
            let {bcx, val: dest} = alloc_ty(bcx, e_ty);
            bcx = copy_val(bcx, INIT, dest, lv.val, e_ty);
            rslt(bcx, dest)
        };
        add_clean_temp(bcx, r.val, e_ty);
        ret r;
      }
      ast::expr_move(dst, src) {
        let lhs_res = trans_lval(cx, dst);
        assert (lhs_res.is_mem);
        // FIXME Fill in lhs_res.bcx.sp

        let rhs_res = trans_lval(lhs_res.bcx, src);
        let t = ty::expr_ty(bcx_tcx(cx), src);
        // FIXME: calculate copy init-ness in typestate.

        let bcx =
            move_val(rhs_res.bcx, DROP_EXISTING, lhs_res.val, rhs_res,
                     t);
        ret rslt(bcx, C_nil());
      }
      ast::expr_assign(dst, src) {
        let lhs_res = trans_lval(cx, dst);
        assert (lhs_res.is_mem);
        // FIXME Fill in lhs_res.bcx.sp
        let rhs = trans_lval(lhs_res.bcx, src);
        let t = ty::expr_ty(bcx_tcx(cx), src);
        // FIXME: calculate copy init-ness in typestate.
        let bcx =
            move_val_if_temp(rhs.bcx, DROP_EXISTING, lhs_res.val, rhs,
                             t);
        ret rslt(bcx, C_nil());
      }
      ast::expr_swap(dst, src) {
        let lhs_res = trans_lval(cx, dst);
        assert (lhs_res.is_mem);
        // FIXME Fill in lhs_res.bcx.sp

        let rhs_res = trans_lval(lhs_res.bcx, src);
        let t = ty::expr_ty(bcx_tcx(cx), src);
        let {bcx: bcx, val: tmp_alloc} = alloc_ty(rhs_res.bcx, t);
        // Swap through a temporary.

        bcx = move_val(bcx, INIT, tmp_alloc, lhs_res, t);
        bcx = move_val(bcx, INIT, lhs_res.val, rhs_res, t);
        bcx =
            move_val(bcx, INIT, rhs_res.val, lval_mem(bcx, tmp_alloc), t);
        ret rslt(bcx, C_nil());
      }
      ast::expr_assign_op(op, dst, src) {
        let tcx = bcx_tcx(cx);
        let t = ty::expr_ty(tcx, src);
        let lhs_res = trans_lval(cx, dst);
        assert (lhs_res.is_mem);

        // Special case for `+= [x]`
        alt ty::struct(tcx, t) {
          ty::ty_vec(_) {
            alt src.node {
              ast::expr_vec(args, _) {
                let bcx =
                    tvec::trans_append_literal(lhs_res.bcx,
                                               lhs_res.val, t, args);
                ret rslt(bcx, C_nil());
              }
              _ { }
            }
          }
          _ { }
        }

        // FIXME Fill in lhs_res.bcx.sp
        let rhs_res = trans_expr(lhs_res.bcx, src);
        if ty::type_is_sequence(tcx, t) {
            alt op {
              ast::add. {
                ret tvec::trans_append(rhs_res.bcx, t, lhs_res.val,
                                       rhs_res.val);
              }
              _ { }
            }
        }
        let lhs_val = load_if_immediate(rhs_res.bcx, lhs_res.val, t);
        let v =
            trans_eager_binop(rhs_res.bcx, op, lhs_val, t, rhs_res.val, t);
        // FIXME: calculate copy init-ness in typestate.
        // This is always a temporary, so can always be safely moved
        let bcx =
            move_val(v.bcx, DROP_EXISTING, lhs_res.val,
                     lval_val(v.bcx, v.val), t);
        ret rslt(bcx, C_nil());
      }
      ast::expr_bind(f, args) { ret trans_bind(cx, f, args, e.id); }
      ast::expr_cast(val, _) { ret trans_cast(cx, val, e.id); }
      ast::expr_vec(args, _) { ret tvec::trans_vec(cx, args, e.id); }
      ast::expr_rec(args, base) { ret trans_rec(cx, args, base, e.id); }
      ast::expr_tup(args) { ret trans_tup(cx, args, e.id); }
      ast::expr_mac(_) { ret bcx_ccx(cx).sess.bug("unexpanded macro"); }
      ast::expr_fail(expr) { ret trans_fail_expr(cx, some(e.span), expr); }
      ast::expr_log(lvl, a) { ret trans_log(lvl, cx, a); }
      ast::expr_assert(a) { ret trans_check_expr(cx, a, "Assertion"); }
      ast::expr_check(ast::checked., a) {
        ret trans_check_expr(cx, a, "Predicate");
      }
      ast::expr_check(ast::unchecked., a) {
        /* Claims are turned on and off by a global variable
           that the RTS sets. This case generates code to
           check the value of that variable, doing nothing
           if it's set to false and acting like a check
           otherwise. */
        let c =
            get_extern_const(bcx_ccx(cx).externs, bcx_ccx(cx).llmod,
                             "check_claims", T_bool());
        let cond = Load(cx, c);

        let then_cx = new_scope_block_ctxt(cx, "claim_then");
        let check_res = trans_check_expr(then_cx, a, "Claim");
        let else_cx = new_scope_block_ctxt(cx, "else");
        let els = rslt(else_cx, C_nil());

        CondBr(cx, cond, then_cx.llbb, else_cx.llbb);
        ret rslt(join_branches(cx, [check_res, els]), C_nil());
      }
      ast::expr_break. { ret trans_break(e.span, cx); }
      ast::expr_cont. { ret trans_cont(e.span, cx); }
      ast::expr_ret(ex) { ret trans_ret(cx, ex); }
      ast::expr_put(ex) { ret trans_put(cx, ex); }
      ast::expr_be(ex) {
        // Ideally, the expr_be tag would have a precondition
        // that is_call_expr(ex) -- but we don't support that
        // yet
        // FIXME
        check (ast_util::is_call_expr(ex));
        ret trans_be(cx, ex);
      }
      ast::expr_anon_obj(anon_obj) {
        ret trans_anon_obj(cx, e.span, anon_obj, e.id);
      }
      ast::expr_call(_, _) | ast::expr_field(_, _) | ast::expr_index(_, _) |
      ast::expr_path(_) | ast::expr_unary(ast::deref., _) {
        let t = ty::expr_ty(bcx_tcx(cx), e);
        let sub = trans_lval(cx, e);
        let v = sub.val;
        if sub.is_mem { v = load_if_immediate(sub.bcx, v, t); }
        ret rslt(sub.bcx, v);
      }
      ast::expr_unary(op, x) {
        ret trans_unary(cx, op, x, e.id);
      }
    }
}

fn with_out_method(work: fn(out_method) -> result, cx: @block_ctxt,
                   id: ast::node_id, outer_output: out_method) -> result {
    let ccx = bcx_ccx(cx);
    if outer_output != return {
        ret work(outer_output);
    } else {
        let tp = node_id_type(ccx, id);
        if ty::type_is_nil(ccx.tcx, tp) { ret work(return); }
        let res_alloca = alloc_ty(cx, tp);
        cx = zero_alloca(res_alloca.bcx, res_alloca.val, tp).bcx;
        let done = work(save_in(res_alloca.val));
        let loaded = load_if_immediate(done.bcx, res_alloca.val, tp);
        add_clean_temp(cx, loaded, tp);
        ret rslt(done.bcx, loaded);
    }
}


// We pass structural values around the compiler "by pointer" and
// non-structural values (scalars, boxes, pointers) "by value". We call the
// latter group "immediates" and, in some circumstances when we know we have a
// pointer (or need one), perform load/store operations based on the
// immediate-ness of the type.
fn type_is_immediate(ccx: @crate_ctxt, t: ty::t) -> bool {
    ret ty::type_is_scalar(ccx.tcx, t) || ty::type_is_boxed(ccx.tcx, t) ||
        ty::type_is_unique_box(ccx.tcx, t) || ty::type_is_native(ccx.tcx, t);
}

fn do_spill(cx: @block_ctxt, v: ValueRef, t: ty::t) -> result {
    // We have a value but we have to spill it, and root it, to pass by alias.
    let bcx = cx;

    if ty::type_is_bot(bcx_tcx(bcx), t) {
        ret rslt(bcx, C_null(T_ptr(T_i8())));
    }

    let r = alloc_ty(bcx, t);
    bcx = r.bcx;
    let llptr = r.val;

    Store(bcx, v, llptr);

    ret rslt(bcx, llptr);
}

// Since this function does *not* root, it is the caller's responsibility to
// ensure that the referent is pointed to by a root.
fn do_spill_noroot(cx: @block_ctxt, v: ValueRef) -> ValueRef {
    let llptr = alloca(cx, val_ty(v));
    Store(cx, v, llptr);
    ret llptr;
}

fn spill_if_immediate(cx: @block_ctxt, v: ValueRef, t: ty::t) -> result {
    if type_is_immediate(bcx_ccx(cx), t) { ret do_spill(cx, v, t); }
    ret rslt(cx, v);
}

fn load_if_immediate(cx: @block_ctxt, v: ValueRef, t: ty::t) -> ValueRef {
    if type_is_immediate(bcx_ccx(cx), t) { ret Load(cx, v); }
    ret v;
}

fn trans_log(lvl: int, cx: @block_ctxt, e: @ast::expr) -> result {
    let lcx = cx.fcx.lcx;
    let modname = str::connect(lcx.module_path, "::");
    let global;
    if lcx.ccx.module_data.contains_key(modname) {
        global = lcx.ccx.module_data.get(modname);
    } else {
        let s =
            link::mangle_internal_name_by_path_and_seq(lcx.ccx,
                                                       lcx.module_path,
                                                       "loglevel");
        global =
            str::as_buf(s,
                        {|buf|
                            llvm::LLVMAddGlobal(lcx.ccx.llmod, T_int(), buf)
                        });
        llvm::LLVMSetGlobalConstant(global, False);
        llvm::LLVMSetInitializer(global, C_null(T_int()));
        llvm::LLVMSetLinkage(global,
                             lib::llvm::LLVMInternalLinkage as llvm::Linkage);
        lcx.ccx.module_data.insert(modname, global);
    }
    let log_cx = new_scope_block_ctxt(cx, "log");
    let after_cx = new_sub_block_ctxt(cx, "after");
    let load = Load(cx, global);
    let test = ICmp(cx, lib::llvm::LLVMIntSGE, load, C_int(lvl));
    CondBr(cx, test, log_cx.llbb, after_cx.llbb);
    let sub = trans_expr(log_cx, e);
    let e_ty = ty::expr_ty(bcx_tcx(cx), e);
    let log_bcx = sub.bcx;

    let ti = none::<@tydesc_info>;
    let r = get_tydesc(log_bcx, e_ty, false, tps_normal, ti).result;
    log_bcx = r.bcx;
    let lltydesc = r.val;

    // Call the polymorphic log function.
    r = spill_if_immediate(log_bcx, sub.val, e_ty);
    log_bcx = r.bcx;
    let llvalptr = r.val;

    let llval_i8 = PointerCast(log_bcx, llvalptr, T_ptr(T_i8()));

    Call(log_bcx, bcx_ccx(log_bcx).upcalls.log_type,
         [log_bcx.fcx.lltaskptr, lltydesc, llval_i8, C_int(lvl)]);

    log_bcx = trans_block_cleanups(log_bcx, log_cx);
    Br(log_bcx, after_cx.llbb);
    ret rslt(after_cx, C_nil());
}

fn trans_check_expr(cx: @block_ctxt, e: @ast::expr, s: str) -> result {
    let cond_res = trans_expr(cx, e);
    let expr_str = s + " " + expr_to_str(e) + " failed";
    let fail_cx = new_sub_block_ctxt(cx, "fail");
    trans_fail(fail_cx, some::<span>(e.span), expr_str);
    let next_cx = new_sub_block_ctxt(cx, "next");
    CondBr(cond_res.bcx, cond_res.val, next_cx.llbb, fail_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn trans_fail_expr(cx: @block_ctxt, sp_opt: option::t<span>,
                   fail_expr: option::t<@ast::expr>) -> result {
    let bcx = cx;
    alt fail_expr {
      some(expr) {
        let tcx = bcx_tcx(bcx);
        let expr_res = trans_expr(bcx, expr);
        let e_ty = ty::expr_ty(tcx, expr);
        bcx = expr_res.bcx;

        if ty::type_is_str(tcx, e_ty) {
            let data =
                tvec::get_dataptr(bcx, expr_res.val,
                                  type_of_or_i8(bcx,
                                                ty::mk_mach(tcx,
                                                            ast::ty_u8)));
            ret trans_fail_value(bcx, sp_opt, data);
        } else {
            bcx_ccx(cx).sess.span_bug(expr.span,
                                      "fail called with unsupported type " +
                                          ty_to_str(tcx, e_ty));
        }
      }
      _ { ret trans_fail(bcx, sp_opt, "explicit failure"); }
    }
}

fn trans_fail(cx: @block_ctxt, sp_opt: option::t<span>, fail_str: str) ->
   result {
    let V_fail_str = C_cstr(bcx_ccx(cx), fail_str);
    ret trans_fail_value(cx, sp_opt, V_fail_str);
}

fn trans_fail_value(cx: @block_ctxt, sp_opt: option::t<span>,
                    V_fail_str: ValueRef) -> result {
    let V_filename;
    let V_line;
    alt sp_opt {
      some(sp) {
        let loc = bcx_ccx(cx).sess.lookup_pos(sp.lo);
        V_filename = C_cstr(bcx_ccx(cx), loc.filename);
        V_line = loc.line as int;
      }
      none. { V_filename = C_cstr(bcx_ccx(cx), "<runtime>"); V_line = 0; }
    }
    let V_str = PointerCast(cx, V_fail_str, T_ptr(T_i8()));
    V_filename = PointerCast(cx, V_filename, T_ptr(T_i8()));
    let args = [cx.fcx.lltaskptr, V_str, V_filename, C_int(V_line)];
    let cx = invoke(cx, bcx_ccx(cx).upcalls._fail, args).bcx;
    Unreachable(cx);
    ret rslt(cx, C_nil());
}

fn trans_put(in_cx: @block_ctxt, e: option::t<@ast::expr>) -> result {
    let cx = new_scope_block_ctxt(in_cx, "put");
    Br(in_cx, cx.llbb);
    let llcallee = C_nil();
    let llenv = C_nil();
    alt cx.fcx.lliterbody {
      some(lli) {
        let slot = alloca(cx, val_ty(lli));
        Store(cx, lli, slot);
        llcallee = GEP(cx, slot, [C_int(0), C_int(abi::fn_field_code)]);
        llcallee = Load(cx, llcallee);
        llenv = GEP(cx, slot, [C_int(0), C_int(abi::fn_field_box)]);
        llenv = Load(cx, llenv);
      }
    }
    let bcx = cx;
    let dummy_retslot = alloca(bcx, T_nil());
    let llargs: [ValueRef] = [dummy_retslot, cx.fcx.lltaskptr, llenv];
    alt e {
      none. {
        llargs += [C_null(T_ptr(T_nil()))];
      }
      some(x) {
        let e_ty = ty::expr_ty(bcx_tcx(cx), x);
        let arg = {mode: ast::by_ref, ty: e_ty};
        let arg_tys = type_of_explicit_args(bcx_ccx(cx), x.span, [arg]);
        let z = [];
        let k = [];
        let r = trans_arg_expr(bcx, arg, arg_tys[0], z, k, x);
        bcx = r.bcx;
        llargs += [r.val];
      }
    }
    bcx = invoke_fastcall(bcx, llcallee, llargs, [], []).bcx;
    bcx = trans_block_cleanups(bcx, cx);
    let next_cx = new_sub_block_ctxt(in_cx, "next");
    Br(bcx, next_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn trans_uniq(cx: @block_ctxt, contents: @ast::expr,
              node_id: ast::node_id) -> result {
    let bcx = cx;

    let lv = trans_lval(bcx, contents);
    bcx = lv.bcx;

    let contents_ty = ty::expr_ty(bcx_tcx(bcx), contents);
    let r = size_of(bcx, contents_ty);
    bcx = r.bcx;
    let llsz = r.val;

    let llptrty = T_ptr(type_of_or_i8(bcx, contents_ty));

    r = trans_shared_malloc(bcx, llptrty, llsz);
    bcx = r.bcx;
    let llptr = r.val;

    bcx = move_val_if_temp(bcx, INIT, llptr, lv, contents_ty);

    let uniq_ty = node_id_type(bcx_ccx(cx), node_id);
    add_clean_temp(r.bcx, llptr, uniq_ty);
    ret rslt(r.bcx, llptr);
}

fn trans_break_cont(sp: span, cx: @block_ctxt, to_end: bool) -> result {
    let bcx = cx;
    // Locate closest loop block, outputting cleanup as we go.

    let cleanup_cx = cx;
    while true {
        bcx = trans_block_cleanups(bcx, cleanup_cx);
        alt copy cleanup_cx.kind {
          LOOP_SCOPE_BLOCK(_cont, _break) {
            if to_end {
                Br(bcx, _break.llbb);
            } else {
                alt _cont {
                  option::some(_cont) { Br(bcx, _cont.llbb); }
                  _ { Br(bcx, cleanup_cx.llbb); }
                }
            }
            ret rslt(new_sub_block_ctxt(bcx, "break_cont.unreachable"),
                     C_nil());
          }
          _ {
            alt cleanup_cx.parent {
              parent_some(cx) { cleanup_cx = cx; }
              parent_none. {
                bcx_ccx(cx).sess.span_fatal(sp,
                                            if to_end {
                                                "Break"
                                            } else { "Cont" } +
                                                " outside a loop");
              }
            }
          }
        }
    }
    // If we get here without returning, it's a bug

    bcx_ccx(cx).sess.bug("in trans::trans_break_cont()");
}

fn trans_break(sp: span, cx: @block_ctxt) -> result {
    ret trans_break_cont(sp, cx, true);
}

fn trans_cont(sp: span, cx: @block_ctxt) -> result {
    ret trans_break_cont(sp, cx, false);
}

fn trans_ret(cx: @block_ctxt, e: option::t<@ast::expr>) -> result {
    let bcx = cx;
    alt e {
      some(x) {
        let t = ty::expr_ty(bcx_tcx(cx), x);
        let lv = trans_lval(cx, x);
        bcx = lv.bcx;
        if ty::type_is_nil(bcx_tcx(cx), t) {
            // Don't write nil
        } else if ast_util::ret_by_ref(cx.fcx.ret_style) {
            assert lv.is_mem;
            Store(bcx, lv.val, cx.fcx.llretptr);
        } else {
            let is_local = alt x.node {
              ast::expr_path(p) {
                alt bcx_tcx(bcx).def_map.get(x.id) {
                  ast::def_local(_, _) { true }
                  _ { false }
                }
              }
              _ { false }
            };
            if is_local {
                bcx = move_val(bcx, INIT, cx.fcx.llretptr, lv, t);
            } else {
                bcx = move_val_if_temp(bcx, INIT, cx.fcx.llretptr, lv, t);
            }
        }
      }
      _ {}
    }
    // run all cleanups and back out.

    let more_cleanups: bool = true;
    let cleanup_cx = cx;
    while more_cleanups {
        bcx = trans_block_cleanups(bcx, cleanup_cx);
        alt cleanup_cx.parent {
          parent_some(b) { cleanup_cx = b; }
          parent_none. { more_cleanups = false; }
        }
    }
    build_return(bcx);
    ret rslt(new_sub_block_ctxt(bcx, "ret.unreachable"), C_nil());
}

fn build_return(bcx: @block_ctxt) { Br(bcx, bcx_fcx(bcx).llreturn); }

// fn trans_be(cx: &@block_ctxt, e: &@ast::expr) -> result {
fn trans_be(cx: @block_ctxt, e: @ast::expr) : ast_util::is_call_expr(e) ->
   result {

    // FIXME: Turn this into a real tail call once
    // calling convention issues are settled

    ret trans_ret(cx, some(e));
}

fn init_local(bcx: @block_ctxt, local: @ast::local) -> @block_ctxt {
    let ty = node_id_type(bcx_ccx(bcx), local.node.id);
    let llptr = bcx.fcx.lllocals.get(local.node.id);
    // Make a note to drop this slot on the way out.
    add_clean(bcx, llptr, ty);

    if must_zero(bcx_ccx(bcx), local) {
        bcx = zero_alloca(bcx, llptr, ty).bcx;
    }

    alt local.node.init {
      some(init) {
        alt init.op {
          ast::init_assign. {
            // Use the type of the RHS because if it's _|_, the LHS
            // type might be something else, but we don't want to copy
            // the value.
            ty = node_id_type(bcx_ccx(bcx), init.expr.id);
            let sub = trans_lval(bcx, init.expr);
            bcx = move_val_if_temp(sub.bcx, INIT, llptr, sub, ty);
          }
          ast::init_move. {
            let sub = trans_lval(bcx, init.expr);
            bcx = move_val(sub.bcx, INIT, llptr, sub, ty);
          }
        }
      }
      _ { }
    }
    bcx =
        trans_alt::bind_irrefutable_pat(bcx, local.node.pat, llptr,
                                        bcx.fcx.lllocals, false);
    ret bcx;

    fn must_zero(ccx: @crate_ctxt, local: @ast::local) -> bool {
        alt local.node.init {
          some(init) { might_not_init(ccx, init.expr) }
          none. { true }
        }
    }

    fn might_not_init(ccx: @crate_ctxt, expr: @ast::expr) -> bool {
        type env = {mutable mightnt: bool,
                    ccx: @crate_ctxt};
        let e = {mutable mightnt: false,
                 ccx: ccx};
        fn visit_expr(ex: @ast::expr, e: env, v: vt<env>) {
            let might_not_init = alt ex.node {
              ast::expr_alt(_, _) { true }
              ast::expr_ret(_) { true }
              ast::expr_break. { true }
              ast::expr_cont. { true }
              ast::expr_call(_, _) { true }
              _ {
                let ex_ty = ty::expr_ty(e.ccx.tcx, ex);
                ty::type_is_bot(e.ccx.tcx, ex_ty)
              }
            };
            if might_not_init {
                e.mightnt = true;
            } else { visit::visit_expr(ex, e, v); }
        }
        let visitor =
            visit::mk_vt(@{visit_expr: visit_expr
                           with *visit::default_visitor()});
        visitor.visit_expr(expr, e, visitor);
        ret e.mightnt;
    }
}

fn init_ref_local(bcx: @block_ctxt, local: @ast::local) -> @block_ctxt {
    let init_expr = option::get(local.node.init).expr;
    let val = trans_lval(bcx, init_expr);
    assert val.is_mem;
    ret trans_alt::bind_irrefutable_pat(val.bcx, local.node.pat,
                                        val.val, bcx.fcx.lllocals, false);
}

fn zero_alloca(cx: @block_ctxt, llptr: ValueRef, t: ty::t) -> result {
    let bcx = cx;
    let ccx = bcx_ccx(cx);
    if check type_has_static_size(ccx, t) {
        let sp = cx.sp;
        let llty = type_of(ccx, sp, t);
        Store(bcx, C_null(llty), llptr);
    } else {
        let llsz = size_of(bcx, t);
        // FIXME passing in the align here is correct, but causes issue #843
        // let llalign = align_of(llsz.bcx, t);
        bcx = call_bzero(llsz.bcx, llptr, llsz.val, C_int(0)).bcx;
    }
    rslt(bcx, llptr)
}

fn trans_stmt(cx: @block_ctxt, s: ast::stmt) -> result {
    // FIXME Fill in cx.sp

    let bcx = cx;
    alt s.node {
      ast::stmt_expr(e, _) { bcx = trans_expr(cx, e).bcx; }
      ast::stmt_decl(d, _) {
        alt d.node {
          ast::decl_local(locals) {
            for (style, local) in locals {
                if style == ast::let_copy {
                    bcx = init_local(bcx, local);
                } else {
                    bcx = init_ref_local(bcx, local);
                }
            }
          }
          ast::decl_item(i) { trans_item(cx.fcx.lcx, *i); }
        }
      }
      _ { bcx_ccx(cx).sess.unimpl("stmt variant"); }
    }
    rslt(bcx, C_nil())
}

// You probably don't want to use this one. See the
// next three functions instead.
fn new_block_ctxt(cx: @fn_ctxt, parent: block_parent, kind: block_kind,
                  name: str) -> @block_ctxt {
    let s = "";
    if cx.lcx.ccx.sess.get_opts().save_temps ||
           cx.lcx.ccx.sess.get_opts().debuginfo {
        s = cx.lcx.ccx.names.next(name);
    }
    let llbb: BasicBlockRef =
        str::as_buf(s, {|buf| llvm::LLVMAppendBasicBlock(cx.llfn, buf) });
    ret @{llbb: llbb,
          mutable terminated: false,
          parent: parent,
          kind: kind,
          mutable cleanups: [],
          mutable lpad_dirty: true,
          mutable lpad: option::none,
          sp: cx.sp,
          fcx: cx};
}


// Use this when you're at the top block of a function or the like.
fn new_top_block_ctxt(fcx: @fn_ctxt) -> @block_ctxt {
    ret new_block_ctxt(fcx, parent_none, SCOPE_BLOCK, "function top level");
}


// Use this when you're at a curly-brace or similar lexical scope.
fn new_scope_block_ctxt(bcx: @block_ctxt, n: str) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, parent_some(bcx), SCOPE_BLOCK, n);
}

fn new_loop_scope_block_ctxt(bcx: @block_ctxt, _cont: option::t<@block_ctxt>,
                             _break: @block_ctxt, n: str) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, parent_some(bcx),
                       LOOP_SCOPE_BLOCK(_cont, _break), n);
}


// Use this when you're making a general CFG BB within a scope.
fn new_sub_block_ctxt(bcx: @block_ctxt, n: str) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, parent_some(bcx), NON_SCOPE_BLOCK, n);
}

fn new_raw_block_ctxt(fcx: @fn_ctxt, llbb: BasicBlockRef) -> @block_ctxt {
    ret @{llbb: llbb,
          mutable terminated: false,
          parent: parent_none,
          kind: NON_SCOPE_BLOCK,
          mutable cleanups: [],
          mutable lpad_dirty: true,
          mutable lpad: option::none,
          sp: fcx.sp,
          fcx: fcx};
}


// trans_block_cleanups: Go through all the cleanups attached to this
// block_ctxt and execute them.
//
// When translating a block that introdces new variables during its scope, we
// need to make sure those variables go out of scope when the block ends.  We
// do that by running a 'cleanup' function for each variable.
// trans_block_cleanups runs all the cleanup functions for the block.
fn trans_block_cleanups(cx: @block_ctxt, cleanup_cx: @block_ctxt) ->
   @block_ctxt {
    let bcx = cx;
    if cleanup_cx.kind == NON_SCOPE_BLOCK {
        assert (std::vec::len::<cleanup>(cleanup_cx.cleanups) == 0u);
    }
    let i = std::vec::len::<cleanup>(cleanup_cx.cleanups);
    while i > 0u {
        i -= 1u;
        let c = cleanup_cx.cleanups[i];
        alt c {
          clean(cfn) { bcx = cfn(bcx); }
          clean_temp(_, cfn) { bcx = cfn(bcx); }
        }
    }
    ret bcx;
}

fn trans_fn_cleanups(fcx: @fn_ctxt, cx: @block_ctxt) {
    alt fcx.llobstacktoken {
      some(lltoken_) {
        let lltoken = lltoken_; // satisfy alias checker
        Call(cx, fcx_ccx(fcx).upcalls.dynastack_free,
             [fcx.lltaskptr, lltoken]);
      }
      none. {/* nothing to do */ }
    }
}

iter block_locals(b: ast::blk) -> @ast::local {
    for s: @ast::stmt in b.node.stmts {
        alt s.node {
          ast::stmt_decl(d, _) {
            alt d.node {
              ast::decl_local(locals) {
                for (style, local) in locals {
                    if style == ast::let_copy { put local; }
                }
              }
              _ {/* fall through */ }
            }
          }
          _ {/* fall through */ }
        }
    }
}

fn llstaticallocas_block_ctxt(fcx: @fn_ctxt) -> @block_ctxt {
    ret @{llbb: fcx.llstaticallocas,
          mutable terminated: false,
          parent: parent_none,
          kind: SCOPE_BLOCK,
          mutable cleanups: [],
          mutable lpad_dirty: true,
          mutable lpad: option::none,
          sp: fcx.sp,
          fcx: fcx};
}

fn llderivedtydescs_block_ctxt(fcx: @fn_ctxt) -> @block_ctxt {
    ret @{llbb: fcx.llderivedtydescs,
          mutable terminated: false,
          parent: parent_none,
          kind: SCOPE_BLOCK,
          mutable cleanups: [],
          mutable lpad_dirty: true,
          mutable lpad: option::none,
          sp: fcx.sp,
          fcx: fcx};
}


fn alloc_ty(cx: @block_ctxt, t: ty::t) -> result {
    let bcx = cx;
    let ccx = bcx_ccx(cx);
    let val =
        if check type_has_static_size(ccx, t) {
            let sp = cx.sp;
            alloca(bcx, type_of(ccx, sp, t))
        } else {
            // NB: we have to run this particular 'size_of' in a
            // block_ctxt built on the llderivedtydescs block for the fn,
            // so that the size dominates the array_alloca that
            // comes next.

            let n = size_of(llderivedtydescs_block_ctxt(bcx.fcx), t);
            bcx.fcx.llderivedtydescs = n.bcx.llbb;
            dynastack_alloca(bcx, T_i8(), n.val, t)
        };

    // NB: since we've pushed all size calculations in this
    // function up to the alloca block, we actually return the
    // block passed into us unmodified; it doesn't really
    // have to be passed-and-returned here, but it fits
    // past caller conventions and may well make sense again,
    // so we leave it as-is.

    if bcx_tcx(cx).sess.get_opts().do_gc {
        bcx = gc::add_gc_root(bcx, val, t);
    }

    ret rslt(cx, val);
}

fn alloc_local(cx: @block_ctxt, local: @ast::local) -> result {
    let t = node_id_type(bcx_ccx(cx), local.node.id);
    let r = alloc_ty(cx, t);
    alt local.node.pat.node {
      ast::pat_bind(ident) {
        if bcx_ccx(cx).sess.get_opts().debuginfo {
            let _: () =
                str::as_buf(ident,
                            {|buf| llvm::LLVMSetValueName(r.val, buf) });
        }
      }
      _ { }
    }
    ret r;
}

fn trans_block(cx: @block_ctxt, b: ast::blk, output: out_method) -> result {
    let bcx = cx;
    for each local: @ast::local in block_locals(b) {
        // FIXME Update bcx.sp
        let r = alloc_local(bcx, local);
        bcx = r.bcx;
        bcx.fcx.lllocals.insert(local.node.id, r.val);
    }
    let r = rslt(bcx, C_nil());
    for s: @ast::stmt in b.node.stmts {
        r = trans_stmt(bcx, *s);
        bcx = r.bcx;

        // If we hit a terminator, control won't go any further so
        // we're in dead-code land. Stop here.
        if is_terminated(bcx) { ret r; }
    }
    fn accept_out_method(expr: @ast::expr) -> bool {
        ret alt expr.node {
              ast::expr_if(_, _, _) { true }
              ast::expr_alt(_, _) { true }
              ast::expr_block(_) { true }
              _ { false }
            };
    }
    alt b.node.expr {
      some(e) {
        let ccx = bcx_ccx(cx);
        let r_ty = ty::expr_ty(ccx.tcx, e);
        let pass = output != return && accept_out_method(e);
        if pass {
            r = trans_expr_out(bcx, e, output);
            bcx = r.bcx;
            if is_terminated(bcx) || ty::type_is_bot(ccx.tcx, r_ty) { ret r; }
        } else {
            let lv = trans_lval(bcx, e);
            r = {bcx: lv.bcx, val: lv.val};
            bcx = r.bcx;
            if is_terminated(bcx) || ty::type_is_bot(ccx.tcx, r_ty) { ret r; }
            alt output {
              save_in(target) {
                // The output method is to save the value at target,
                // and we didn't pass it to the recursive trans_expr
                // call.
                bcx = move_val_if_temp(bcx, INIT, target, lv, r_ty);
                r = rslt(bcx, C_nil());
              }
              return. { }
            }
        }
      }
      none. { r = rslt(bcx, C_nil()); }
    }
    bcx = trans_block_cleanups(bcx, find_scope_cx(bcx));
    ret rslt(bcx, r.val);
}

fn new_local_ctxt(ccx: @crate_ctxt) -> @local_ctxt {
    let pth: [str] = [];
    ret @{path: pth,
          module_path: [ccx.link_meta.name],
          obj_typarams: [],
          obj_fields: [],
          ccx: ccx};
}


// Creates the standard quartet of basic blocks: static allocas, copy args,
// derived tydescs, and dynamic allocas.
fn mk_standard_basic_blocks(llfn: ValueRef) ->
   {sa: BasicBlockRef,
    ca: BasicBlockRef,
    dt: BasicBlockRef,
    da: BasicBlockRef,
    rt: BasicBlockRef} {
    ret {sa:
             str::as_buf("static_allocas",
                         {|buf| llvm::LLVMAppendBasicBlock(llfn, buf) }),
         ca:
             str::as_buf("copy_args",
                         {|buf| llvm::LLVMAppendBasicBlock(llfn, buf) }),
         dt:
             str::as_buf("derived_tydescs",
                         {|buf| llvm::LLVMAppendBasicBlock(llfn, buf) }),
         da:
             str::as_buf("dynamic_allocas",
                         {|buf| llvm::LLVMAppendBasicBlock(llfn, buf) }),
         rt:
             str::as_buf("return",
                         {|buf| llvm::LLVMAppendBasicBlock(llfn, buf) })};
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn new_fn_ctxt_w_id(cx: @local_ctxt, sp: span, llfndecl: ValueRef,
                    id: ast::node_id, rstyle: ast::ret_style)
    -> @fn_ctxt {
    let llbbs = mk_standard_basic_blocks(llfndecl);
    ret @{llfn: llfndecl,
          lltaskptr: llvm::LLVMGetParam(llfndecl, 1u),
          llenv: llvm::LLVMGetParam(llfndecl, 2u),
          llretptr: llvm::LLVMGetParam(llfndecl, 0u),
          mutable llstaticallocas: llbbs.sa,
          mutable llcopyargs: llbbs.ca,
          mutable llderivedtydescs_first: llbbs.dt,
          mutable llderivedtydescs: llbbs.dt,
          mutable lldynamicallocas: llbbs.da,
          mutable llreturn: llbbs.rt,
          mutable llobstacktoken: none::<ValueRef>,
          mutable llself: none::<val_self_pair>,
          mutable lliterbody: none::<ValueRef>,
          mutable iterbodyty: none::<ty::t>,
          llargs: new_int_hash::<ValueRef>(),
          llobjfields: new_int_hash::<ValueRef>(),
          lllocals: new_int_hash::<ValueRef>(),
          llupvars: new_int_hash::<ValueRef>(),
          mutable lltydescs: [],
          derived_tydescs: map::mk_hashmap(ty::hash_ty, ty::eq_ty),
          id: id,
          ret_style: rstyle,
          sp: sp,
          lcx: cx};
}

fn new_fn_ctxt(cx: @local_ctxt, sp: span, llfndecl: ValueRef) -> @fn_ctxt {
    ret new_fn_ctxt_w_id(cx, sp, llfndecl, -1, ast::return_val);
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

// create_llargs_for_fn_args: Creates a mapping from incoming arguments to
// allocas created for them.
//
// When we translate a function, we need to map its incoming arguments to the
// spaces that have been created for them (by code in the llallocas field of
// the function's fn_ctxt).  create_llargs_for_fn_args populates the llargs
// field of the fn_ctxt with
fn create_llargs_for_fn_args(cx: @fn_ctxt, proto: ast::proto,
                             ty_self: option::t<ty::t>, ret_ty: ty::t,
                             args: [ast::arg], ty_params: [ast::ty_param]) {
    // Skip the implicit arguments 0, 1, and 2.  TODO: Pull out 3u and define
    // it as a constant, since we're using it in several places in trans this
    // way.
    let arg_n = 3u;
    alt ty_self {
      some(tt) { cx.llself = some::<val_self_pair>({v: cx.llenv, t: tt}); }
      none. {
        let i = 0u;
        for tp: ast::ty_param in ty_params {
            let llarg = llvm::LLVMGetParam(cx.llfn, arg_n);
            assert (llarg as int != 0);
            cx.lltydescs += [llarg];
            arg_n += 1u;
            i += 1u;
        }
      }
    }

    // If the function is actually an iter, populate the lliterbody field of
    // the function context with the ValueRef that we get from
    // llvm::LLVMGetParam for the iter's body.
    if proto == ast::proto_iter {
        cx.iterbodyty = some(ty::mk_iter_body_fn(fcx_tcx(cx), ret_ty));
        let llarg = llvm::LLVMGetParam(cx.llfn, arg_n);
        assert (llarg as int != 0);
        cx.lliterbody = some::<ValueRef>(llarg);
        arg_n += 1u;
    }

    // Populate the llargs field of the function context with the ValueRefs
    // that we get from llvm::LLVMGetParam for each argument.
    for arg: ast::arg in args {
        let llarg = llvm::LLVMGetParam(cx.llfn, arg_n);
        assert (llarg as int != 0);
        cx.llargs.insert(arg.id, llarg);
        arg_n += 1u;
    }
}

fn copy_args_to_allocas(fcx: @fn_ctxt, scope: @block_ctxt, args: [ast::arg],
                        arg_tys: [ty::arg], ignore_mut: bool) {
    let llcopyargs = new_raw_block_ctxt(fcx, fcx.llcopyargs);
    let bcx = llcopyargs;
    let arg_n: uint = 0u;
    for aarg: ast::arg in args {
        let arg_ty = arg_tys[arg_n].ty;
        alt aarg.mode {
          ast::by_ref. {
            let mutated =
                !ignore_mut && fcx.lcx.ccx.mut_map.contains_key(aarg.id);

            // Overwrite the llargs entry for locally mutated params
            // with a local alloca.
            if mutated {
                let aptr = bcx.fcx.llargs.get(aarg.id);
                let {bcx: bcx, val: alloc} = alloc_ty(bcx, arg_ty);
                bcx =
                    copy_val(bcx, INIT, alloc,
                             load_if_immediate(bcx, aptr, arg_ty), arg_ty);
                bcx.fcx.llargs.insert(aarg.id, alloc);
                add_clean(scope, alloc, arg_ty);
            }
          }
          ast::by_move. {
            add_clean(scope, bcx.fcx.llargs.get(aarg.id), arg_ty);
          }
          _ { }
        }
        arg_n += 1u;
    }
    fcx.llcopyargs = llcopyargs.llbb;
}

fn is_terminated(cx: @block_ctxt) -> bool {
    let inst = llvm::LLVMGetLastInstruction(cx.llbb);
    ret llvm::LLVMIsATerminatorInst(inst) as int != 0;
}

fn arg_tys_of_fn(ccx: @crate_ctxt, id: ast::node_id) -> [ty::arg] {
    alt ty::struct(ccx.tcx, ty::node_id_to_type(ccx.tcx, id)) {
      ty::ty_fn(_, arg_tys, _, _, _) { ret arg_tys; }
    }
}

fn populate_fn_ctxt_from_llself(fcx: @fn_ctxt, llself: val_self_pair) {
    let bcx = llstaticallocas_block_ctxt(fcx);
    let field_tys: [ty::t] = [];
    for f: ast::obj_field in bcx.fcx.lcx.obj_fields {
        field_tys += [node_id_type(bcx_ccx(bcx), f.id)];
    }
    // Synthesize a tuple type for the fields so that GEP_tup_like() can work
    // its magic.

    let fields_tup_ty = ty::mk_tup(fcx.lcx.ccx.tcx, field_tys);
    let n_typarams = std::vec::len::<ast::ty_param>(bcx.fcx.lcx.obj_typarams);
    let llobj_box_ty: TypeRef = T_obj_ptr(*bcx_ccx(bcx), n_typarams);
    let box_cell = GEP(bcx, llself.v, [C_int(0), C_int(abi::obj_field_box)]);
    let box_ptr = Load(bcx, box_cell);
    box_ptr = PointerCast(bcx, box_ptr, llobj_box_ty);
    let obj_typarams =
        GEP(bcx, box_ptr,
            [C_int(0), C_int(abi::box_rc_field_body),
             C_int(abi::obj_body_elt_typarams)]);

    // The object fields immediately follow the type parameters, so we skip
    // over them to get the pointer.
    let obj_fields =
        PointerCast(bcx, GEP(bcx, obj_typarams, [C_int(1)]),
                    T_ptr(type_of_or_i8(bcx, fields_tup_ty)));

    let i: int = 0;
    for p: ast::ty_param in fcx.lcx.obj_typarams {
        let lltyparam: ValueRef =
            GEP(bcx, obj_typarams, [C_int(0), C_int(i)]);
        lltyparam = Load(bcx, lltyparam);
        fcx.lltydescs += [lltyparam];
        i += 1;
    }
    i = 0;
    for f: ast::obj_field in fcx.lcx.obj_fields {
        // FIXME: silly check
        check type_is_tup_like(bcx, fields_tup_ty);
        let rslt = GEP_tup_like(bcx, fields_tup_ty, obj_fields, [0, i]);
        bcx = llstaticallocas_block_ctxt(fcx);
        let llfield = rslt.val;
        fcx.llobjfields.insert(f.id, llfield);
        i += 1;
    }
    fcx.llstaticallocas = bcx.llbb;
}


// Ties up the llstaticallocas -> llcopyargs -> llderivedtydescs ->
// lldynamicallocas -> lltop edges, and builds the return block.
fn finish_fn(fcx: @fn_ctxt, lltop: BasicBlockRef) {
    Br(new_raw_block_ctxt(fcx, fcx.llstaticallocas), fcx.llcopyargs);
    Br(new_raw_block_ctxt(fcx, fcx.llcopyargs), fcx.llderivedtydescs_first);
    Br(new_raw_block_ctxt(fcx, fcx.llderivedtydescs), fcx.lldynamicallocas);
    Br(new_raw_block_ctxt(fcx, fcx.lldynamicallocas), lltop);

    let ret_cx = new_raw_block_ctxt(fcx, fcx.llreturn);
    trans_fn_cleanups(fcx, ret_cx);
    RetVoid(ret_cx);
}

// trans_closure: Builds an LLVM function out of a source function.
// If the function closes over its environment a closure will be
// returned.
fn trans_closure(bcx_maybe: option::t<@block_ctxt>,
                 llfnty: option::t<TypeRef>, cx: @local_ctxt, sp: span,
                 f: ast::_fn, llfndecl: ValueRef, ty_self: option::t<ty::t>,
                 ty_params: [ast::ty_param], id: ast::node_id) ->
   option::t<{fn_pair: ValueRef, bcx: @block_ctxt}> {
    set_uwtable(llfndecl);

    // Set up arguments to the function.
    let fcx = new_fn_ctxt_w_id(cx, sp, llfndecl, id, f.decl.cf);
    create_llargs_for_fn_args(fcx, f.proto, ty_self,
                              ty::ret_ty_of_fn(cx.ccx.tcx, id), f.decl.inputs,
                              ty_params);
    alt fcx.llself {
      some(llself) { populate_fn_ctxt_from_llself(fcx, llself); }
      _ { }
    }

    // Create the first basic block in the function and keep a handle on it to
    //  pass to finish_fn later.
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;
    let block_ty = node_id_type(cx.ccx, f.body.node.id);

    let arg_tys = arg_tys_of_fn(fcx.lcx.ccx, id);
    copy_args_to_allocas(fcx, bcx, f.decl.inputs, arg_tys, false);

    // Figure out if we need to build a closure and act accordingly
    let res =
        alt f.proto {
          ast::proto_block. | ast::proto_closure. {
            let bcx = option::get(bcx_maybe);
            let upvars = get_freevars(cx.ccx.tcx, id);

            let copying = f.proto == ast::proto_closure;
            let env = build_closure(bcx, upvars, copying);
            load_environment(bcx, fcx, env.ptrty, upvars, copying);

            let closure =
                create_real_fn_pair(env.bcx, option::get(llfnty), llfndecl,
                                    env.ptr);
            if copying {
                add_clean_temp(bcx, closure, node_id_type(cx.ccx, id));
            }
            some({fn_pair: closure, bcx: env.bcx})
          }
          _ { none }
        };


    // This call to trans_block is the place where we bridge between
    // translation calls that don't have a return value (trans_crate,
    // trans_mod, trans_item, trans_obj, et cetera) and those that do
    // (trans_block, trans_expr, et cetera).
    let rslt =
        if !ty::type_is_bot(cx.ccx.tcx, block_ty) &&
           !ty::type_is_nil(cx.ccx.tcx, block_ty) &&
           f.proto != ast::proto_iter {
            trans_block(bcx, f.body, save_in(fcx.llretptr))
        } else { trans_block(bcx, f.body, return) };
    bcx = rslt.bcx;

    if !is_terminated(bcx) {
        // FIXME: until LLVM has a unit type, we are moving around
        // C_nil values rather than their void type.
        build_return(bcx);
    }

    // Insert the mandatory first few basic blocks before lltop.
    finish_fn(fcx, lltop);

    ret res;
}

fn trans_fn_inner(cx: @local_ctxt, sp: span, f: ast::_fn, llfndecl: ValueRef,
                  ty_self: option::t<ty::t>, ty_params: [ast::ty_param],
                  id: ast::node_id) {
    trans_closure(none, none, cx, sp, f, llfndecl, ty_self, ty_params, id);
}


// trans_fn: creates an LLVM function corresponding to a source language
// function.
fn trans_fn(cx: @local_ctxt, sp: span, f: ast::_fn, llfndecl: ValueRef,
            ty_self: option::t<ty::t>, ty_params: [ast::ty_param],
            id: ast::node_id) {
    if !cx.ccx.sess.get_opts().stats {
        trans_fn_inner(cx, sp, f, llfndecl, ty_self, ty_params, id);
        ret;
    }

    let start = time::get_time();
    trans_fn_inner(cx, sp, f, llfndecl, ty_self, ty_params, id);
    let end = time::get_time();
    log_fn_time(cx.ccx, str::connect(cx.path, "::"), start, end);
}

fn trans_res_ctor(cx: @local_ctxt, sp: span, dtor: ast::_fn,
                  ctor_id: ast::node_id, ty_params: [ast::ty_param]) {
    // Create a function for the constructor
    let llctor_decl;
    alt cx.ccx.item_ids.find(ctor_id) {
      some(x) { llctor_decl = x; }
      _ { cx.ccx.sess.span_fatal(sp, "unbound ctor_id in trans_res_ctor"); }
    }
    let fcx = new_fn_ctxt(cx, sp, llctor_decl);
    let ret_t = ty::ret_ty_of_fn(cx.ccx.tcx, ctor_id);
    create_llargs_for_fn_args(fcx, ast::proto_fn, none::<ty::t>, ret_t,
                              dtor.decl.inputs, ty_params);
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;
    let arg_t = arg_tys_of_fn(cx.ccx, ctor_id)[0].ty;
    let tup_t = ty::mk_tup(cx.ccx.tcx, [ty::mk_int(cx.ccx.tcx), arg_t]);
    let arg;
    alt fcx.llargs.find(dtor.decl.inputs[0].id) {
      some(x) { arg = load_if_immediate(bcx, x, arg_t); }
      _ { cx.ccx.sess.span_fatal(sp, "unbound dtor decl in trans_res_ctor"); }
    }
    let llretptr = fcx.llretptr;
    if ty::type_has_dynamic_size(cx.ccx.tcx, ret_t) {
        let llret_t = T_ptr(T_struct([T_i32(), llvm::LLVMTypeOf(arg)]));
        llretptr = BitCast(bcx, llretptr, llret_t);
    }

    // FIXME: silly checks
    check type_is_tup_like(bcx, tup_t);
    let dst = GEP_tup_like(bcx, tup_t, llretptr, [0, 1]);
    bcx = dst.bcx;
    bcx = copy_val(bcx, INIT, dst.val, arg, arg_t);
    check type_is_tup_like(bcx, tup_t);
    let flag = GEP_tup_like(bcx, tup_t, llretptr, [0, 0]);
    bcx = flag.bcx;
    Store(bcx, C_int(1), flag.val);
    build_return(bcx);
    finish_fn(fcx, lltop);
}


fn trans_tag_variant(cx: @local_ctxt, tag_id: ast::node_id,
                     variant: ast::variant, index: int, is_degen: bool,
                     ty_params: [ast::ty_param]) {
    if std::vec::len::<ast::variant_arg>(variant.node.args) == 0u {
        ret; // nullary constructors are just constants

    }
    // Translate variant arguments to function arguments.

    let fn_args: [ast::arg] = [];
    let i = 0u;
    for varg: ast::variant_arg in variant.node.args {
        fn_args +=
            [{mode: ast::by_ref,
              ty: varg.ty,
              ident: "arg" + uint::to_str(i, 10u),
              id: varg.id}];
    }
    assert (cx.ccx.item_ids.contains_key(variant.node.id));
    let llfndecl: ValueRef;
    alt cx.ccx.item_ids.find(variant.node.id) {
      some(x) { llfndecl = x; }
      _ {
        cx.ccx.sess.span_fatal(variant.span,
                               "unbound variant id in trans_tag_variant");
      }
    }
    let fcx = new_fn_ctxt(cx, variant.span, llfndecl);
    create_llargs_for_fn_args(fcx, ast::proto_fn, none::<ty::t>,
                              ty::ret_ty_of_fn(cx.ccx.tcx, variant.node.id),
                              fn_args, ty_params);
    let ty_param_substs: [ty::t] = [];
    i = 0u;
    for tp: ast::ty_param in ty_params {
        ty_param_substs += [ty::mk_param(cx.ccx.tcx, i, tp.kind)];
        i += 1u;
    }
    let arg_tys = arg_tys_of_fn(cx.ccx, variant.node.id);
    let bcx = new_top_block_ctxt(fcx);
    copy_args_to_allocas(fcx, bcx, fn_args, arg_tys, true);
    let lltop = bcx.llbb;

    // Cast the tag to a type we can GEP into.
    let llblobptr =
        if is_degen {
            fcx.llretptr
        } else {
            let lltagptr =
                PointerCast(bcx, fcx.llretptr,
                            T_opaque_tag_ptr(fcx.lcx.ccx.tn));
            let lldiscrimptr = GEP(bcx, lltagptr, [C_int(0), C_int(0)]);
            Store(bcx, C_int(index), lldiscrimptr);
            GEP(bcx, lltagptr, [C_int(0), C_int(1)])
        };
    i = 0u;
    let t_id = ast_util::local_def(tag_id);
    let v_id = ast_util::local_def(variant.node.id);
    for va: ast::variant_arg in variant.node.args {
        check (valid_variant_index(i, bcx, t_id, v_id));
        let rslt = GEP_tag(bcx, llblobptr, t_id, v_id, ty_param_substs, i);
        bcx = rslt.bcx;
        let lldestptr = rslt.val;
        // If this argument to this function is a tag, it'll have come in to
        // this function as an opaque blob due to the way that type_of()
        // works. So we have to cast to the destination's view of the type.

        let llargptr;
        alt fcx.llargs.find(va.id) {
          some(x) { llargptr = PointerCast(bcx, x, val_ty(lldestptr)); }
          none. {
            bcx_ccx(bcx).sess.bug("unbound argptr in \
                                      trans_tag_variant");
          }
        }
        let arg_ty = arg_tys[i].ty;
        let llargval;
        if ty::type_is_structural(cx.ccx.tcx, arg_ty) ||
               ty::type_has_dynamic_size(cx.ccx.tcx, arg_ty) ||
               ty::type_is_unique(cx.ccx.tcx, arg_ty) {
            llargval = llargptr;
        } else { llargval = Load(bcx, llargptr); }
        bcx = copy_val(bcx, INIT, lldestptr, llargval, arg_ty);
        i += 1u;
    }
    bcx = trans_block_cleanups(bcx, find_scope_cx(bcx));
    build_return(bcx);
    finish_fn(fcx, lltop);
}


// FIXME: this should do some structural hash-consing to avoid
// duplicate constants. I think. Maybe LLVM has a magical mode
// that does so later on?
fn trans_const_expr(cx: @crate_ctxt, e: @ast::expr) -> ValueRef {
    alt e.node {
      ast::expr_lit(lit) { ret trans_crate_lit(cx, *lit); }
      _ { cx.sess.span_unimpl(e.span, "consts that's not a plain literal"); }
    }
}

fn trans_const(cx: @crate_ctxt, e: @ast::expr, id: ast::node_id) {
    let v = trans_const_expr(cx, e);

    // The scalars come back as 1st class LLVM vals
    // which we have to stick into global constants.

    alt cx.consts.find(id) {
      some(g) {
        llvm::LLVMSetInitializer(g, v);
        llvm::LLVMSetGlobalConstant(g, True);
      }
      _ { cx.sess.span_fatal(e.span, "Unbound const in trans_const"); }
    }
}

fn trans_item(cx: @local_ctxt, item: ast::item) {
    alt item.node {
      ast::item_fn(f, tps) {
        let sub_cx = extend_path(cx, item.ident);
        alt cx.ccx.item_ids.find(item.id) {
          some(llfndecl) {
            trans_fn(sub_cx, item.span, f, llfndecl, none, tps, item.id);
          }
          _ {
            cx.ccx.sess.span_fatal(item.span,
                                   "unbound function item in trans_item");
          }
        }
      }
      ast::item_obj(ob, tps, ctor_id) {
        let sub_cx =
            @{obj_typarams: tps, obj_fields: ob.fields
                 with *extend_path(cx, item.ident)};
        trans_obj(sub_cx, item.span, ob, ctor_id, tps);
      }
      ast::item_res(dtor, dtor_id, tps, ctor_id) {
        trans_res_ctor(cx, item.span, dtor, ctor_id, tps);

        // Create a function for the destructor
        alt cx.ccx.item_ids.find(item.id) {
          some(lldtor_decl) {
            trans_fn(cx, item.span, dtor, lldtor_decl, none, tps, dtor_id);
          }
          _ {
            cx.ccx.sess.span_fatal(item.span, "unbound dtor in trans_item");
          }
        }
      }
      ast::item_mod(m) {
        let sub_cx =
            @{path: cx.path + [item.ident],
              module_path: cx.module_path + [item.ident] with *cx};
        trans_mod(sub_cx, m);
      }
      ast::item_tag(variants, tps) {
        let sub_cx = extend_path(cx, item.ident);
        let degen = std::vec::len(variants) == 1u;
        let i = 0;
        for variant: ast::variant in variants {
            trans_tag_variant(sub_cx, item.id, variant, i, degen, tps);
            i += 1;
        }
      }
      ast::item_const(_, expr) { trans_const(cx.ccx, expr, item.id); }
      _ {/* fall through */ }
    }
}


// Translate a module.  Doing this amounts to translating the items in the
// module; there ends up being no artifact (aside from linkage names) of
// separate modules in the compiled program.  That's because modules exist
// only as a convenience for humans working with the code, to organize names
// and control visibility.
fn trans_mod(cx: @local_ctxt, m: ast::_mod) {
    for item: @ast::item in m.items { trans_item(cx, *item); }
}

fn get_pair_fn_ty(llpairty: TypeRef) -> TypeRef {
    // Bit of a kludge: pick the fn typeref out of the pair.

    ret struct_elt(llpairty, 0u);
}

fn register_fn(ccx: @crate_ctxt, sp: span, path: [str], flav: str,
               ty_params: [ast::ty_param], node_id: ast::node_id) {
    // FIXME: pull this out
    let t = node_id_type(ccx, node_id);
    check returns_non_ty_var(ccx, t);
    register_fn_full(ccx, sp, path, flav, ty_params, node_id, t);
}

fn register_fn_full(ccx: @crate_ctxt, sp: span, path: [str], _flav: str,
                    ty_params: [ast::ty_param], node_id: ast::node_id,
                    node_type: ty::t)
    : returns_non_ty_var(ccx, node_type) {
    let path = path;
    let llfty =
        type_of_fn_from_ty(ccx, sp, node_type, std::vec::len(ty_params));
    alt ty::struct(ccx.tcx, node_type) {
      ty::ty_fn(proto, inputs, output, rs, _) {
        check non_ty_var(ccx, output);
        llfty = type_of_fn(ccx, sp, proto, false,
                           ast_util::ret_by_ref(rs), inputs, output,
                           vec::len(ty_params));
      }
      _ { ccx.sess.bug("register_fn(): fn item doesn't have fn type!"); }
    }
    let ps: str = mangle_exported_name(ccx, path, node_type);
    let llfn: ValueRef = decl_fastcall_fn(ccx.llmod, ps, llfty);
    ccx.item_ids.insert(node_id, llfn);
    ccx.item_symbols.insert(node_id, ps);

    let is_main: bool = is_main_name(path) && !ccx.sess.get_opts().library;
    if is_main { create_main_wrapper(ccx, sp, llfn, node_type); }
}

// Create a _rust_main(args: [str]) function which will be called from the
// runtime rust_start function
fn create_main_wrapper(ccx: @crate_ctxt, sp: span, main_llfn: ValueRef,
                       main_node_type: ty::t) {

    if ccx.main_fn != none::<ValueRef> {
        ccx.sess.span_fatal(sp, "multiple 'main' functions");
    }

    let main_takes_argv =
        alt ty::struct(ccx.tcx, main_node_type) {
          ty::ty_fn(_, args, _, _, _) { std::vec::len(args) != 0u }
        };

    let llfn = create_main(ccx, sp, main_llfn, main_takes_argv);
    ccx.main_fn = some(llfn);

    fn create_main(ccx: @crate_ctxt, sp: span, main_llfn: ValueRef,
                   takes_argv: bool) -> ValueRef {
        let unit_ty = ty::mk_str(ccx.tcx);
        let vecarg_ty: ty::arg =
            {mode: ast::by_ref,
             ty: ty::mk_vec(ccx.tcx, {ty: unit_ty, mut: ast::imm})};
        // FIXME: mk_nil should have a postcondition
        let nt = ty::mk_nil(ccx.tcx);
        check non_ty_var(ccx, nt);

        let llfty = type_of_fn(ccx, sp, ast::proto_fn, false, false,
                               [vecarg_ty], nt, 0u);
        let llfdecl = decl_fastcall_fn(ccx.llmod, "_rust_main", llfty);

        let fcx = new_fn_ctxt(new_local_ctxt(ccx), sp, llfdecl);

        let bcx = new_top_block_ctxt(fcx);
        let lltop = bcx.llbb;

        let lloutputarg = llvm::LLVMGetParam(llfdecl, 0u);
        let lltaskarg = llvm::LLVMGetParam(llfdecl, 1u);
        let llenvarg = llvm::LLVMGetParam(llfdecl, 2u);
        let args = [lloutputarg, lltaskarg, llenvarg];
        if takes_argv {
            let llargvarg = llvm::LLVMGetParam(llfdecl, 3u);
            // The runtime still passes the arg vector by value, this kludge
            // makes sure it becomes a pointer (to a pointer to a vec).
            let minus_ptr = llvm::LLVMGetElementType(val_ty(llargvarg));
            llargvarg = PointerCast(bcx, llargvarg, minus_ptr);
            args += [do_spill_noroot(bcx, llargvarg)];
        }
        FastCall(bcx, main_llfn, args);
        build_return(bcx);

        finish_fn(fcx, lltop);

        ret llfdecl;
    }
}

// Create a /real/ closure: this is like create_fn_pair, but creates a
// a fn value on the stack with a specified environment (which need not be
// on the stack).
fn create_real_fn_pair(cx: @block_ctxt, llfnty: TypeRef, llfn: ValueRef,
                       llenvptr: ValueRef) -> ValueRef {
    let lcx = cx.fcx.lcx;

    let pair = alloca(cx, T_fn_pair(*lcx.ccx, llfnty));
    let code_cell = GEP(cx, pair, [C_int(0), C_int(abi::fn_field_code)]);
    Store(cx, llfn, code_cell);
    let env_cell = GEP(cx, pair, [C_int(0), C_int(abi::fn_field_box)]);
    let llenvblobptr =
        PointerCast(cx, llenvptr, T_opaque_closure_ptr(*lcx.ccx));
    Store(cx, llenvblobptr, env_cell);
    ret pair;
}

// Returns the number of type parameters that the given native function has.
fn native_fn_ty_param_count(cx: @crate_ctxt, id: ast::node_id) -> uint {
    let count;
    let native_item =
        alt cx.ast_map.find(id) { some(ast_map::node_native_item(i)) { i } };
    alt native_item.node {
      ast::native_item_ty. {
        cx.sess.bug("register_native_fn(): native fn isn't \
                        actually a fn");
      }
      ast::native_item_fn(_, _, tps) {
        count = std::vec::len::<ast::ty_param>(tps);
      }
    }
    ret count;
}

fn native_fn_wrapper_type(cx: @crate_ctxt, sp: span, ty_param_count: uint,
                          x: ty::t) -> TypeRef {
    alt ty::struct(cx.tcx, x) {
      ty::ty_native_fn(abi, args, out) {
        check non_ty_var(cx, out);
        ret type_of_fn(cx, sp, ast::proto_fn, false, false, args, out,
                       ty_param_count);
      }
    }
}

fn register_native_fn(ccx: @crate_ctxt, sp: span, path: [str], name: str,
                           id: ast::node_id) {
    let path = path;
    let num_ty_param = native_fn_ty_param_count(ccx, id);
    // Declare the wrapper.

    let t = node_id_type(ccx, id);
    let wrapper_type = native_fn_wrapper_type(ccx, sp, num_ty_param, t);
    let ps: str = mangle_exported_name(ccx, path, node_id_type(ccx, id));
    let wrapper_fn = decl_fastcall_fn(ccx.llmod, ps, wrapper_type);
    ccx.item_ids.insert(id, wrapper_fn);
    ccx.item_symbols.insert(id, ps);

    // Build the wrapper.
    let fcx = new_fn_ctxt(new_local_ctxt(ccx), sp, wrapper_fn);
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;

    // Declare the function itself.
    let fn_type = node_id_type(ccx, id); // NB: has no type params

    let abi = ty::ty_fn_abi(ccx.tcx, fn_type);
    // FIXME: If the returned type is not nil, then we assume it's 32 bits
    // wide. This is obviously wildly unsafe. We should have a better FFI
    // that allows types of different sizes to be returned.

    let rty = ty::ty_fn_ret(ccx.tcx, fn_type);
    let rty_is_nil = ty::type_is_nil(ccx.tcx, rty);

    let pass_task;
    let uses_retptr;
    let cast_to_i32;
    alt abi {
      ast::native_abi_rust. {
        pass_task = true;
        uses_retptr = false;
        cast_to_i32 = true;
      }
      ast::native_abi_rust_intrinsic. {
        pass_task = true;
        uses_retptr = true;
        cast_to_i32 = false;
      }
      ast::native_abi_cdecl. {
        pass_task = false;
        uses_retptr = false;
        cast_to_i32 = true;
      }
      ast::native_abi_llvm. {
        pass_task = false;
        uses_retptr = false;
        cast_to_i32 = false;
      }
      ast::native_abi_x86stdcall. {
        pass_task = false;
        uses_retptr = false;
        cast_to_i32 = true;
      }
    }

    let lltaskptr;
    if cast_to_i32 {
        lltaskptr = vp2i(bcx, fcx.lltaskptr);
    } else { lltaskptr = fcx.lltaskptr; }

    let call_args: [ValueRef] = [];
    if pass_task { call_args += [lltaskptr]; }
    if uses_retptr { call_args += [bcx.fcx.llretptr]; }

    let arg_n = 3u;
    for each i: uint in uint::range(0u, num_ty_param) {
        let llarg = llvm::LLVMGetParam(fcx.llfn, arg_n);
        fcx.lltydescs += [llarg];
        assert (llarg as int != 0);
        if cast_to_i32 {
            call_args += [vp2i(bcx, llarg)];
        } else { call_args += [llarg]; }
        arg_n += 1u;
    }
    fn convert_arg_to_i32(cx: @block_ctxt, v: ValueRef, t: ty::t,
                          mode: ty::mode) -> ValueRef {
        if mode == ast::by_ref {
            if ty::type_is_integral(bcx_tcx(cx), t) {
                // FIXME: would be nice to have a postcondition that says
                // if a type is integral, then it has static size (#586)
                let lldsttype = T_int();
                let ccx = bcx_ccx(cx);
                let sp = cx.sp;
                check (type_has_static_size(ccx, t));
                let llsrctype = type_of(ccx, sp, t);
                if llvm::LLVMGetIntTypeWidth(lldsttype) >
                       llvm::LLVMGetIntTypeWidth(llsrctype) {
                    ret ZExtOrBitCast(cx, v, T_int());
                }
                ret TruncOrBitCast(cx, v, T_int());
            }
            if ty::type_is_fp(bcx_tcx(cx), t) { ret FPToSI(cx, v, T_int()); }
        }
        ret vp2i(cx, v);
    }

    fn trans_simple_native_abi(bcx: @block_ctxt, name: str,
                               &call_args: [ValueRef], fn_type: ty::t,
                               uses_retptr: bool, cc: uint) ->
       {val: ValueRef, rptr: ValueRef} {
        let call_arg_tys: [TypeRef] = [];
        for arg: ValueRef in call_args { call_arg_tys += [val_ty(arg)]; }
        let ccx = bcx_ccx(bcx);

        let llnativefnty =
            if uses_retptr {
                T_fn(call_arg_tys, T_void())
            } else {
                let fn_ret_ty = ty::ty_fn_ret(bcx_tcx(bcx), fn_type);
                // FIXME: Could follow from a constraint on fn_type...
                check (type_has_static_size(ccx, fn_ret_ty));
                let sp = bcx.sp;
                T_fn(call_arg_tys, type_of(ccx, sp, fn_ret_ty))
            };

        let llnativefn =
            get_extern_fn(ccx.externs, ccx.llmod, name, cc, llnativefnty);
        let r =
            if cc == lib::llvm::LLVMCCallConv {
                Call(bcx, llnativefn, call_args)
            } else { CallWithConv(bcx, llnativefn, call_args, cc) };
        let rptr = bcx.fcx.llretptr;
        ret {val: r, rptr: rptr};
    }

    let args = ty::ty_fn_args(ccx.tcx, fn_type);
    // Build up the list of arguments.

    let i = arg_n;
    for arg: ty::arg in args {
        let llarg = llvm::LLVMGetParam(fcx.llfn, i);
        if arg.mode == ast::by_ref {
            llarg = load_if_immediate(bcx, llarg, arg.ty);
        }
        assert (llarg as int != 0);
        if cast_to_i32 {
            let llarg_i32 = convert_arg_to_i32(bcx, llarg, arg.ty, arg.mode);
            call_args += [llarg_i32];
        } else { call_args += [llarg]; }
        i += 1u;
    }
    let r;
    let rptr;
    alt abi {
      ast::native_abi_llvm. {
        let result =
            trans_simple_native_abi(bcx, name, call_args, fn_type,
                                    uses_retptr, lib::llvm::LLVMCCallConv);
        r = result.val;
        rptr = result.rptr;
      }
      ast::native_abi_rust_intrinsic. {
        let external_name = "rust_intrinsic_" + name;
        let result =
            trans_simple_native_abi(bcx, external_name, call_args, fn_type,
                                    uses_retptr, lib::llvm::LLVMCCallConv);
        r = result.val;
        rptr = result.rptr;
      }
      ast::native_abi_x86stdcall. {
        let result =
            trans_simple_native_abi(bcx, name, call_args, fn_type,
                                    uses_retptr,
                                    lib::llvm::LLVMX86StdcallCallConv);
        r = result.val;
        rptr = result.rptr;
      }
      _ {
        r =
            trans_native_call(new_raw_block_ctxt(bcx.fcx, bcx.llbb),
                              ccx.externs, ccx.llmod, name, call_args);
        rptr = BitCast(bcx, fcx.llretptr, T_ptr(T_i32()));
      }
    }
    // We don't store the return value if it's nil, to avoid stomping on a nil
    // pointer. This is the only concession made to non-i32 return values. See
    // the FIXME above.

    if !rty_is_nil && !uses_retptr { Store(bcx, r, rptr); }

    build_return(bcx);
    finish_fn(fcx, lltop);
}

fn item_path(item: @ast::item) -> [str] { ret [item.ident]; }

fn collect_native_item(ccx: @crate_ctxt, i: @ast::native_item, pt: [str],
                       _v: vt<[str]>) {
    alt i.node {
      ast::native_item_fn(_, _, _) {
        if !ccx.obj_methods.contains_key(i.id) {
            register_native_fn(ccx, i.span, pt, i.ident, i.id);
        }
      }
      _ { }
    }
}

fn collect_item_1(ccx: @crate_ctxt, i: @ast::item, pt: [str], v: vt<[str]>) {
    visit::visit_item(i, pt + item_path(i), v);
    alt i.node {
      ast::item_const(_, _) {
        let typ = node_id_type(ccx, i.id);
        let s =
            mangle_exported_name(ccx, pt + [i.ident],
                                 node_id_type(ccx, i.id));
        // FIXME: Could follow from a constraint on types of const
        // items
        let g = str::as_buf(s, {|buf|
            check (type_has_static_size(ccx, typ));
            llvm::LLVMAddGlobal(ccx.llmod, type_of(ccx, i.span, typ), buf)
        });
        ccx.item_symbols.insert(i.id, s);
        ccx.consts.insert(i.id, g);
      }
      _ { }
    }
}

fn collect_item_2(ccx: @crate_ctxt, i: @ast::item, pt: [str], v: vt<[str]>) {
    let new_pt = pt + item_path(i);
    visit::visit_item(i, new_pt, v);
    alt i.node {
      ast::item_fn(f, tps) {
        if !ccx.obj_methods.contains_key(i.id) {
            register_fn(ccx, i.span, new_pt, "fn", tps, i.id);
        }
      }
      ast::item_obj(ob, tps, ctor_id) {
        register_fn(ccx, i.span, new_pt, "obj_ctor", tps, ctor_id);
        for m: @ast::method in ob.methods {
            ccx.obj_methods.insert(m.node.id, ());
        }
      }
      ast::item_res(_, dtor_id, tps, ctor_id) {
        register_fn(ccx, i.span, new_pt, "res_ctor", tps, ctor_id);
        // Note that the destructor is associated with the item's id, not
        // the dtor_id. This is a bit counter-intuitive, but simplifies
        // ty_res, which would have to carry around two def_ids otherwise
        // -- one to identify the type, and one to find the dtor symbol.
        let t = node_id_type(ccx, dtor_id);
        // FIXME: how to get rid of this check?
        check returns_non_ty_var(ccx, t);
        register_fn_full(ccx, i.span, new_pt, "res_dtor", tps, i.id, t);
      }
      _ { }
    }
}

fn collect_items(ccx: @crate_ctxt, crate: @ast::crate) {
    let visitor0 = visit::default_visitor();
    let visitor1 =
        @{visit_native_item: bind collect_native_item(ccx, _, _, _),
          visit_item: bind collect_item_1(ccx, _, _, _) with *visitor0};
    let visitor2 =
        @{visit_item: bind collect_item_2(ccx, _, _, _) with *visitor0};
    visit::visit_crate(*crate, [], visit::mk_vt(visitor1));
    visit::visit_crate(*crate, [], visit::mk_vt(visitor2));
}

fn collect_tag_ctor(ccx: @crate_ctxt, i: @ast::item, pt: [str],
                    v: vt<[str]>) {
    let new_pt = pt + item_path(i);
    visit::visit_item(i, new_pt, v);
    alt i.node {
      ast::item_tag(variants, tps) {
        for variant: ast::variant in variants {
            if std::vec::len(variant.node.args) != 0u {
                register_fn(ccx, i.span, new_pt + [variant.node.name],
                            "tag", tps, variant.node.id);
            }
        }
      }
      _ {/* fall through */ }
    }
}

fn collect_tag_ctors(ccx: @crate_ctxt, crate: @ast::crate) {
    let visitor =
        @{visit_item: bind collect_tag_ctor(ccx, _, _, _)
             with *visit::default_visitor()};
    visit::visit_crate(*crate, [], visit::mk_vt(visitor));
}


// The constant translation pass.
fn trans_constant(ccx: @crate_ctxt, it: @ast::item, pt: [str], v: vt<[str]>) {
    let new_pt = pt + item_path(it);
    visit::visit_item(it, new_pt, v);
    alt it.node {
      ast::item_tag(variants, _) {
        let i = 0u;
        let n_variants = std::vec::len::<ast::variant>(variants);
        while i < n_variants {
            let variant = variants[i];
            let p = new_pt + [it.ident, variant.node.name, "discrim"];
            let s = mangle_exported_name(ccx, p, ty::mk_int(ccx.tcx));
            let discrim_gvar =
                str::as_buf(s,
                            {|buf|
                                llvm::LLVMAddGlobal(ccx.llmod, T_int(), buf)
                            });
            llvm::LLVMSetInitializer(discrim_gvar, C_int(i as int));
            llvm::LLVMSetGlobalConstant(discrim_gvar, True);
            ccx.discrims.insert(variant.node.id, discrim_gvar);
            ccx.discrim_symbols.insert(variant.node.id, s);
            i += 1u;
        }
      }
      _ { }
    }
}

fn trans_constants(ccx: @crate_ctxt, crate: @ast::crate) {
    let visitor =
        @{visit_item: bind trans_constant(ccx, _, _, _)
             with *visit::default_visitor()};
    visit::visit_crate(*crate, [], visit::mk_vt(visitor));
}

fn vp2i(cx: @block_ctxt, v: ValueRef) -> ValueRef {
    ret PtrToInt(cx, v, T_int());
}

fn p2i(v: ValueRef) -> ValueRef { ret llvm::LLVMConstPtrToInt(v, T_int()); }

fn declare_intrinsics(llmod: ModuleRef) -> hashmap<str, ValueRef> {
    let T_memmove32_args: [TypeRef] =
        [T_ptr(T_i8()), T_ptr(T_i8()), T_i32(), T_i32(), T_i1()];
    let T_memmove64_args: [TypeRef] =
        [T_ptr(T_i8()), T_ptr(T_i8()), T_i64(), T_i32(), T_i1()];
    let T_memset32_args: [TypeRef] =
        [T_ptr(T_i8()), T_i8(), T_i32(), T_i32(), T_i1()];
    let T_memset64_args: [TypeRef] =
        [T_ptr(T_i8()), T_i8(), T_i64(), T_i32(), T_i1()];
    let T_trap_args: [TypeRef] = [];
    let gcroot =
        decl_cdecl_fn(llmod, "llvm.gcroot",
                      T_fn([T_ptr(T_ptr(T_i8())), T_ptr(T_i8())], T_void()));
    let gcread =
        decl_cdecl_fn(llmod, "llvm.gcread",
                      T_fn([T_ptr(T_i8()), T_ptr(T_ptr(T_i8()))], T_void()));
    let memmove32 =
        decl_cdecl_fn(llmod, "llvm.memmove.p0i8.p0i8.i32",
                      T_fn(T_memmove32_args, T_void()));
    let memmove64 =
        decl_cdecl_fn(llmod, "llvm.memmove.p0i8.p0i8.i64",
                      T_fn(T_memmove64_args, T_void()));
    let memset32 =
        decl_cdecl_fn(llmod, "llvm.memset.p0i8.i32",
                      T_fn(T_memset32_args, T_void()));
    let memset64 =
        decl_cdecl_fn(llmod, "llvm.memset.p0i8.i64",
                      T_fn(T_memset64_args, T_void()));
    let trap = decl_cdecl_fn(llmod, "llvm.trap", T_fn(T_trap_args, T_void()));
    let intrinsics = new_str_hash::<ValueRef>();
    intrinsics.insert("llvm.gcroot", gcroot);
    intrinsics.insert("llvm.gcread", gcread);
    intrinsics.insert("llvm.memmove.p0i8.p0i8.i32", memmove32);
    intrinsics.insert("llvm.memmove.p0i8.p0i8.i64", memmove64);
    intrinsics.insert("llvm.memset.p0i8.i32", memset32);
    intrinsics.insert("llvm.memset.p0i8.i64", memset64);
    intrinsics.insert("llvm.trap", trap);
    ret intrinsics;
}

fn trap(bcx: @block_ctxt) {
    let v: [ValueRef] = [];
    alt bcx_ccx(bcx).intrinsics.find("llvm.trap") {
      some(x) { Call(bcx, x, v); }
      _ { bcx_ccx(bcx).sess.bug("unbound llvm.trap in trap"); }
    }
}

fn decl_no_op_type_glue(llmod: ModuleRef, taskptr_type: TypeRef) -> ValueRef {
    let ty = T_fn([taskptr_type, T_ptr(T_i8())], T_void());
    ret decl_fastcall_fn(llmod, abi::no_op_type_glue_name(), ty);
}

fn make_glues(llmod: ModuleRef, taskptr_type: TypeRef) -> @glue_fns {
    ret @{no_op_type_glue: decl_no_op_type_glue(llmod, taskptr_type)};
}

fn make_common_glue(sess: session::session, output: str) {
    // FIXME: part of this is repetitive and is probably a good idea
    // to autogen it.
    let task_type = T_task();
    let taskptr_type = T_ptr(task_type);

    let llmod = str::as_buf("rust_out", {|buf|
        llvm::LLVMModuleCreateWithNameInContext
            (buf, llvm::LLVMGetGlobalContext())
    });
    let _: () =
        str::as_buf(x86::get_data_layout(),
                    {|buf| llvm::LLVMSetDataLayout(llmod, buf) });
    let _: () =
        str::as_buf(x86::get_target_triple(),
                    {|buf| llvm::LLVMSetTarget(llmod, buf) });
    mk_target_data(x86::get_data_layout());
    declare_intrinsics(llmod);
    let _: () =
        str::as_buf(x86::get_module_asm(),
                    {|buf| llvm::LLVMSetModuleInlineAsm(llmod, buf) });
    make_glues(llmod, taskptr_type);
    link::write::run_passes(sess, llmod, output);
}

fn create_module_map(ccx: @crate_ctxt) -> ValueRef {
    let elttype = T_struct([T_int(), T_int()]);
    let maptype = T_array(elttype, ccx.module_data.size() + 1u);
    let map =
        str::as_buf("_rust_mod_map",
                    {|buf| llvm::LLVMAddGlobal(ccx.llmod, maptype, buf) });
    llvm::LLVMSetLinkage(map,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    let elts: [ValueRef] = [];
    for each item: @{key: str, val: ValueRef} in ccx.module_data.items() {
        let elt = C_struct([p2i(C_cstr(ccx, item.key)), p2i(item.val)]);
        elts += [elt];
    }
    let term = C_struct([C_int(0), C_int(0)]);
    elts += [term];
    llvm::LLVMSetInitializer(map, C_array(elttype, elts));
    ret map;
}


// FIXME use hashed metadata instead of crate names once we have that
fn create_crate_map(ccx: @crate_ctxt) -> ValueRef {
    let subcrates: [ValueRef] = [];
    let i = 1;
    let cstore = ccx.sess.get_cstore();
    while cstore::have_crate_data(cstore, i) {
        let nm = "_rust_crate_map_" + cstore::get_crate_data(cstore, i).name;
        let cr =
            str::as_buf(nm,
                        {|buf|
                            llvm::LLVMAddGlobal(ccx.llmod, T_int(), buf)
                        });
        subcrates += [p2i(cr)];
        i += 1;
    }
    subcrates += [C_int(0)];
    let mapname;
    if ccx.sess.get_opts().library {
        mapname = ccx.link_meta.name;
    } else { mapname = "toplevel"; }
    let sym_name = "_rust_crate_map_" + mapname;
    let arrtype = T_array(T_int(), std::vec::len::<ValueRef>(subcrates));
    let maptype = T_struct([T_int(), arrtype]);
    let map =
        str::as_buf(sym_name,
                    {|buf| llvm::LLVMAddGlobal(ccx.llmod, maptype, buf) });
    llvm::LLVMSetLinkage(map,
                         lib::llvm::LLVMExternalLinkage as llvm::Linkage);
    llvm::LLVMSetInitializer(map,
                             C_struct([p2i(create_module_map(ccx)),
                                       C_array(T_int(), subcrates)]));
    ret map;
}

fn write_metadata(cx: @crate_ctxt, crate: @ast::crate) {
    if !cx.sess.get_opts().library { ret; }
    let llmeta = C_postr(metadata::encoder::encode_metadata(cx, crate));
    let llconst = trans_common::C_struct([llmeta]);
    let llglobal =
        str::as_buf("rust_metadata",
                    {|buf|
                        llvm::LLVMAddGlobal(cx.llmod, val_ty(llconst), buf)
                    });
    llvm::LLVMSetInitializer(llglobal, llconst);
    let _: () =
        str::as_buf(x86::get_meta_sect_name(),
                    {|buf| llvm::LLVMSetSection(llglobal, buf) });
    llvm::LLVMSetLinkage(llglobal,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);

    let t_ptr_i8 = T_ptr(T_i8());
    llglobal = llvm::LLVMConstBitCast(llglobal, t_ptr_i8);
    let llvm_used =
        str::as_buf("llvm.used",
                    {|buf|
                        llvm::LLVMAddGlobal(cx.llmod, T_array(t_ptr_i8, 1u),
                                            buf)
                    });
    llvm::LLVMSetLinkage(llvm_used,
                         lib::llvm::LLVMAppendingLinkage as llvm::Linkage);
    llvm::LLVMSetInitializer(llvm_used, C_array(t_ptr_i8, [llglobal]));
}

// Writes the current ABI version into the crate.
fn write_abi_version(ccx: @crate_ctxt) {
    shape::mk_global(ccx, "rust_abi_version", C_uint(abi::abi_version),
                     false);
}

fn trans_crate(sess: session::session, crate: @ast::crate, tcx: ty::ctxt,
               output: str, amap: ast_map::map, mut_map: mut::mut_map,
               copy_map: alias::copy_map) -> ModuleRef {
    let llmod = str::as_buf("rust_out", {|buf|
        llvm::LLVMModuleCreateWithNameInContext
            (buf, llvm::LLVMGetGlobalContext())
    });
    let _: () =
        str::as_buf(x86::get_data_layout(),
                    {|buf| llvm::LLVMSetDataLayout(llmod, buf) });
    let _: () =
        str::as_buf(x86::get_target_triple(),
                    {|buf| llvm::LLVMSetTarget(llmod, buf) });
    let td = mk_target_data(x86::get_data_layout());
    let tn = mk_type_names();
    let intrinsics = declare_intrinsics(llmod);
    let task_type = T_task();
    let taskptr_type = T_ptr(task_type);
    tn.associate("taskptr", taskptr_type);
    let tydesc_type = T_tydesc(taskptr_type);
    tn.associate("tydesc", tydesc_type);
    let glues = make_glues(llmod, taskptr_type);
    let hasher = ty::hash_ty;
    let eqer = ty::eq_ty;
    let tag_sizes = map::mk_hashmap::<ty::t, uint>(hasher, eqer);
    let tydescs = map::mk_hashmap::<ty::t, @tydesc_info>(hasher, eqer);
    let lltypes = map::mk_hashmap::<ty::t, TypeRef>(hasher, eqer);
    let sha1s = map::mk_hashmap::<ty::t, str>(hasher, eqer);
    let short_names = map::mk_hashmap::<ty::t, str>(hasher, eqer);
    let sha = std::sha1::mk_sha1();
    let ccx =
        @{sess: sess,
          llmod: llmod,
          td: td,
          tn: tn,
          externs: new_str_hash::<ValueRef>(),
          intrinsics: intrinsics,
          item_ids: new_int_hash::<ValueRef>(),
          ast_map: amap,
          item_symbols: new_int_hash::<str>(),
          mutable main_fn: none::<ValueRef>,
          link_meta: link::build_link_meta(sess, *crate, output, sha),
          tag_sizes: tag_sizes,
          discrims: new_int_hash::<ValueRef>(),
          discrim_symbols: new_int_hash::<str>(),
          consts: new_int_hash::<ValueRef>(),
          obj_methods: new_int_hash::<()>(),
          tydescs: tydescs,
          module_data: new_str_hash::<ValueRef>(),
          lltypes: lltypes,
          glues: glues,
          names: namegen(0),
          sha: sha,
          type_sha1s: sha1s,
          type_short_names: short_names,
          tcx: tcx,
          mut_map: mut_map,
          copy_map: copy_map,
          stats:
              {mutable n_static_tydescs: 0u,
               mutable n_derived_tydescs: 0u,
               mutable n_glues_created: 0u,
               mutable n_null_glues: 0u,
               mutable n_real_glues: 0u,
               fn_times: @mutable []},
          upcalls:
              upcall::declare_upcalls(tn, tydesc_type, taskptr_type, llmod),
          rust_object_type: T_rust_object(),
          tydesc_type: tydesc_type,
          task_type: task_type,
          builder: BuilderRef_res(llvm::LLVMCreateBuilder()),
          shape_cx: shape::mk_ctxt(llmod),
          gc_cx: gc::mk_ctxt()};
    let cx = new_local_ctxt(ccx);
    collect_items(ccx, crate);
    collect_tag_ctors(ccx, crate);
    trans_constants(ccx, crate);
    trans_mod(cx, crate.node.module);
    create_crate_map(ccx);
    emit_tydescs(ccx);
    shape::gen_shape_tables(ccx);
    write_abi_version(ccx);

    // Translate the metadata.
    write_metadata(cx.ccx, crate);
    if ccx.sess.get_opts().stats {
        log_err "--- trans stats ---";
        log_err #fmt["n_static_tydescs: %u", ccx.stats.n_static_tydescs];
        log_err #fmt["n_derived_tydescs: %u", ccx.stats.n_derived_tydescs];
        log_err #fmt["n_glues_created: %u", ccx.stats.n_glues_created];
        log_err #fmt["n_null_glues: %u", ccx.stats.n_null_glues];
        log_err #fmt["n_real_glues: %u", ccx.stats.n_real_glues];


        for timing: {ident: str, time: int} in *ccx.stats.fn_times {
            log_err #fmt["time: %s took %d ms", timing.ident, timing.time];
        }
    }
    ret llmod;
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
