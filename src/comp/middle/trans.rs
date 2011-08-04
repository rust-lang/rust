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
import std::int;
import std::str;
import std::uint;
import std::str::rustrt::sbuf;
import std::map;
import std::map::hashmap;
import std::option;
import std::option::some;
import std::option::none;
import std::fs;
import std::time;
import syntax::ast;
import driver::session;
import middle::ty;
import middle::freevars::*;
import back::link;
import back::x86;
import back::abi;
import back::upcall;
import syntax::visit;
import visit::vt;
import util::common;
import util::common::*;
import std::map::new_int_hash;
import std::map::new_str_hash;
import syntax::codemap::span;
import lib::llvm::llvm;
import lib::llvm::builder;
import lib::llvm::new_builder;
import lib::llvm::target_data;
import lib::llvm::type_names;
import lib::llvm::mk_target_data;
import lib::llvm::mk_type_names;
import lib::llvm::llvm::ModuleRef;
import lib::llvm::llvm::ValueRef;
import lib::llvm::llvm::TypeRef;
import lib::llvm::llvm::TypeHandleRef;
import lib::llvm::llvm::BuilderRef;
import lib::llvm::llvm::BasicBlockRef;
import lib::llvm::False;
import lib::llvm::True;
import lib::llvm::Bool;
import link::mangle_internal_name_by_type_only;
import link::mangle_internal_name_by_seq;
import link::mangle_internal_name_by_path;
import link::mangle_internal_name_by_path_and_seq;
import link::mangle_exported_name;
import metadata::creader;
import metadata::csearch;
import metadata::cstore;
import util::ppaux::ty_to_str;
import util::ppaux::ty_to_short_str;
import syntax::print::pprust::expr_to_str;
import syntax::print::pprust::path_to_str;

import trans_common::*;

import trans_comm::trans_port;
import trans_comm::trans_chan;
import trans_comm::trans_spawn;
import trans_comm::trans_send;
import trans_comm::trans_recv;

// This function now fails if called on a type with dynamic size (as its
// return value was always meaningless in that case anyhow). Beware!
//
// TODO: Enforce via a predicate.
fn type_of(cx: &@crate_ctxt, sp: &span, t: &ty::t) -> TypeRef {
    if ty::type_has_dynamic_size(cx.tcx, t) {
        cx.sess.span_fatal(sp,
                           "type_of() called on a type with dynamic size: " +
                               ty_to_str(cx.tcx, t));
    }
    ret type_of_inner(cx, sp, t);
}

fn type_of_explicit_args(cx: &@crate_ctxt, sp: &span, inputs: &ty::arg[]) ->
   TypeRef[] {
    let atys: TypeRef[] = ~[];
    for arg: ty::arg  in inputs {
        let t: TypeRef = type_of_inner(cx, sp, arg.ty);
        t = alt arg.mode {
          ty::mo_alias(_) { T_ptr(t) }
          _ { t }
        };
        atys += ~[t];
    }
    ret atys;
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn_full
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn type_of_fn_full(cx: &@crate_ctxt, sp: &span, proto: ast::proto,
                   is_method: bool, inputs: &ty::arg[], output: &ty::t,
                   ty_param_count: uint) -> TypeRef {
    let atys: TypeRef[] = ~[];

    // Arg 0: Output pointer.
    atys += ~[T_ptr(type_of_inner(cx, sp, output))];

    // Arg 1: task pointer.
    atys += ~[T_taskptr(*cx)];

    // Arg 2: Env (closure-bindings / self-obj)
    if is_method {
        atys += ~[cx.rust_object_type];
    } else { atys += ~[T_opaque_closure_ptr(*cx)]; }

    // Args >3: ty params, if not acquired via capture...
    if !is_method {
        let i = 0u;
        while i < ty_param_count {
            atys += ~[T_ptr(cx.tydesc_type)];
            i += 1u;
        }
    }
    if proto == ast::proto_iter {
        // If it's an iter, the 'output' type of the iter is actually the
        // *input* type of the function we're given as our iter-block
        // argument.
        atys +=
            ~[type_of_inner(cx, sp, ty::mk_iter_body_fn(cx.tcx, output))];
    }

    // ... then explicit args.
    atys += type_of_explicit_args(cx, sp, inputs);
    ret T_fn(atys, llvm::LLVMVoidType());
}

fn type_of_fn(cx: &@crate_ctxt, sp: &span, proto: ast::proto,
              inputs: &ty::arg[], output: &ty::t, ty_param_count: uint) ->
   TypeRef {
    ret type_of_fn_full(cx, sp, proto, false, inputs, output, ty_param_count);
}

// Given a function type and a count of ty params, construct an llvm type
fn type_of_fn_from_ty(cx: &@crate_ctxt, sp: &span,
                      fty: &ty::t, ty_param_count: uint) -> TypeRef {
    ret type_of_fn(cx, sp,
                   ty::ty_fn_proto(cx.tcx, fty),
                   ty::ty_fn_args(cx.tcx, fty),
                   ty::ty_fn_ret(cx.tcx, fty),
                   ty_param_count);
}

fn type_of_native_fn(cx: &@crate_ctxt, sp: &span, abi: ast::native_abi,
                     inputs: &ty::arg[], output: &ty::t, ty_param_count: uint)
   -> TypeRef {
    let atys: TypeRef[] = ~[];
    if abi == ast::native_abi_rust {
        atys += ~[T_taskptr(*cx)];
        let i = 0u;
        while i < ty_param_count {
            atys += ~[T_ptr(cx.tydesc_type)];
            i += 1u;
        }
    }
    atys += type_of_explicit_args(cx, sp, inputs);
    ret T_fn(atys, type_of_inner(cx, sp, output));
}

fn type_of_inner(cx: &@crate_ctxt, sp: &span, t: &ty::t) -> TypeRef {
    // Check the cache.

    if cx.lltypes.contains_key(t) { ret cx.lltypes.get(t); }
    let llty: TypeRef = 0 as TypeRef;
    alt ty::struct(cx.tcx, t) {
      ty::ty_native(_) { llty = T_ptr(T_i8()); }
      ty::ty_nil. { llty = T_nil(); }
      ty::ty_bot. {
        llty = T_nil(); /* ...I guess? */

      }
      ty::ty_bool. { llty = T_bool(); }
      ty::ty_int. { llty = T_int(); }
      ty::ty_float. { llty = T_float(); }
      ty::ty_uint. { llty = T_int(); }
      ty::ty_machine(tm) {
        alt tm {
          ast::ty_i8. { llty = T_i8(); }
          ast::ty_u8. { llty = T_i8(); }
          ast::ty_i16. { llty = T_i16(); }
          ast::ty_u16. { llty = T_i16(); }
          ast::ty_i32. { llty = T_i32(); }
          ast::ty_u32. { llty = T_i32(); }
          ast::ty_i64. { llty = T_i64(); }
          ast::ty_u64. { llty = T_i64(); }
          ast::ty_f32. { llty = T_f32(); }
          ast::ty_f64. { llty = T_f64(); }
        }
      }
      ty::ty_char. { llty = T_char(); }
      ty::ty_str. { llty = T_ptr(T_str()); }
      ty::ty_istr. { llty = T_ivec(T_i8()); }
      ty::ty_tag(did, _) { llty = type_of_tag(cx, sp, did, t); }
      ty::ty_box(mt) { llty = T_ptr(T_box(type_of_inner(cx, sp, mt.ty))); }
      ty::ty_vec(mt) { llty = T_ptr(T_vec(type_of_inner(cx, sp, mt.ty))); }
      ty::ty_ivec(mt) {
        if ty::type_has_dynamic_size(cx.tcx, mt.ty) {
            llty = T_opaque_ivec();
        } else { llty = T_ivec(type_of_inner(cx, sp, mt.ty)); }
      }
      ty::ty_ptr(mt) { llty = T_ptr(type_of_inner(cx, sp, mt.ty)); }
      ty::ty_port(t) { llty = T_ptr(T_port(type_of_inner(cx, sp, t))); }
      ty::ty_chan(t) { llty = T_ptr(T_chan(type_of_inner(cx, sp, t))); }
      ty::ty_task. { llty = T_taskptr(*cx); }
      ty::ty_rec(fields) {
        let tys: TypeRef[] = ~[];
        for f: ty::field  in fields {
            tys += ~[type_of_inner(cx, sp, f.mt.ty)];
        }
        llty = T_struct(tys);
      }
      ty::ty_fn(_, _, _, _, _) {
        llty = T_fn_pair(*cx, type_of_fn_from_ty(cx, sp, t, 0u));
      }
      ty::ty_native_fn(abi, args, out) {
        let nft = native_fn_wrapper_type(cx, sp, 0u, t);
        llty = T_fn_pair(*cx, nft);
      }
      ty::ty_obj(meths) { llty = cx.rust_object_type; }
      ty::ty_res(_, sub, tps) {
        let sub1 = ty::substitute_type_params(cx.tcx, tps, sub);
        ret T_struct(~[T_i32(), type_of_inner(cx, sp, sub1)]);
      }
      ty::ty_var(_) {
        cx.tcx.sess.span_fatal(sp, "trans::type_of called on ty_var");
      }
      ty::ty_param(_, _) { llty = T_typaram(cx.tn); }
      ty::ty_type. { llty = T_ptr(cx.tydesc_type); }
    }
    assert (llty as int != 0);
    cx.lltypes.insert(t, llty);
    ret llty;
}

fn type_of_tag(cx: &@crate_ctxt, sp: &span, did: &ast::def_id, t: &ty::t) ->
   TypeRef {
    let degen = std::ivec::len(ty::tag_variants(cx.tcx, did)) == 1u;
    if ty::type_has_dynamic_size(cx.tcx, t) {
        if degen { ret T_i8(); } else { ret T_opaque_tag(cx.tn); }
    } else {
        let size = static_size_of_tag(cx, sp, t);
        if !degen { ret T_tag(cx.tn, size); }
        // LLVM does not like 0-size arrays, apparently
        if size == 0u { size = 1u; }
        ret T_array(T_i8(), size);
    }
}

fn type_of_ty_param_kinds_and_ty(lcx: @local_ctxt, sp: &span,
                                 tpt: &ty::ty_param_kinds_and_ty) -> TypeRef {
    alt ty::struct(lcx.ccx.tcx, tpt.ty) {
      ty::ty_fn(_, _, _, _, _) {
        let llfnty = type_of_fn_from_ty(lcx.ccx, sp, tpt.ty,
                                        std::ivec::len(tpt.kinds));
        ret T_fn_pair(*lcx.ccx, llfnty);
      }
      _ {
        // fall through
      }
    }
    ret type_of(lcx.ccx, sp, tpt.ty);
}

fn type_of_or_i8(bcx: &@block_ctxt, typ: ty::t) -> TypeRef {
    if ty::type_has_dynamic_size(bcx_tcx(bcx), typ) { ret T_i8(); }
    ret type_of(bcx_ccx(bcx), bcx.sp, typ);
}


// Name sanitation. LLVM will happily accept identifiers with weird names, but
// gas doesn't!
fn sanitize(s: &str) -> str {
    let result = "";
    for c: u8  in s {
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
                        result += str::from_bytes(v);
                    }
                }
            }
        }
    }
    ret result;
}


fn log_fn_time(ccx: &@crate_ctxt, name: str, start: &time::timeval,
               end: &time::timeval) {
    let elapsed =
        1000 * (end.sec - start.sec as int) +
            ((end.usec as int) - (start.usec as int)) / 1000;
    *ccx.stats.fn_times += ~[{ident: name, time: elapsed}];
}


fn decl_fn(llmod: ModuleRef, name: &str, cc: uint, llty: TypeRef) ->
   ValueRef {
    let llfn: ValueRef = llvm::LLVMAddFunction(llmod, str::buf(name), llty);
    llvm::LLVMSetFunctionCallConv(llfn, cc);
    ret llfn;
}

fn decl_cdecl_fn(llmod: ModuleRef, name: &str, llty: TypeRef) -> ValueRef {
    ret decl_fn(llmod, name, lib::llvm::LLVMCCallConv, llty);
}

fn decl_fastcall_fn(llmod: ModuleRef, name: &str, llty: TypeRef) -> ValueRef {
    ret decl_fn(llmod, name, lib::llvm::LLVMFastCallConv, llty);
}


// Only use this if you are going to actually define the function. It's
// not valid to simply declare a function as internal.
fn decl_internal_fastcall_fn(llmod: ModuleRef, name: &str, llty: TypeRef) ->
   ValueRef {
    let llfn = decl_fn(llmod, name, lib::llvm::LLVMFastCallConv, llty);
    llvm::LLVMSetLinkage(llfn,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    ret llfn;
}

fn decl_glue(llmod: ModuleRef, cx: &crate_ctxt, s: &str) -> ValueRef {
    ret decl_cdecl_fn(llmod, s, T_fn(~[T_taskptr(cx)], T_void()));
}

fn get_extern_fn(externs: &hashmap[str, ValueRef], llmod: ModuleRef,
                 name: &str, cc: uint, ty: TypeRef) -> ValueRef {
    if externs.contains_key(name) { ret externs.get(name); }
    let f = decl_fn(llmod, name, cc, ty);
    externs.insert(name, f);
    ret f;
}

fn get_extern_const(externs: &hashmap[str, ValueRef], llmod: ModuleRef,
                    name: &str, ty: TypeRef) -> ValueRef {
    if externs.contains_key(name) { ret externs.get(name); }
    let c = llvm::LLVMAddGlobal(llmod, ty, str::buf(name));
    externs.insert(name, c);
    ret c;
}

fn get_simple_extern_fn(externs: &hashmap[str, ValueRef], llmod: ModuleRef,
                        name: &str, n_args: int) -> ValueRef {
    let inputs = std::ivec::init_elt[TypeRef](T_int(), n_args as uint);
    let output = T_int();
    let t = T_fn(inputs, output);
    ret get_extern_fn(externs, llmod, name, lib::llvm::LLVMCCallConv, t);
}

fn trans_native_call(b: &builder, glues: @glue_fns, lltaskptr: ValueRef,
                     externs: &hashmap[str, ValueRef], tn: &type_names,
                     llmod: ModuleRef, name: &str, pass_task: bool,
                     args: &ValueRef[]) -> ValueRef {
    let n: int = std::ivec::len[ValueRef](args) as int;
    let llnative: ValueRef = get_simple_extern_fn(externs, llmod, name, n);
    let call_args: ValueRef[] = ~[];
    for a: ValueRef  in args { call_args += ~[b.ZExtOrBitCast(a, T_int())]; }
    ret b.Call(llnative, call_args);
}

fn trans_non_gc_free(cx: &@block_ctxt, v: ValueRef) -> result {
    cx.build.Call(bcx_ccx(cx).upcalls.free,
                  ~[cx.fcx.lltaskptr, cx.build.PointerCast(v, T_ptr(T_i8())),
                    C_int(0)]);
    ret rslt(cx, C_int(0));
}

fn trans_shared_free(cx: &@block_ctxt, v: ValueRef) -> result {
    cx.build.Call(bcx_ccx(cx).upcalls.shared_free,
                  ~[cx.fcx.lltaskptr,
                    cx.build.PointerCast(v, T_ptr(T_i8()))]);
    ret rslt(cx, C_int(0));
}

fn umax(cx: &@block_ctxt, a: ValueRef, b: ValueRef) -> ValueRef {
    let cond = cx.build.ICmp(lib::llvm::LLVMIntULT, a, b);
    ret cx.build.Select(cond, b, a);
}

fn umin(cx: &@block_ctxt, a: ValueRef, b: ValueRef) -> ValueRef {
    let cond = cx.build.ICmp(lib::llvm::LLVMIntULT, a, b);
    ret cx.build.Select(cond, a, b);
}

fn align_to(cx: &@block_ctxt, off: ValueRef, align: ValueRef) -> ValueRef {
    let mask = cx.build.Sub(align, C_int(1));
    let bumped = cx.build.Add(off, mask);
    ret cx.build.And(bumped, cx.build.Not(mask));
}


// Returns the real size of the given type for the current target.
fn llsize_of_real(cx: &@crate_ctxt, t: TypeRef) -> uint {
    ret llvm::LLVMStoreSizeOfType(cx.td.lltd, t);
}

fn llsize_of(t: TypeRef) -> ValueRef {
    ret llvm::LLVMConstIntCast(lib::llvm::llvm::LLVMSizeOf(t), T_int(),
                               False);
}

fn llalign_of(t: TypeRef) -> ValueRef {
    ret llvm::LLVMConstIntCast(lib::llvm::llvm::LLVMAlignOf(t), T_int(),
                               False);
}

fn size_of(cx: &@block_ctxt, t: &ty::t) -> result {
    if !ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        ret rslt(cx, llsize_of(type_of(bcx_ccx(cx), cx.sp, t)));
    }
    ret dynamic_size_of(cx, t);
}

fn align_of(cx: &@block_ctxt, t: &ty::t) -> result {
    if !ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        ret rslt(cx, llalign_of(type_of(bcx_ccx(cx), cx.sp, t)));
    }
    ret dynamic_align_of(cx, t);
}

fn alloca(cx: &@block_ctxt, t: TypeRef) -> ValueRef {
    ret new_builder(cx.fcx.llstaticallocas).Alloca(t);
}

fn array_alloca(cx: &@block_ctxt, t: TypeRef, n: ValueRef) -> ValueRef {
    ret new_builder(cx.fcx.lldynamicallocas).ArrayAlloca(t, n);
}


// Creates a simpler, size-equivalent type. The resulting type is guaranteed
// to have (a) the same size as the type that was passed in; (b) to be non-
// recursive. This is done by replacing all boxes in a type with boxed unit
// types.
fn simplify_type(ccx: &@crate_ctxt, typ: &ty::t) -> ty::t {
    fn simplifier(ccx: @crate_ctxt, typ: ty::t) -> ty::t {
        alt ty::struct(ccx.tcx, typ) {
          ty::ty_box(_) { ret ty::mk_imm_box(ccx.tcx, ty::mk_nil(ccx.tcx)); }
          ty::ty_vec(_) { ret ty::mk_imm_vec(ccx.tcx, ty::mk_nil(ccx.tcx)); }
          ty::ty_fn(_, _, _, _, _) {
            ret ty::mk_imm_tup(ccx.tcx,
                               ~[ty::mk_imm_box(ccx.tcx, ty::mk_nil(ccx.tcx)),
                                 ty::mk_imm_box(ccx.tcx,
                                                ty::mk_nil(ccx.tcx))]);
          }
          ty::ty_obj(_) {
            ret ty::mk_imm_tup(ccx.tcx,
                               ~[ty::mk_imm_box(ccx.tcx, ty::mk_nil(ccx.tcx)),
                                 ty::mk_imm_box(ccx.tcx,
                                                ty::mk_nil(ccx.tcx))]);
          }
          ty::ty_res(_, sub, tps) {
            let sub1 = ty::substitute_type_params(ccx.tcx, tps, sub);
            ret ty::mk_imm_tup(ccx.tcx,
                               ~[ty::mk_int(ccx.tcx),
                                 simplify_type(ccx, sub1)]);
          }
          _ { ret typ; }
        }
    }
    ret ty::fold_ty(ccx.tcx, ty::fm_general(bind simplifier(ccx, _)), typ);
}


// Computes the size of the data part of a non-dynamically-sized tag.
fn static_size_of_tag(cx: &@crate_ctxt, sp: &span, t: &ty::t) -> uint {
    if ty::type_has_dynamic_size(cx.tcx, t) {
        cx.tcx.sess.span_fatal(sp,
                               "dynamically sized type passed to \
                               static_size_of_tag()");
    }
    if cx.tag_sizes.contains_key(t) { ret cx.tag_sizes.get(t); }
    alt ty::struct(cx.tcx, t) {
      ty::ty_tag(tid, subtys) {
        // Compute max(variant sizes).

        let max_size = 0u;
        let variants = ty::tag_variants(cx.tcx, tid);
        for variant: ty::variant_info  in variants {
            let tup_ty =
                simplify_type(cx, ty::mk_imm_tup(cx.tcx, variant.args));
            // Perform any type parameter substitutions.

            tup_ty = ty::substitute_type_params(cx.tcx, subtys, tup_ty);
            // Here we possibly do a recursive call.

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

fn dynamic_size_of(cx: &@block_ctxt, t: ty::t) -> result {
    fn align_elements(cx: &@block_ctxt, elts: &ty::t[]) -> result {
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
        for e: ty::t  in elts {
            let elt_align = align_of(bcx, e);
            bcx = elt_align.bcx;
            let elt_size = size_of(bcx, e);
            bcx = elt_size.bcx;
            let aligned_off = align_to(bcx, off, elt_align.val);
            off = bcx.build.Add(aligned_off, elt_size.val);
            max_align = umax(bcx, max_align, elt_align.val);
        }
        off = align_to(bcx, off, max_align);
        ret rslt(bcx, off);
    }
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_param(p,_) {
        let szptr = field_of_tydesc(cx, t, false, abi::tydesc_field_size);
        ret rslt(szptr.bcx, szptr.bcx.build.Load(szptr.val));
      }
      ty::ty_rec(flds) {
        let tys: ty::t[] = ~[];
        for f: ty::field  in flds { tys += ~[f.mt.ty]; }
        ret align_elements(cx, tys);
      }
      ty::ty_tag(tid, tps) {
        let bcx = cx;
        // Compute max(variant sizes).

        let max_size: ValueRef = alloca(bcx, T_int());
        bcx.build.Store(C_int(0), max_size);
        let variants = ty::tag_variants(bcx_tcx(bcx), tid);
        for variant: ty::variant_info  in variants {
            // Perform type substitution on the raw argument types.

            let raw_tys: ty::t[] = variant.args;
            let tys: ty::t[] = ~[];
            for raw_ty: ty::t  in raw_tys {
                let t = ty::substitute_type_params(bcx_tcx(cx), tps, raw_ty);
                tys += ~[t];
            }
            let rslt = align_elements(bcx, tys);
            bcx = rslt.bcx;
            let this_size = rslt.val;
            let old_max_size = bcx.build.Load(max_size);
            bcx.build.Store(umax(bcx, this_size, old_max_size), max_size);
        }
        let max_size_val = bcx.build.Load(max_size);
        let total_size =
            if std::ivec::len(variants) != 1u {
                bcx.build.Add(max_size_val, llsize_of(T_int()))
            } else { max_size_val };
        ret rslt(bcx, total_size);
      }
      ty::ty_ivec(mt) {
        let rs = size_of(cx, mt.ty);
        let bcx = rs.bcx;
        let llunitsz = rs.val;
        let llsz =
            bcx.build.Add(llsize_of(T_opaque_ivec()),
                          bcx.build.Mul(llunitsz,
                                        C_uint(abi::ivec_default_length)));
        ret rslt(bcx, llsz);
      }
    }
}

fn dynamic_align_of(cx: &@block_ctxt, t: &ty::t) -> result {
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_param(p,_) {
        let aptr = field_of_tydesc(cx, t, false, abi::tydesc_field_align);
        ret rslt(aptr.bcx, aptr.bcx.build.Load(aptr.val));
      }
      ty::ty_rec(flds) {
        let a = C_int(1);
        let bcx = cx;
        for f: ty::field  in flds {
            let align = align_of(bcx, f.mt.ty);
            bcx = align.bcx;
            a = umax(bcx, a, align.val);
        }
        ret rslt(bcx, a);
      }
      ty::ty_tag(_, _) {
        ret rslt(cx, C_int(1)); // FIXME: stub
      }
      ty::ty_ivec(tm) {
        let rs = align_of(cx, tm.ty);
        let bcx = rs.bcx;
        let llunitalign = rs.val;
        let llalign = umax(bcx, llalign_of(T_int()), llunitalign);
        ret rslt(bcx, llalign);
      }
    }
}

// Simple wrapper around GEP that takes an array of ints and wraps them
// in C_int()
fn GEPi(cx: &@block_ctxt, base: ValueRef, ixs: &int[]) -> ValueRef {
    let v: ValueRef[] = ~[];
    for i: int  in ixs { v += ~[C_int(i)]; }
    ret cx.build.GEP(base, v);
}

// Increment a pointer by a given amount and then cast it to be a pointer
// to a given type.
fn bump_ptr(bcx: &@block_ctxt, t: &ty::t, base: ValueRef, sz: ValueRef)
    -> ValueRef {
    let raw = bcx.build.PointerCast(base, T_ptr(T_i8()));
    let bumped = bcx.build.GEP(raw, ~[sz]);
    if ty::type_has_dynamic_size(bcx_tcx(bcx), t) {
        ret bumped;
    }
    let typ = T_ptr(type_of(bcx_ccx(bcx), bcx.sp, t));
    ret bcx.build.PointerCast(bumped, typ);
}

// Replacement for the LLVM 'GEP' instruction when field-indexing into a
// tuple-like structure (tup, rec) with a static index. This one is driven off
// ty::struct and knows what to do when it runs into a ty_param stuck in the
// middle of the thing it's GEP'ing into. Much like size_of and align_of,
// above.
fn GEP_tup_like(cx: &@block_ctxt, t: &ty::t, base: ValueRef, ixs: &int[]) ->
   result {
    assert (ty::type_is_tup_like(bcx_tcx(cx), t));
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

    fn split_type(ccx: &@crate_ctxt, t: &ty::t, ixs: &int[], n: uint) ->
       {prefix: ty::t[], target: ty::t} {
        let len: uint = std::ivec::len[int](ixs);
        // We don't support 0-index or 1-index GEPs: The former is nonsense
        // and the latter would only be meaningful if we supported non-0
        // values for the 0th index (we don't).

        assert (len > 1u);
        if n == 0u {
            // Since we're starting from a value that's a pointer to a
            // *single* structure, the first index (in GEP-ese) should just be
            // 0, to yield the pointee.

            assert (ixs.(n) == 0);
            ret split_type(ccx, t, ixs, n + 1u);
        }
        assert (n < len);
        let ix: int = ixs.(n);
        let prefix: ty::t[] = ~[];
        let i: int = 0;
        while i < ix {
            prefix += ~[ty::get_element_type(ccx.tcx, t, i as uint)];
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

    let args = ~[];
    for typ: ty::t  in s.prefix { args += ~[typ]; }
    let prefix_ty = ty::mk_imm_tup(bcx_tcx(cx), args);

    let bcx = cx;
    let sz = size_of(bcx, prefix_ty);
    ret rslt(sz.bcx, bump_ptr(sz.bcx, s.target, base, sz.val));
}


// Replacement for the LLVM 'GEP' instruction when field indexing into a tag.
// This function uses GEP_tup_like() above and automatically performs casts as
// appropriate. @llblobptr is the data part of a tag value; its actual type is
// meaningless, as it will be cast away.
fn GEP_tag(cx: @block_ctxt, llblobptr: ValueRef, tag_id: &ast::def_id,
           variant_id: &ast::def_id, ty_substs: &ty::t[], ix: int) -> result {
    let variant = ty::tag_variant_with_id(bcx_tcx(cx), tag_id, variant_id);
    // Synthesize a tuple type so that GEP_tup_like() can work its magic.
    // Separately, store the type of the element we're interested in.

    let arg_tys = variant.args;
    let elem_ty = ty::mk_nil(bcx_tcx(cx)); // typestate infelicity

    let i = 0;
    let true_arg_tys: ty::t[] = ~[];
    for aty: ty::t  in arg_tys {
        let arg_ty = ty::substitute_type_params(bcx_tcx(cx), ty_substs, aty);
        true_arg_tys += ~[arg_ty];
        if i == ix { elem_ty = arg_ty; }
        i += 1;
    }
    let tup_ty = ty::mk_imm_tup(bcx_tcx(cx), true_arg_tys);
    // Cast the blob pointer to the appropriate type, if we need to (i.e. if
    // the blob pointer isn't dynamically sized).

    let llunionptr: ValueRef;
    if !ty::type_has_dynamic_size(bcx_tcx(cx), tup_ty) {
        let llty = type_of(bcx_ccx(cx), cx.sp, tup_ty);
        llunionptr = cx.build.TruncOrBitCast(llblobptr, T_ptr(llty));
    } else { llunionptr = llblobptr; }
    // Do the GEP_tup_like().

    let rs = GEP_tup_like(cx, tup_ty, llunionptr, ~[0, ix]);
    // Cast the result to the appropriate type, if necessary.

    let val;
    if !ty::type_has_dynamic_size(bcx_tcx(cx), elem_ty) {
        let llelemty = type_of(bcx_ccx(rs.bcx), cx.sp, elem_ty);
        val = rs.bcx.build.PointerCast(rs.val, T_ptr(llelemty));
    } else { val = rs.val; }
    ret rslt(rs.bcx, val);
}

// trans_raw_malloc: expects a type indicating which pointer type we want and
// a size indicating how much space we want malloc'd.
fn trans_raw_malloc(cx: &@block_ctxt, llptr_ty: TypeRef, llsize: ValueRef) ->
   result {
    // FIXME: need a table to collect tydesc globals.

    let tydesc = C_null(T_ptr(bcx_ccx(cx).tydesc_type));
    let rval =
        cx.build.Call(bcx_ccx(cx).upcalls.malloc,
                      ~[cx.fcx.lltaskptr, llsize, tydesc]);
    ret rslt(cx, cx.build.PointerCast(rval, llptr_ty));
}

// trans_shared_malloc: expects a type indicating which pointer type we want
// and a size indicating how much space we want malloc'd.
fn trans_shared_malloc(cx: &@block_ctxt, llptr_ty: TypeRef, llsize: ValueRef)
   -> result {
    // FIXME: need a table to collect tydesc globals.

    let tydesc = C_null(T_ptr(bcx_ccx(cx).tydesc_type));
    let rval =
        cx.build.Call(bcx_ccx(cx).upcalls.shared_malloc,
                      ~[cx.fcx.lltaskptr, llsize, tydesc]);
    ret rslt(cx, cx.build.PointerCast(rval, llptr_ty));
}

// trans_malloc_boxed_raw: expects an unboxed type and returns a pointer to
// enough space for something of that type, along with space for a reference
// count; in other words, it allocates a box for something of that type.
fn trans_malloc_boxed_raw(cx: &@block_ctxt, t: ty::t) -> result {
    // Synthesize a fake box type structurally so we have something
    // to measure the size of.

    // We synthesize two types here because we want both the type of the
    // pointer and the pointee.  boxed_body is the type that we measure the
    // size of; box_ptr is the type that's converted to a TypeRef and used as
    // the pointer cast target in trans_raw_malloc.

    // The mk_int here is the space being
    // reserved for the refcount.
    let boxed_body =
        ty::mk_imm_tup(bcx_tcx(cx), ~[ty::mk_int(bcx_tcx(cx)), t]);
    let box_ptr = ty::mk_imm_box(bcx_tcx(cx), t);
    let sz = size_of(cx, boxed_body);

    // Grab the TypeRef type of box_ptr, because that's what trans_raw_malloc
    // wants.
    let llty = type_of(bcx_ccx(cx), cx.sp, box_ptr);
    ret trans_raw_malloc(sz.bcx, llty, sz.val);
}

// trans_malloc_boxed: usefully wraps trans_malloc_box_raw; allocates a box,
// initializes the reference count to 1, and pulls out the body and rc
fn trans_malloc_boxed(cx: &@block_ctxt, t: ty::t) ->
    {bcx: @block_ctxt, box: ValueRef, body: ValueRef} {
    let res = trans_malloc_boxed_raw(cx, t);
    let box = res.val;
    let rc = GEPi(res.bcx, box, ~[0, abi::box_rc_field_refcnt]);
    res.bcx.build.Store(C_int(1), rc);
    let body = GEPi(res.bcx, box, ~[0, abi::box_rc_field_body]);
    ret {bcx: res.bcx, box: res.val, body: body};
}

// Type descriptor and type glue stuff

// Given a type and a field index into its corresponding type descriptor,
// returns an LLVM ValueRef of that field from the tydesc, generating the
// tydesc if necessary.
fn field_of_tydesc(cx: &@block_ctxt, t: &ty::t, escapes: bool, field: int) ->
   result {
    let ti = none[@tydesc_info];
    let tydesc = get_tydesc(cx, t, escapes, ti);
    ret rslt(tydesc.bcx,
             tydesc.bcx.build.GEP(tydesc.val, ~[C_int(0), C_int(field)]));
}


// Given a type containing ty params, build a vector containing a ValueRef for
// each of the ty params it uses (from the current frame) and a vector of the
// indices of the ty params present in the type. This is used solely for
// constructing derived tydescs.
fn linearize_ty_params(cx: &@block_ctxt, t: &ty::t) ->
   {params: uint[], descs: ValueRef[]} {
    let param_vals: ValueRef[] = ~[];
    let param_defs: uint[] = ~[];
    type rr =
        {cx: @block_ctxt, mutable vals: ValueRef[], mutable defs: uint[]};

    fn linearizer(r: @rr, t: ty::t) {
        alt ty::struct(bcx_tcx(r.cx), t) {
          ty::ty_param(pid,_) {
            let seen: bool = false;
            for d: uint  in r.defs { if d == pid { seen = true; } }
            if !seen {
                r.vals += ~[r.cx.fcx.lltydescs.(pid)];
                r.defs += ~[pid];
            }
          }
          _ { }
        }
    }
    let x = @{cx: cx, mutable vals: param_vals, mutable defs: param_defs};
    let f = bind linearizer(x, _);
    ty::walk_ty(bcx_tcx(cx), f, t);
    ret {params: x.defs, descs: x.vals};
}

fn trans_stack_local_derived_tydesc(cx: &@block_ctxt, llsz: ValueRef,
                                    llalign: ValueRef, llroottydesc: ValueRef,
                                    llparamtydescs: ValueRef) -> ValueRef {
    let llmyroottydesc = alloca(cx, bcx_ccx(cx).tydesc_type);
    // By convention, desc 0 is the root descriptor.

    llroottydesc = cx.build.Load(llroottydesc);
    cx.build.Store(llroottydesc, llmyroottydesc);
    // Store a pointer to the rest of the descriptors.

    let llfirstparam = cx.build.GEP(llparamtydescs, ~[C_int(0), C_int(0)]);
    cx.build.Store(llfirstparam,
                   cx.build.GEP(llmyroottydesc, ~[C_int(0), C_int(0)]));
    cx.build.Store(llsz, cx.build.GEP(llmyroottydesc, ~[C_int(0), C_int(1)]));
    cx.build.Store(llalign,
                   cx.build.GEP(llmyroottydesc, ~[C_int(0), C_int(2)]));
    ret llmyroottydesc;
}

fn get_derived_tydesc(cx: &@block_ctxt, t: &ty::t, escapes: bool,
                      static_ti: &mutable option::t[@tydesc_info]) -> result {
    alt cx.fcx.derived_tydescs.find(t) {
      some(info) {


        // If the tydesc escapes in this context, the cached derived
        // tydesc also has to be one that was marked as escaping.
        if !(escapes && !info.escapes) { ret rslt(cx, info.lltydesc); }
      }
      none. {/* fall through */ }
    }
    bcx_ccx(cx).stats.n_derived_tydescs += 1u;
    let bcx = new_raw_block_ctxt(cx.fcx, cx.fcx.llderivedtydescs);
    let n_params: uint = ty::count_ty_params(bcx_tcx(bcx), t);
    let tys = linearize_ty_params(bcx, t);
    assert (n_params == std::ivec::len[uint](tys.params));
    assert (n_params == std::ivec::len[ValueRef](tys.descs));
    let root_ti = get_static_tydesc(bcx, t, tys.params);
    static_ti = some[@tydesc_info](root_ti);
    lazily_emit_all_tydesc_glue(cx, static_ti);
    let root = root_ti.tydesc;
    let sz = size_of(bcx, t);
    bcx = sz.bcx;
    let align = align_of(bcx, t);
    bcx = align.bcx;
    let v;
    if escapes {
        /* for root*/
        let tydescs =
            alloca(bcx,
                   T_array(T_ptr(bcx_ccx(bcx).tydesc_type), 1u + n_params));
        let i = 0;
        let tdp = bcx.build.GEP(tydescs, ~[C_int(0), C_int(i)]);
        bcx.build.Store(root, tdp);
        i += 1;
        for td: ValueRef  in tys.descs {
            let tdp = bcx.build.GEP(tydescs, ~[C_int(0), C_int(i)]);
            bcx.build.Store(td, tdp);
            i += 1;
        }
        let lltydescsptr =
            bcx.build.PointerCast(tydescs,
                                  T_ptr(T_ptr(bcx_ccx(bcx).tydesc_type)));
        let td_val =
            bcx.build.Call(bcx_ccx(bcx).upcalls.get_type_desc,
                           ~[bcx.fcx.lltaskptr, C_null(T_ptr(T_nil())),
                             sz.val, align.val, C_int(1u + n_params as int),
                             lltydescsptr]);
        v = td_val;
    } else {
        let llparamtydescs =
            alloca(bcx, T_array(T_ptr(bcx_ccx(bcx).tydesc_type), n_params));
        let i = 0;
        for td: ValueRef  in tys.descs {
            let tdp = bcx.build.GEP(llparamtydescs, ~[C_int(0), C_int(i)]);
            bcx.build.Store(td, tdp);
            i += 1;
        }
        v =
            trans_stack_local_derived_tydesc(bcx, sz.val, align.val, root,
                                             llparamtydescs);
    }
    bcx.fcx.derived_tydescs.insert(t, {lltydesc: v, escapes: escapes});
    ret rslt(cx, v);
}

fn get_tydesc(cx: &@block_ctxt, orig_t: &ty::t, escapes: bool,
              static_ti: &mutable option::t[@tydesc_info]) -> result {

    let t = ty::strip_cname(bcx_tcx(cx), orig_t);

    // Is the supplied type a type param? If so, return the passed-in tydesc.
    alt ty::type_param(bcx_tcx(cx), t) {
      some(id) { ret rslt(cx, cx.fcx.lltydescs.(id)); }
      none. {/* fall through */ }
    }

    // Does it contain a type param? If so, generate a derived tydesc.
    if ty::type_contains_params(bcx_tcx(cx), t) {
        ret get_derived_tydesc(cx, t, escapes, static_ti);
    }

    // Otherwise, generate a tydesc if necessary, and return it.
    let info = get_static_tydesc(cx, t, ~[]);
    static_ti = some[@tydesc_info](info);
    ret rslt(cx, info.tydesc);
}

fn get_static_tydesc(cx: &@block_ctxt, orig_t: &ty::t, ty_params: &uint[]) ->
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

fn set_glue_inlining(cx: &@local_ctxt, f: ValueRef, t: &ty::t) {
    if ty::type_is_structural(cx.ccx.tcx, t) {
        set_no_inline(f);
    } else { set_always_inline(f); }
}


// Generates the declaration for (but doesn't emit) a type descriptor.
fn declare_tydesc(cx: &@local_ctxt, sp: &span, t: &ty::t, ty_params: &uint[])
   -> @tydesc_info {
    log "+++ declare_tydesc " + ty_to_str(cx.ccx.tcx, t);
    let ccx = cx.ccx;
    let llsize;
    let llalign;
    if !ty::type_has_dynamic_size(ccx.tcx, t) {
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
        llvm::LLVMAddGlobal(ccx.llmod, ccx.tydesc_type, str::buf(name));
    let info =
        @{ty: t,
          tydesc: gvar,
          size: llsize,
          align: llalign,
          mutable copy_glue: none[ValueRef],
          mutable drop_glue: none[ValueRef],
          mutable free_glue: none[ValueRef],
          mutable cmp_glue: none[ValueRef],
          ty_params: ty_params};
    log "--- declare_tydesc " + ty_to_str(cx.ccx.tcx, t);
    ret info;
}

tag make_generic_glue_helper_fn {
    mgghf_single(fn(&@block_ctxt, ValueRef, &ty::t) );
    mgghf_cmp;
}

fn declare_generic_glue(cx: &@local_ctxt, t: &ty::t, llfnty: TypeRef,
                        name: &str) -> ValueRef {
    let fn_nm;
    if cx.ccx.sess.get_opts().debuginfo {
        fn_nm = mangle_internal_name_by_type_only(cx.ccx, t, "glue_" + name);
        fn_nm = sanitize(fn_nm);
    } else { fn_nm = mangle_internal_name_by_seq(cx.ccx, "glue_" + name); }
    let llfn = decl_cdecl_fn(cx.ccx.llmod, fn_nm, llfnty);
    set_glue_inlining(cx, llfn, t);
    ret llfn;
}

fn make_generic_glue_inner(cx: &@local_ctxt, sp: &span, t: &ty::t,
                           llfn: ValueRef,
                           helper: &make_generic_glue_helper_fn,
                           ty_params: &uint[]) -> ValueRef {
    let fcx = new_fn_ctxt(cx, sp, llfn);
    llvm::LLVMSetLinkage(llfn,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    cx.ccx.stats.n_glues_created += 1u;
    // Any nontrivial glue is with values passed *by alias*; this is a
    // requirement since in many contexts glue is invoked indirectly and
    // the caller has no idea if it's dealing with something that can be
    // passed by value.

    let llty;
    if ty::type_has_dynamic_size(cx.ccx.tcx, t) {
        llty = T_ptr(T_i8());
    } else { llty = T_ptr(type_of(cx.ccx, sp, t)); }
    let ty_param_count = std::ivec::len[uint](ty_params);
    let lltyparams = llvm::LLVMGetParam(llfn, 3u);
    let copy_args_bcx = new_raw_block_ctxt(fcx, fcx.llcopyargs);
    let lltydescs = ~[mutable ];
    let p = 0u;
    while p < ty_param_count {
        let llparam = copy_args_bcx.build.GEP(lltyparams, ~[C_int(p as int)]);
        llparam = copy_args_bcx.build.Load(llparam);
        std::ivec::grow_set(lltydescs, ty_params.(p), 0 as ValueRef, llparam);
        p += 1u;
    }

    // TODO: Implement some kind of freeze operation in the standard library.
    let lltydescs_frozen = ~[];
    for lltydesc: ValueRef  in lltydescs { lltydescs_frozen += ~[lltydesc]; }
    fcx.lltydescs = lltydescs_frozen;

    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;
    let llrawptr0 = llvm::LLVMGetParam(llfn, 4u);
    let llval0 = bcx.build.BitCast(llrawptr0, llty);
    alt helper {
      mgghf_single(single_fn) { single_fn(bcx, llval0, t); }
      mgghf_cmp. {
        let llrawptr1 = llvm::LLVMGetParam(llfn, 5u);
        let llval1 = bcx.build.BitCast(llrawptr1, llty);
        let llcmpval = llvm::LLVMGetParam(llfn, 6u);
        make_cmp_glue(bcx, llval0, llval1, t, llcmpval);
      }
    }
    finish_fn(fcx, lltop);
    ret llfn;
}

fn make_generic_glue(cx: &@local_ctxt, sp: &span, t: &ty::t, llfn: ValueRef,
                     helper: &make_generic_glue_helper_fn, ty_params: &uint[],
                     name: &str) -> ValueRef {
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

fn emit_tydescs(ccx: &@crate_ctxt) {
    for each pair: @{key: ty::t, val: @tydesc_info}  in ccx.tydescs.items() {
        let glue_fn_ty = T_ptr(T_glue_fn(*ccx));
        let cmp_fn_ty = T_ptr(T_cmp_glue_fn(*ccx));
        let ti = pair.val;
        let copy_glue =
            alt { ti.copy_glue } {
              none. { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let drop_glue =
            alt { ti.drop_glue } {
              none. { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let free_glue =
            alt { ti.free_glue } {
              none. { ccx.stats.n_null_glues += 1u; C_null(glue_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let cmp_glue =
            alt { ti.cmp_glue } {
              none. { ccx.stats.n_null_glues += 1u; C_null(cmp_fn_ty) }
              some(v) { ccx.stats.n_real_glues += 1u; v }
            };
        let  // copy_glue
             // drop_glue
             // free_glue
             // sever_glue
             // mark_glue
             // obj_drop_glue
             // is_stateful
            tydesc =
            C_named_struct(ccx.tydesc_type,
                           ~[C_null(T_ptr(T_ptr(ccx.tydesc_type))), ti.size,
                             ti.align, copy_glue, drop_glue, free_glue,
                             C_null(glue_fn_ty), C_null(glue_fn_ty),
                             C_null(glue_fn_ty), C_null(glue_fn_ty),
                             cmp_glue]); // cmp_glue

        let gvar = ti.tydesc;
        llvm::LLVMSetInitializer(gvar, tydesc);
        llvm::LLVMSetGlobalConstant(gvar, True);
        llvm::LLVMSetLinkage(gvar,
                             lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    }
}

fn make_copy_glue(cx: &@block_ctxt, v: ValueRef, t: &ty::t) {
    // NB: v is an *alias* of type t here, not a direct value.

    let bcx;

    if ty::type_is_task(bcx_tcx(cx), t) {
        let task_ptr = cx.build.Load(v);
        cx.build.Call(bcx_ccx(cx).upcalls.take_task,
                      ~[cx.fcx.lltaskptr, task_ptr]);
        bcx = cx;
    } else if ty::type_is_boxed(bcx_tcx(cx), t) {
        bcx = incr_refcnt_of_boxed(cx, cx.build.Load(v)).bcx;
    } else if (ty::type_is_structural(bcx_tcx(cx), t)) {
        bcx = duplicate_heap_parts_if_necessary(cx, v, t).bcx;
        bcx = iter_structural_ty(bcx, v, t, bind copy_ty(_, _, _)).bcx;
    } else { bcx = cx; }
    bcx.build.RetVoid();
}

fn incr_refcnt_of_boxed(cx: &@block_ctxt, box_ptr: ValueRef) -> result {
    let rc_ptr =
        cx.build.GEP(box_ptr, ~[C_int(0), C_int(abi::box_rc_field_refcnt)]);
    let rc = cx.build.Load(rc_ptr);
    let rc_adj_cx = new_sub_block_ctxt(cx, "rc++");
    let next_cx = new_sub_block_ctxt(cx, "next");
    let const_test =
        cx.build.ICmp(lib::llvm::LLVMIntEQ, C_int(abi::const_refcount as int),
                      rc);
    cx.build.CondBr(const_test, next_cx.llbb, rc_adj_cx.llbb);
    rc = rc_adj_cx.build.Add(rc, C_int(1));
    rc_adj_cx.build.Store(rc, rc_ptr);
    rc_adj_cx.build.Br(next_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn make_free_glue(cx: &@block_ctxt, v0: ValueRef, t: &ty::t) {
    // NB: v is an *alias* of type t here, not a direct value.

    // FIXME: switch gc/non-gc on layer of the type.
    // FIXME: switch gc/non-gc on layer of the type.
    // TODO: call upcall_kill


    // Call through the obj's own fields-drop glue first.

    // Then free the body.
    // FIXME: switch gc/non-gc on layer of the type.
    // Call through the closure's own fields-drop glue first.

    // Then free the body.
    // FIXME: switch gc/non-gc on layer of the type.
    let rs =
        alt ty::struct(bcx_tcx(cx), t) {
          ty::ty_str. { let v = cx.build.Load(v0); trans_non_gc_free(cx, v) }
          ty::ty_vec(_) {
            let v = cx.build.Load(v0);
            let rs = iter_sequence(cx, v, t, bind drop_ty(_, _, _));
            trans_non_gc_free(rs.bcx, v)
          }
          ty::ty_box(body_mt) {
            let v = cx.build.Load(v0);
            let body =
                cx.build.GEP(v, ~[C_int(0), C_int(abi::box_rc_field_body)]);
            let body_ty = body_mt.ty;
            let body_val = load_if_immediate(cx, body, body_ty);
            let rs = drop_ty(cx, body_val, body_ty);
            trans_non_gc_free(rs.bcx, v)
          }
          ty::ty_port(_) {
            let v = cx.build.Load(v0);
            cx.build.Call(bcx_ccx(cx).upcalls.del_port,
                          ~[cx.fcx.lltaskptr,
                            cx.build.PointerCast(v, T_opaque_port_ptr())]);
            rslt(cx, C_int(0))
          }
          ty::ty_chan(_) {
            let v = cx.build.Load(v0);
            cx.build.Call(bcx_ccx(cx).upcalls.del_chan,
                          ~[cx.fcx.lltaskptr,
                            cx.build.PointerCast(v, T_opaque_chan_ptr())]);
            rslt(cx, C_int(0))
          }
          ty::ty_task. { rslt(cx, C_nil()) }
          ty::ty_obj(_) {
            let box_cell =
                cx.build.GEP(v0, ~[C_int(0), C_int(abi::obj_field_box)]);
            let b = cx.build.Load(box_cell);
            let ccx = bcx_ccx(cx);
            let llbox_ty = T_opaque_obj_ptr(*ccx);
            b = cx.build.PointerCast(b, llbox_ty);
            let body =
                cx.build.GEP(b, ~[C_int(0), C_int(abi::box_rc_field_body)]);
            let tydescptr =
                cx.build.GEP(body,
                             ~[C_int(0), C_int(abi::obj_body_elt_tydesc)]);
            let tydesc = cx.build.Load(tydescptr);
            let ti = none[@tydesc_info];
            call_tydesc_glue_full(cx, body, tydesc,
                                  abi::tydesc_field_drop_glue, ti);
            trans_non_gc_free(cx, b)
          }
          ty::ty_fn(_, _, _, _, _) {
            let box_cell =
                cx.build.GEP(v0, ~[C_int(0), C_int(abi::fn_field_box)]);
            let v = cx.build.Load(box_cell);
            let body =
                cx.build.GEP(v, ~[C_int(0), C_int(abi::box_rc_field_body)]);
            let bindings =
                cx.build.GEP(body,
                             ~[C_int(0), C_int(abi::closure_elt_bindings)]);
            let tydescptr =
                cx.build.GEP(body,
                             ~[C_int(0), C_int(abi::closure_elt_tydesc)]);
            let ti = none[@tydesc_info];
            call_tydesc_glue_full(cx, bindings, cx.build.Load(tydescptr),
                                  abi::tydesc_field_drop_glue, ti);
            trans_non_gc_free(cx, v)
          }
          _ { rslt(cx, C_nil()) }
        };
    rs.bcx.build.RetVoid();
}

fn maybe_free_ivec_heap_part(cx: &@block_ctxt, v0: ValueRef, unit_ty: ty::t)
   -> result {
    let llunitty = type_of_or_i8(cx, unit_ty);
    let stack_len =
        cx.build.Load(cx.build.InBoundsGEP(v0,
                                           ~[C_int(0),
                                             C_uint(abi::ivec_elt_len)]));
    let maybe_on_heap_cx = new_sub_block_ctxt(cx, "maybe_on_heap");
    let next_cx = new_sub_block_ctxt(cx, "next");
    let maybe_on_heap =
        cx.build.ICmp(lib::llvm::LLVMIntEQ, stack_len, C_int(0));
    cx.build.CondBr(maybe_on_heap, maybe_on_heap_cx.llbb, next_cx.llbb);
    // Might be on the heap. Load the heap pointer and free it. (It's ok to
    // free a null pointer.)

    let stub_ptr =
        maybe_on_heap_cx.build.PointerCast(v0, T_ptr(T_ivec_heap(llunitty)));
    let heap_ptr =
        {
            let v = ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];
            let m = maybe_on_heap_cx.build.InBoundsGEP(stub_ptr, v);
            maybe_on_heap_cx.build.Load(m)
        };
    let after_free_cx = trans_shared_free(maybe_on_heap_cx, heap_ptr).bcx;
    after_free_cx.build.Br(next_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn make_drop_glue(cx: &@block_ctxt, v0: ValueRef, t: &ty::t) {
    // NB: v0 is an *alias* of type t here, not a direct value.
    let ccx = bcx_ccx(cx);
    let rs =
        alt ty::struct(ccx.tcx, t) {
          ty::ty_str. { decr_refcnt_maybe_free(cx, v0, v0, t) }
          ty::ty_vec(_) { decr_refcnt_maybe_free(cx, v0, v0, t) }
          ty::ty_ivec(tm) {
            let v1;
            if ty::type_has_dynamic_size(ccx.tcx, tm.ty) {
                v1 = cx.build.PointerCast(v0, T_ptr(T_opaque_ivec()));
            } else { v1 = v0; }
            let rslt = iter_structural_ty(cx, v1, t, drop_ty);
            maybe_free_ivec_heap_part(rslt.bcx, v1, tm.ty)
          }
          ty::ty_box(_) { decr_refcnt_maybe_free(cx, v0, v0, t) }
          ty::ty_port(_) { decr_refcnt_maybe_free(cx, v0, v0, t) }
          ty::ty_chan(_) { decr_refcnt_maybe_free(cx, v0, v0, t) }
          ty::ty_task. {
            let task_ptr = cx.build.Load(v0);
            {bcx: cx,
             val: cx.build.Call(bcx_ccx(cx).upcalls.drop_task,
                                ~[cx.fcx.lltaskptr, task_ptr])}
          }
          ty::ty_obj(_) {
            let box_cell =
                cx.build.GEP(v0, ~[C_int(0), C_int(abi::obj_field_box)]);
            decr_refcnt_maybe_free(cx, box_cell, v0, t)
          }
          ty::ty_res(did, inner, tps) {
            trans_res_drop(cx, v0, did, inner, tps)
          }
          ty::ty_fn(_, _, _, _, _) {
            let box_cell =
                cx.build.GEP(v0, ~[C_int(0), C_int(abi::fn_field_box)]);
            decr_refcnt_maybe_free(cx, box_cell, v0, t)
          }
          _ {
            if ty::type_has_pointers(ccx.tcx, t) &&
                   ty::type_is_structural(ccx.tcx, t) {
                iter_structural_ty(cx, v0, t, bind drop_ty(_, _, _))
            } else { rslt(cx, C_nil()) }
          }
        };
    rs.bcx.build.RetVoid();
}

fn trans_res_drop(cx: @block_ctxt, rs: ValueRef, did: &ast::def_id,
                  inner_t: ty::t, tps: &ty::t[]) -> result {
    let ccx = bcx_ccx(cx);
    let inner_t_s = ty::substitute_type_params(ccx.tcx, tps, inner_t);
    let tup_ty = ty::mk_imm_tup(ccx.tcx, ~[ty::mk_int(ccx.tcx), inner_t_s]);
    let drop_cx = new_sub_block_ctxt(cx, "drop res");
    let next_cx = new_sub_block_ctxt(cx, "next");

    let drop_flag = GEP_tup_like(cx, tup_ty, rs, ~[0, 0]);
    cx = drop_flag.bcx;
    let null_test = cx.build.IsNull(cx.build.Load(drop_flag.val));
    cx.build.CondBr(null_test, next_cx.llbb, drop_cx.llbb);
    cx = drop_cx;

    let val = GEP_tup_like(cx, tup_ty, rs, ~[0, 1]);
    cx = val.bcx;
    // Find and call the actual destructor.
    let dtor_pair =
        if did.crate == ast::local_crate {
            alt ccx.fn_pairs.find(did.node) {
              some(x) { x }
              _ { ccx.tcx.sess.bug("internal error in trans_res_drop") }
            }
        } else {
            let params =
                csearch::get_type_param_count(ccx.sess.get_cstore(), did);
            let f_t =
                type_of_fn(ccx, cx.sp, ast::proto_fn,
                           ~[{mode: ty::mo_alias(false), ty: inner_t}],
                           ty::mk_nil(ccx.tcx), params);
            get_extern_const(ccx.externs, ccx.llmod,
                             csearch::get_symbol(ccx.sess.get_cstore(), did),
                             T_fn_pair(*ccx, f_t))
        };
    let dtor_addr =
        cx.build.Load(cx.build.GEP(dtor_pair,
                                   ~[C_int(0), C_int(abi::fn_field_code)]));
    let dtor_env =
        cx.build.Load(cx.build.GEP(dtor_pair,
                                   ~[C_int(0), C_int(abi::fn_field_box)]));
    let args = ~[cx.fcx.llretptr, cx.fcx.lltaskptr, dtor_env];
    for tp: ty::t  in tps {
        let ti: option::t[@tydesc_info] = none;
        let td = get_tydesc(cx, tp, false, ti);
        args += ~[td.val];
        cx = td.bcx;
    }
    // Kludge to work around the fact that we know the precise type of the
    // value here, but the dtor expects a type that still has opaque pointers
    // for type variables.
    let val_llty =
        lib::llvm::fn_ty_param_tys(llvm::LLVMGetElementType
                                   (llvm::LLVMTypeOf(dtor_addr)))
                                    .(std::ivec::len(args));
    let val_cast = cx.build.BitCast(val.val, val_llty);
    cx.build.FastCall(dtor_addr, args + ~[val_cast]);

    cx = drop_slot(cx, val.val, inner_t_s).bcx;
    cx.build.Store(C_int(0), drop_flag.val);
    cx.build.Br(next_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn decr_refcnt_maybe_free(cx: &@block_ctxt, box_ptr_alias: ValueRef,
                          full_alias: ValueRef, t: &ty::t) -> result {
    let ccx = bcx_ccx(cx);
    let load_rc_cx = new_sub_block_ctxt(cx, "load rc");
    let rc_adj_cx = new_sub_block_ctxt(cx, "rc--");
    let free_cx = new_sub_block_ctxt(cx, "free");
    let next_cx = new_sub_block_ctxt(cx, "next");
    let box_ptr = cx.build.Load(box_ptr_alias);
    let llbox_ty = T_opaque_obj_ptr(*ccx);
    box_ptr = cx.build.PointerCast(box_ptr, llbox_ty);
    let null_test = cx.build.IsNull(box_ptr);
    cx.build.CondBr(null_test, next_cx.llbb, load_rc_cx.llbb);
    let rc_ptr =
        load_rc_cx.build.GEP(box_ptr,
                             ~[C_int(0), C_int(abi::box_rc_field_refcnt)]);
    let rc = load_rc_cx.build.Load(rc_ptr);
    let const_test =
        load_rc_cx.build.ICmp(lib::llvm::LLVMIntEQ,
                              C_int(abi::const_refcount as int), rc);
    load_rc_cx.build.CondBr(const_test, next_cx.llbb, rc_adj_cx.llbb);
    rc = rc_adj_cx.build.Sub(rc, C_int(1));
    rc_adj_cx.build.Store(rc, rc_ptr);
    let zero_test = rc_adj_cx.build.ICmp(lib::llvm::LLVMIntEQ, C_int(0), rc);
    rc_adj_cx.build.CondBr(zero_test, free_cx.llbb, next_cx.llbb);
    let free_res =
        free_ty(free_cx, load_if_immediate(free_cx, full_alias, t), t);
    free_res.bcx.build.Br(next_cx.llbb);
    let t_else = T_nil();
    let v_else = C_nil();
    let phi =
        next_cx.build.Phi(t_else, ~[v_else, v_else, v_else, free_res.val],
                          ~[cx.llbb, load_rc_cx.llbb, rc_adj_cx.llbb,
                            free_res.bcx.llbb]);
    ret rslt(next_cx, phi);
}


// Structural comparison: a rather involved form of glue.
fn maybe_name_value(cx: &@crate_ctxt, v: ValueRef, s: &str) {
    if cx.sess.get_opts().save_temps {
        llvm::LLVMSetValueName(v, str::buf(s));
    }
}

fn make_cmp_glue(cx: &@block_ctxt, lhs0: ValueRef, rhs0: ValueRef, t: &ty::t,
                 llop: ValueRef) {
    let lhs = load_if_immediate(cx, lhs0, t);
    let rhs = load_if_immediate(cx, rhs0, t);
    if ty::type_is_scalar(bcx_tcx(cx), t) {
        make_scalar_cmp_glue(cx, lhs, rhs, t, llop);
    } else if (ty::type_is_box(bcx_tcx(cx), t)) {
        lhs = cx.build.GEP(lhs, ~[C_int(0), C_int(abi::box_rc_field_body)]);
        rhs = cx.build.GEP(rhs, ~[C_int(0), C_int(abi::box_rc_field_body)]);
        let t_inner =
            alt ty::struct(bcx_tcx(cx), t) { ty::ty_box(ti) { ti.ty } };
        let rslt = compare(cx, lhs, rhs, t_inner, llop);
        rslt.bcx.build.Store(rslt.val, cx.fcx.llretptr);
        rslt.bcx.build.RetVoid();
    } else if (ty::type_is_structural(bcx_tcx(cx), t) ||
                   ty::type_is_sequence(bcx_tcx(cx), t)) {
        let scx = new_sub_block_ctxt(cx, "structural compare start");
        let next = new_sub_block_ctxt(cx, "structural compare end");
        cx.build.Br(scx.llbb);
        /*
         * We're doing lexicographic comparison here. We start with the
         * assumption that the two input elements are equal. Depending on
         * operator, this means that the result is either true or false;
         * equality produces 'true' for ==, <= and >=. It produces 'false' for
         * !=, < and >.
         *
         * We then move one element at a time through the structure checking
         * for pairwise element equality: If we have equality, our assumption
         * about overall sequence equality is not modified, so we have to move
         * to the next element.
         *
         * If we do not have pairwise element equality, we have reached an
         * element that 'decides' the lexicographic comparison. So we exit the
         * loop with a flag that indicates the true/false sense of that
         * decision, by testing the element again with the operator we're
         * interested in.
         *
         * When we're lucky, LLVM should be able to fold some of these two
         * tests together (as they're applied to the same operands and in some
         * cases are sometimes redundant). But we don't bother trying to
         * optimize combinations like that, at this level.
         */

        let flag = alloca(scx, T_i1());
        maybe_name_value(bcx_ccx(cx), flag, "flag");
        let r;
        if ty::type_is_sequence(bcx_tcx(cx), t) {
            // If we hit == all the way through the minimum-shared-length
            // section, default to judging the relative sequence lengths.

            let lhs_fill;
            let rhs_fill;
            let bcx;
            if ty::sequence_is_interior(bcx_tcx(cx), t) {
                let st = ty::sequence_element_type(bcx_tcx(cx), t);
                let lad = ivec::get_len_and_data(scx, lhs, st);
                bcx = lad.bcx;
                lhs_fill = lad.len;
                lad = ivec::get_len_and_data(bcx, rhs, st);
                bcx = lad.bcx;
                rhs_fill = lad.len;
            } else {
                lhs_fill = vec_fill(scx, lhs);
                rhs_fill = vec_fill(scx, rhs);
                bcx = scx;
            }
            r =
                compare_scalar_values(bcx, lhs_fill, rhs_fill, unsigned_int,
                                      llop);
            r.bcx.build.Store(r.val, flag);
        } else {
            // == and <= default to true if they find == all the way. <
            // defaults to false if it finds == all the way.

            let result_if_equal =
                scx.build.ICmp(lib::llvm::LLVMIntNE, llop,
                               C_u8(abi::cmp_glue_op_lt));
            scx.build.Store(result_if_equal, flag);
            r = rslt(scx, C_nil());
        }
        fn inner(last_cx: @block_ctxt, load_inner: bool, flag: ValueRef,
                 llop: ValueRef, cx: &@block_ctxt, av0: ValueRef,
                 bv0: ValueRef, t: ty::t) -> result {
            let cnt_cx = new_sub_block_ctxt(cx, "continue_comparison");
            let stop_cx = new_sub_block_ctxt(cx, "stop_comparison");
            let av = av0;
            let bv = bv0;
            if load_inner {
                // If `load_inner` is true, then the pointer type will always
                // be i8, because the data part of a vector always has type
                // i8[]. So we need to cast it to the proper type.

                if !ty::type_has_dynamic_size(bcx_tcx(last_cx), t) {
                    let llelemty =
                        T_ptr(type_of(bcx_ccx(last_cx), last_cx.sp, t));
                    av = cx.build.PointerCast(av, llelemty);
                    bv = cx.build.PointerCast(bv, llelemty);
                }
                av = load_if_immediate(cx, av, t);
                bv = load_if_immediate(cx, bv, t);
            }

            // First 'eq' comparison: if so, continue to next elts.
            let eq_r = compare(cx, av, bv, t, C_u8(abi::cmp_glue_op_eq));
            eq_r.bcx.build.CondBr(eq_r.val, cnt_cx.llbb, stop_cx.llbb);

            // Second 'op' comparison: find out how this elt-pair decides.
            let stop_r = compare(stop_cx, av, bv, t, llop);
            stop_r.bcx.build.Store(stop_r.val, flag);
            stop_r.bcx.build.Br(last_cx.llbb);
            ret rslt(cnt_cx, C_nil());
        }
        if ty::type_is_structural(bcx_tcx(cx), t) {
            r = iter_structural_ty_full(r.bcx, lhs, rhs, t,
                                        bind inner(next, false, flag, llop, _,
                                                   _, _, _));
        } else {
            let lhs_p0 = vec_p0(r.bcx, lhs);
            let rhs_p0 = vec_p0(r.bcx, rhs);
            let min_len =
                umin(r.bcx, vec_fill(r.bcx, lhs), vec_fill(r.bcx, rhs));
            let rhs_lim = r.bcx.build.GEP(rhs_p0, ~[min_len]);
            let elt_ty = ty::sequence_element_type(bcx_tcx(cx), t);
            r = size_of(r.bcx, elt_ty);
            r = iter_sequence_raw(r.bcx, lhs_p0, rhs_p0, rhs_lim, r.val,
                                  bind inner(next, true, flag, llop, _, _, _,
                                             elt_ty));
        }
        r.bcx.build.Br(next.llbb);
        let v = next.build.Load(flag);
        next.build.Store(v, cx.fcx.llretptr);
        next.build.RetVoid();
    } else {
        // FIXME: compare obj, fn by pointer?

        trans_fail(cx, none[span],
                   "attempt to compare values of type " +
                       ty_to_str(bcx_tcx(cx), t));
    }
}


// Used only for creating scalar comparison glue.
tag scalar_type { nil_type; signed_int; unsigned_int; floating_point; }


fn compare_scalar_types(cx: @block_ctxt, lhs: ValueRef, rhs: ValueRef,
                        t: &ty::t, llop: ValueRef) -> result {
    let f = bind compare_scalar_values(cx, lhs, rhs, _, llop);

    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_nil. { ret f(nil_type); }
      ty::ty_bool. | ty::ty_uint. | ty::ty_ptr(_) |
      ty::ty_char. { ret f(unsigned_int); }
      ty::ty_int. { ret f(signed_int); }
      ty::ty_float. { ret f(floating_point); }
      ty::ty_machine(_) {
        if ty::type_is_fp(bcx_tcx(cx), t) {
            // Floating point machine types
            ret f(floating_point);
        } else if (ty::type_is_signed(bcx_tcx(cx), t)) {
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
        trans_fail(cx, none[span],
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

// A helper function to create scalar comparison glue.
fn make_scalar_cmp_glue(cx: &@block_ctxt, lhs: ValueRef, rhs: ValueRef,
                        t: &ty::t, llop: ValueRef) {
    assert (ty::type_is_scalar(bcx_tcx(cx), t));

    // In most cases, we need to know whether to do signed, unsigned, or float
    // comparison.

    let rslt = compare_scalar_types(cx, lhs, rhs, t, llop);
    let bcx = rslt.bcx;
    let compare_result = rslt.val;
    bcx.build.Store(compare_result, cx.fcx.llretptr);
    bcx.build.RetVoid();
}


// A helper function to do the actual comparison of scalar values.
fn compare_scalar_values(cx: &@block_ctxt, lhs: ValueRef, rhs: ValueRef,
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
    // we could just write, e.g., "cmp_fn = bind cx.build.FCmp(_, _, _);" in
    // the above, and "auto eq_result = cmp_fn(eq_cmp, lhs, rhs);" in the
    // below.

    fn generic_cmp(cx: &@block_ctxt, nt: scalar_type, op: uint, lhs: ValueRef,
                   rhs: ValueRef) -> ValueRef {
        let r: ValueRef;
        if nt == nil_type {
            r = C_bool(op != 0u);
        } else if (nt == floating_point) {
            r = cx.build.FCmp(op, lhs, rhs);
        } else { r = cx.build.ICmp(op, lhs, rhs); }
        ret r;
    }
    let last_cx = new_sub_block_ctxt(cx, "last");
    let eq_cx = new_sub_block_ctxt(cx, "eq");
    let eq_result = generic_cmp(eq_cx, nt, eq_cmp, lhs, rhs);
    eq_cx.build.Br(last_cx.llbb);
    let lt_cx = new_sub_block_ctxt(cx, "lt");
    let lt_result = generic_cmp(lt_cx, nt, lt_cmp, lhs, rhs);
    lt_cx.build.Br(last_cx.llbb);
    let le_cx = new_sub_block_ctxt(cx, "le");
    let le_result = generic_cmp(le_cx, nt, le_cmp, lhs, rhs);
    le_cx.build.Br(last_cx.llbb);
    let unreach_cx = new_sub_block_ctxt(cx, "unreach");
    unreach_cx.build.Unreachable();
    let llswitch = cx.build.Switch(llop, unreach_cx.llbb, 3u);
    llvm::LLVMAddCase(llswitch, C_u8(abi::cmp_glue_op_eq), eq_cx.llbb);
    llvm::LLVMAddCase(llswitch, C_u8(abi::cmp_glue_op_lt), lt_cx.llbb);
    llvm::LLVMAddCase(llswitch, C_u8(abi::cmp_glue_op_le), le_cx.llbb);
    let last_result =
        last_cx.build.Phi(T_i1(), ~[eq_result, lt_result, le_result],
                          ~[eq_cx.llbb, lt_cx.llbb, le_cx.llbb]);
    ret rslt(last_cx, last_result);
}

type val_pair_fn = fn(&@block_ctxt, ValueRef, ValueRef) -> result ;

type val_and_ty_fn = fn(&@block_ctxt, ValueRef, ty::t) -> result ;

type val_pair_and_ty_fn =
    fn(&@block_ctxt, ValueRef, ValueRef, ty::t) -> result ;


// Iterates through the elements of a structural type.
fn iter_structural_ty(cx: &@block_ctxt, v: ValueRef, t: &ty::t,
                      f: val_and_ty_fn) -> result {
    fn adaptor_fn(f: val_and_ty_fn, cx: &@block_ctxt, av: ValueRef,
                  bv: ValueRef, t: ty::t) -> result {
        ret f(cx, av, t);
    }
    ret iter_structural_ty_full(cx, v, v, t, bind adaptor_fn(f, _, _, _, _));
}

fn load_inbounds(cx: &@block_ctxt, p: ValueRef, idxs: &ValueRef[]) ->
   ValueRef {
    ret cx.build.Load(cx.build.InBoundsGEP(p, idxs));
}

fn store_inbounds(cx: &@block_ctxt, v: ValueRef, p: ValueRef,
                  idxs: &ValueRef[]) {
    cx.build.Store(v, cx.build.InBoundsGEP(p, idxs));
}

// This uses store and inboundsGEP, but it only doing so superficially; it's
// really storing an incremented pointer to another pointer.
fn incr_ptr(cx: &@block_ctxt, p: ValueRef, incr: ValueRef, pp: ValueRef) {
    cx.build.Store(cx.build.InBoundsGEP(p, ~[incr]), pp);
}

fn iter_structural_ty_full(cx: &@block_ctxt, av: ValueRef, bv: ValueRef,
                           t: &ty::t, f: &val_pair_and_ty_fn) -> result {
    fn iter_boxpp(cx: @block_ctxt, box_a_cell: ValueRef, box_b_cell: ValueRef,
                  f: &val_pair_and_ty_fn) -> result {
        let box_a_ptr = cx.build.Load(box_a_cell);
        let box_b_ptr = cx.build.Load(box_b_cell);
        let tnil = ty::mk_nil(bcx_tcx(cx));
        let tbox = ty::mk_imm_box(bcx_tcx(cx), tnil);
        let inner_cx = new_sub_block_ctxt(cx, "iter box");
        let next_cx = new_sub_block_ctxt(cx, "next");
        let null_test = cx.build.IsNull(box_a_ptr);
        cx.build.CondBr(null_test, next_cx.llbb, inner_cx.llbb);
        let r = f(inner_cx, box_a_ptr, box_b_ptr, tbox);
        r.bcx.build.Br(next_cx.llbb);
        ret rslt(next_cx, C_nil());
    }

    fn iter_ivec(bcx: @block_ctxt, av: ValueRef, bv: ValueRef, unit_ty: ty::t,
                 f: &val_pair_and_ty_fn) -> result {
        // FIXME: "unimplemented rebinding existing function" workaround

        fn adapter(bcx: &@block_ctxt, av: ValueRef, bv: ValueRef,
                   unit_ty: ty::t, f: val_pair_and_ty_fn) -> result {
            ret f(bcx, av, bv, unit_ty);
        }
        let llunitty = type_of_or_i8(bcx, unit_ty);
        let rs = size_of(bcx, unit_ty);
        let unit_sz = rs.val;
        bcx = rs.bcx;
        let a_len_and_data = ivec::get_len_and_data(bcx, av, unit_ty);
        let a_len = a_len_and_data.len;
        let a_elem = a_len_and_data.data;
        bcx = a_len_and_data.bcx;
        let b_len_and_data = ivec::get_len_and_data(bcx, bv, unit_ty);
        let b_len = b_len_and_data.len;
        let b_elem = b_len_and_data.data;
        bcx = b_len_and_data.bcx;
        // Calculate the last pointer address we want to handle.
        // TODO: Optimize this when the size of the unit type is statically
        // known to not use pointer casts, which tend to confuse LLVM.

        let len = umin(bcx, a_len, b_len);
        let b_elem_i8 = bcx.build.PointerCast(b_elem, T_ptr(T_i8()));
        let b_end_i8 = bcx.build.GEP(b_elem_i8, ~[len]);
        let b_end = bcx.build.PointerCast(b_end_i8, T_ptr(llunitty));

        let dest_elem_ptr = alloca(bcx, T_ptr(llunitty));
        let src_elem_ptr = alloca(bcx, T_ptr(llunitty));
        bcx.build.Store(a_elem, dest_elem_ptr);
        bcx.build.Store(b_elem, src_elem_ptr);

        // Now perform the iteration.
        let loop_header_cx = new_sub_block_ctxt(bcx, "iter_ivec_loop_header");
        bcx.build.Br(loop_header_cx.llbb);
        let dest_elem = loop_header_cx.build.Load(dest_elem_ptr);
        let src_elem = loop_header_cx.build.Load(src_elem_ptr);
        let not_yet_at_end =
            loop_header_cx.build.ICmp(lib::llvm::LLVMIntULT, dest_elem,
                                      b_end);
        let loop_body_cx = new_sub_block_ctxt(bcx, "iter_ivec_loop_body");
        let next_cx = new_sub_block_ctxt(bcx, "iter_ivec_next");
        loop_header_cx.build.CondBr(not_yet_at_end, loop_body_cx.llbb,
                                    next_cx.llbb);

        rs =
            f(loop_body_cx,
              load_if_immediate(loop_body_cx, dest_elem, unit_ty),
              load_if_immediate(loop_body_cx, src_elem, unit_ty), unit_ty);

        loop_body_cx = rs.bcx;

        let increment;
        if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
            increment = unit_sz;
        } else { increment = C_int(1); }

        incr_ptr(loop_body_cx, dest_elem, increment, dest_elem_ptr);
        incr_ptr(loop_body_cx, src_elem, increment, src_elem_ptr);
        loop_body_cx.build.Br(loop_header_cx.llbb);

        ret rslt(next_cx, C_nil());
    }

    fn iter_variant(cx: @block_ctxt, a_tup: ValueRef, b_tup: ValueRef,
                    variant: &ty::variant_info, tps: &ty::t[],
                    tid: &ast::def_id, f: &val_pair_and_ty_fn) -> result {
        if std::ivec::len[ty::t](variant.args) == 0u {
            ret rslt(cx, C_nil());
        }
        let fn_ty = variant.ctor_ty;
        let ccx = bcx_ccx(cx);
        alt ty::struct(ccx.tcx, fn_ty) {
          ty::ty_fn(_, args, _, _, _) {
            let j = 0;
            for a: ty::arg  in args {
                let rslt = GEP_tag(cx, a_tup, tid, variant.id, tps, j);
                let llfldp_a = rslt.val;
                cx = rslt.bcx;
                rslt = GEP_tag(cx, b_tup, tid, variant.id, tps, j);
                let llfldp_b = rslt.val;
                cx = rslt.bcx;
                let ty_subst = ty::substitute_type_params(ccx.tcx, tps, a.ty);
                let llfld_a = load_if_immediate(cx, llfldp_a, ty_subst);
                let llfld_b = load_if_immediate(cx, llfldp_b, ty_subst);
                rslt = f(cx, llfld_a, llfld_b, ty_subst);
                cx = rslt.bcx;
                j += 1;
            }
          }
        }
        ret rslt(cx, C_nil());
    }

    let r: result = rslt(cx, C_nil());
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_rec(fields) {
        let i: int = 0;
        for fld: ty::field  in fields {
            r = GEP_tup_like(r.bcx, t, av, ~[0, i]);
            let llfld_a = r.val;
            r = GEP_tup_like(r.bcx, t, bv, ~[0, i]);
            let llfld_b = r.val;
            r =
                f(r.bcx, load_if_immediate(r.bcx, llfld_a, fld.mt.ty),
                  load_if_immediate(r.bcx, llfld_b, fld.mt.ty), fld.mt.ty);
            i += 1;
        }
      }
      ty::ty_res(_, inner, tps) {
        let tcx = bcx_tcx(cx);
        let inner1 = ty::substitute_type_params(tcx, tps, inner);
        let inner_t_s = ty::substitute_type_params(tcx, tps, inner);
        let tup_t = ty::mk_imm_tup(tcx, ~[ty::mk_int(tcx), inner_t_s]);
        r = GEP_tup_like(r.bcx, tup_t, av, ~[0, 1]);
        let llfld_a = r.val;
        r = GEP_tup_like(r.bcx, tup_t, bv, ~[0, 1]);
        let llfld_b = r.val;
        r = f(r.bcx, load_if_immediate(r.bcx, llfld_a, inner1),
              load_if_immediate(r.bcx, llfld_b, inner1), inner1);
      }
      ty::ty_tag(tid, tps) {
        let variants = ty::tag_variants(bcx_tcx(cx), tid);
        let n_variants = std::ivec::len(variants);

        // Cast the tags to types we can GEP into.
        if n_variants == 1u {
            ret iter_variant(cx, av, bv, variants.(0), tps, tid, f);
        }

        let lltagty = T_opaque_tag_ptr(bcx_ccx(cx).tn);
        let av_tag = cx.build.PointerCast(av, lltagty);
        let bv_tag = cx.build.PointerCast(bv, lltagty);
        let lldiscrim_a_ptr = cx.build.GEP(av_tag, ~[C_int(0), C_int(0)]);
        let llunion_a_ptr = cx.build.GEP(av_tag, ~[C_int(0), C_int(1)]);
        let lldiscrim_a = cx.build.Load(lldiscrim_a_ptr);
        let lldiscrim_b_ptr = cx.build.GEP(bv_tag, ~[C_int(0), C_int(0)]);
        let llunion_b_ptr = cx.build.GEP(bv_tag, ~[C_int(0), C_int(1)]);
        let lldiscrim_b = cx.build.Load(lldiscrim_b_ptr);

        // NB: we must hit the discriminant first so that structural
        // comparison know not to proceed when the discriminants differ.
        let bcx = cx;
        bcx = f(bcx, lldiscrim_a, lldiscrim_b, ty::mk_int(bcx_tcx(cx))).bcx;
        let unr_cx = new_sub_block_ctxt(bcx, "tag-iter-unr");
        unr_cx.build.Unreachable();
        let llswitch = bcx.build.Switch(lldiscrim_a, unr_cx.llbb, n_variants);
        let next_cx = new_sub_block_ctxt(bcx, "tag-iter-next");
        let i = 0u;
        for variant: ty::variant_info  in variants {
            let variant_cx =
                new_sub_block_ctxt(bcx,
                                   "tag-iter-variant-" +
                                       uint::to_str(i, 10u));
            llvm::LLVMAddCase(llswitch, C_int(i as int), variant_cx.llbb);
            variant_cx =
                iter_variant(variant_cx, llunion_a_ptr, llunion_b_ptr,
                             variant, tps, tid, f).bcx;
            variant_cx.build.Br(next_cx.llbb);
            i += 1u;
        }
        ret rslt(next_cx, C_nil());
      }
      ty::ty_fn(_, _, _, _, _) {
        let box_cell_a =
            cx.build.GEP(av, ~[C_int(0), C_int(abi::fn_field_box)]);
        let box_cell_b =
            cx.build.GEP(bv, ~[C_int(0), C_int(abi::fn_field_box)]);
        ret iter_boxpp(cx, box_cell_a, box_cell_b, f);
      }
      ty::ty_obj(_) {
        let box_cell_a =
            cx.build.GEP(av, ~[C_int(0), C_int(abi::obj_field_box)]);
        let box_cell_b =
            cx.build.GEP(bv, ~[C_int(0), C_int(abi::obj_field_box)]);
        ret iter_boxpp(cx, box_cell_a, box_cell_b, f);
      }
      ty::ty_ivec(unit_tm) { ret iter_ivec(cx, av, bv, unit_tm.ty, f); }
      ty::ty_istr. {
        let unit_ty = ty::mk_mach(bcx_tcx(cx), ast::ty_u8);
        ret iter_ivec(cx, av, bv, unit_ty, f);
      }
      _ { bcx_ccx(cx).sess.unimpl("type in iter_structural_ty_full"); }
    }
    ret r;
}


// Iterates through a pointer range, until the src* hits the src_lim*.
fn iter_sequence_raw(cx: @block_ctxt, dst: ValueRef,
                     // elt*
                     src: ValueRef,
                     // elt*
                     src_lim: ValueRef,
                     // elt*
                     elt_sz: ValueRef, f: &val_pair_fn) -> result {
    let bcx = cx;
    let dst_int: ValueRef = vp2i(bcx, dst);
    let src_int: ValueRef = vp2i(bcx, src);
    let src_lim_int: ValueRef = vp2i(bcx, src_lim);
    let cond_cx = new_scope_block_ctxt(cx, "sequence-iter cond");
    let body_cx = new_scope_block_ctxt(cx, "sequence-iter body");
    let next_cx = new_sub_block_ctxt(cx, "next");
    bcx.build.Br(cond_cx.llbb);
    let dst_curr: ValueRef =
        cond_cx.build.Phi(T_int(), ~[dst_int], ~[bcx.llbb]);
    let src_curr: ValueRef =
        cond_cx.build.Phi(T_int(), ~[src_int], ~[bcx.llbb]);
    let end_test =
        cond_cx.build.ICmp(lib::llvm::LLVMIntULT, src_curr, src_lim_int);
    cond_cx.build.CondBr(end_test, body_cx.llbb, next_cx.llbb);
    let dst_curr_ptr = vi2p(body_cx, dst_curr, T_ptr(T_i8()));
    let src_curr_ptr = vi2p(body_cx, src_curr, T_ptr(T_i8()));
    let body_res = f(body_cx, dst_curr_ptr, src_curr_ptr);
    body_cx = body_res.bcx;
    let dst_next = body_cx.build.Add(dst_curr, elt_sz);
    let src_next = body_cx.build.Add(src_curr, elt_sz);
    body_cx.build.Br(cond_cx.llbb);
    cond_cx.build.AddIncomingToPhi(dst_curr, ~[dst_next], ~[body_cx.llbb]);
    cond_cx.build.AddIncomingToPhi(src_curr, ~[src_next], ~[body_cx.llbb]);
    ret rslt(next_cx, C_nil());
}

fn iter_sequence_inner(cx: &@block_ctxt, src: ValueRef,
                       src_lim:

                           // elt*
                           ValueRef,
                       elt_ty: & // elt*
                           ty::t, f: &val_and_ty_fn) -> result {
    fn adaptor_fn(f: val_and_ty_fn, elt_ty: ty::t, cx: &@block_ctxt,
                  dst: ValueRef, src: ValueRef) -> result {
        let llptrty;
        if !ty::type_has_dynamic_size(bcx_tcx(cx), elt_ty) {
            let llty = type_of(bcx_ccx(cx), cx.sp, elt_ty);
            llptrty = T_ptr(llty);
        } else { llptrty = T_ptr(T_ptr(T_i8())); }
        let p = cx.build.PointerCast(src, llptrty);
        ret f(cx, load_if_immediate(cx, p, elt_ty), elt_ty);
    }
    let elt_sz = size_of(cx, elt_ty);
    ret iter_sequence_raw(elt_sz.bcx, src, src, src_lim, elt_sz.val,
                          bind adaptor_fn(f, elt_ty, _, _, _));
}


// Iterates through the elements of a vec or str.
fn iter_sequence(cx: @block_ctxt, v: ValueRef, t: &ty::t, f: &val_and_ty_fn)
   -> result {
    fn iter_sequence_body(cx: @block_ctxt, v: ValueRef, elt_ty: &ty::t,
                          f: &val_and_ty_fn, trailing_null: bool,
                          interior: bool) -> result {
        let p0;
        let len;
        let bcx;
        if !interior {
            p0 = cx.build.GEP(v, ~[C_int(0), C_int(abi::vec_elt_data)]);
            let lp = cx.build.GEP(v, ~[C_int(0), C_int(abi::vec_elt_fill)]);
            len = cx.build.Load(lp);
            bcx = cx;
        } else {
            let len_and_data_rslt = ivec::get_len_and_data(cx, v, elt_ty);
            len = len_and_data_rslt.len;
            p0 = len_and_data_rslt.data;
            bcx = len_and_data_rslt.bcx;
        }

        let llunit_ty = type_of_or_i8(cx, elt_ty);
        if trailing_null {
            let unit_sz = size_of(bcx, elt_ty);
            bcx = unit_sz.bcx;
            len = bcx.build.Sub(len, unit_sz.val);
        }
        let p1 =
            vi2p(bcx, bcx.build.Add(vp2i(bcx, p0), len), T_ptr(llunit_ty));
        ret iter_sequence_inner(bcx, p0, p1, elt_ty, f);
    }


    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_vec(elt) {
        ret iter_sequence_body(cx, v, elt.ty, f, false, false);
      }
      ty::ty_str. {
        let et = ty::mk_mach(bcx_tcx(cx), ast::ty_u8);
        ret iter_sequence_body(cx, v, et, f, true, false);
      }
      ty::ty_ivec(elt) {
        ret iter_sequence_body(cx, v, elt.ty, f, false, true);
      }
      ty::ty_istr. {
        let et = ty::mk_mach(bcx_tcx(cx), ast::ty_u8);
        ret iter_sequence_body(cx, v, et, f, true, true);
      }
      _ {
        bcx_ccx(cx).sess.bug("unexpected type in \
                                 trans::iter_sequence: "
                                 + ty_to_str(cx.fcx.lcx.ccx.tcx, t));
      }
    }
}

fn lazily_emit_all_tydesc_glue(cx: &@block_ctxt,
                               static_ti: &option::t[@tydesc_info]) {
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_copy_glue, static_ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_drop_glue, static_ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_free_glue, static_ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_cmp_glue, static_ti);
}

fn lazily_emit_all_generic_info_tydesc_glues(cx: &@block_ctxt,
                                             gi: &generic_info) {
    for ti: option::t[@tydesc_info]  in gi.static_tis {
        lazily_emit_all_tydesc_glue(cx, ti);
    }
}

fn lazily_emit_tydesc_glue(cx: &@block_ctxt, field: int,
                           static_ti: &option::t[@tydesc_info]) {
    alt static_ti {
      none. { }
      some(ti) {
        if field == abi::tydesc_field_copy_glue {
            alt { ti.copy_glue } {
              some(_) { }
              none. {
                log #fmt("+++ lazily_emit_tydesc_glue TAKE %s",
                         ty_to_str(bcx_tcx(cx), ti.ty));
                let lcx = cx.fcx.lcx;
                let glue_fn =
                    declare_generic_glue(lcx, ti.ty, T_glue_fn(*lcx.ccx),
                                         "copy");
                ti.copy_glue = some[ValueRef](glue_fn);
                make_generic_glue(lcx, cx.sp, ti.ty, glue_fn,
                                  mgghf_single(make_copy_glue), ti.ty_params,
                                  "take");
                log #fmt("--- lazily_emit_tydesc_glue TAKE %s",
                         ty_to_str(bcx_tcx(cx), ti.ty));
              }
            }
        } else if (field == abi::tydesc_field_drop_glue) {
            alt { ti.drop_glue } {
              some(_) { }
              none. {
                log #fmt("+++ lazily_emit_tydesc_glue DROP %s",
                         ty_to_str(bcx_tcx(cx), ti.ty));
                let lcx = cx.fcx.lcx;
                let glue_fn =
                    declare_generic_glue(lcx, ti.ty, T_glue_fn(*lcx.ccx),
                                         "drop");
                ti.drop_glue = some[ValueRef](glue_fn);
                make_generic_glue(lcx, cx.sp, ti.ty, glue_fn,
                                  mgghf_single(make_drop_glue), ti.ty_params,
                                  "drop");
                log #fmt("--- lazily_emit_tydesc_glue DROP %s",
                         ty_to_str(bcx_tcx(cx), ti.ty));
              }
            }
        } else if (field == abi::tydesc_field_free_glue) {
            alt { ti.free_glue } {
              some(_) { }
              none. {
                log #fmt("+++ lazily_emit_tydesc_glue FREE %s",
                         ty_to_str(bcx_tcx(cx), ti.ty));
                let lcx = cx.fcx.lcx;
                let glue_fn =
                    declare_generic_glue(lcx, ti.ty, T_glue_fn(*lcx.ccx),
                                         "free");
                ti.free_glue = some[ValueRef](glue_fn);
                make_generic_glue(lcx, cx.sp, ti.ty, glue_fn,
                                  mgghf_single(make_free_glue), ti.ty_params,
                                  "free");
                log #fmt("--- lazily_emit_tydesc_glue FREE %s",
                         ty_to_str(bcx_tcx(cx), ti.ty));
              }
            }
        } else if (field == abi::tydesc_field_cmp_glue) {
            alt { ti.cmp_glue } {
              some(_) { }
              none. {
                log #fmt("+++ lazily_emit_tydesc_glue CMP %s",
                         ty_to_str(bcx_tcx(cx), ti.ty));
                let lcx = cx.fcx.lcx;
                let glue_fn =
                    declare_generic_glue(lcx, ti.ty, T_cmp_glue_fn(*lcx.ccx),
                                         "cmp");
                ti.cmp_glue = some[ValueRef](glue_fn);
                make_generic_glue(lcx, cx.sp, ti.ty, glue_fn, mgghf_cmp,
                                  ti.ty_params, "cmp");
                log #fmt("--- lazily_emit_tydesc_glue CMP %s",
                         ty_to_str(bcx_tcx(cx), ti.ty));
              }
            }
        }
      }
    }
}

fn call_tydesc_glue_full(cx: &@block_ctxt, v: ValueRef, tydesc: ValueRef,
                         field: int, static_ti: &option::t[@tydesc_info]) {
    lazily_emit_tydesc_glue(cx, field, static_ti);

    let static_glue_fn = none;
    alt static_ti {
      none. {/* no-op */ }
      some(sti) {
        if field == abi::tydesc_field_copy_glue {
            static_glue_fn = sti.copy_glue;
        } else if (field == abi::tydesc_field_drop_glue) {
            static_glue_fn = sti.drop_glue;
        } else if (field == abi::tydesc_field_free_glue) {
            static_glue_fn = sti.free_glue;
        } else if (field == abi::tydesc_field_cmp_glue) {
            static_glue_fn = sti.cmp_glue;
        }
      }
    }

    let llrawptr = cx.build.BitCast(v, T_ptr(T_i8()));
    let lltydescs =
        cx.build.GEP(tydesc,
                     ~[C_int(0), C_int(abi::tydesc_field_first_param)]);
    lltydescs = cx.build.Load(lltydescs);

    let llfn;
    alt static_glue_fn {
      none. {
        let llfnptr = cx.build.GEP(tydesc, ~[C_int(0), C_int(field)]);
        llfn = cx.build.Load(llfnptr);
      }
      some(sgf) { llfn = sgf; }
    }

    cx.build.Call(llfn,
                  ~[C_null(T_ptr(T_nil())), cx.fcx.lltaskptr,
                    C_null(T_ptr(T_nil())), lltydescs, llrawptr]);
}

fn call_tydesc_glue(cx: &@block_ctxt, v: ValueRef, t: &ty::t, field: int) ->
   result {
    let ti: option::t[@tydesc_info] = none[@tydesc_info];
    let td = get_tydesc(cx, t, false, ti);
    call_tydesc_glue_full(td.bcx, spill_if_immediate(td.bcx, v, t), td.val,
                          field, ti);
    ret rslt(td.bcx, C_nil());
}

fn call_cmp_glue(cx: &@block_ctxt, lhs: ValueRef, rhs: ValueRef, t: &ty::t,
                 llop: ValueRef) -> result {
    // We can't use call_tydesc_glue_full() and friends here because compare
    // glue has a special signature.

    let lllhs = spill_if_immediate(cx, lhs, t);
    let llrhs = spill_if_immediate(cx, rhs, t);
    let llrawlhsptr = cx.build.BitCast(lllhs, T_ptr(T_i8()));
    let llrawrhsptr = cx.build.BitCast(llrhs, T_ptr(T_i8()));
    let ti = none[@tydesc_info];
    let r = get_tydesc(cx, t, false, ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_cmp_glue, ti);
    let lltydescs =
        r.bcx.build.GEP(r.val,
                        ~[C_int(0), C_int(abi::tydesc_field_first_param)]);
    lltydescs = r.bcx.build.Load(lltydescs);

    let llfn;
    alt ti {
      none. {
        let llfnptr =
            r.bcx.build.GEP(r.val,
                            ~[C_int(0), C_int(abi::tydesc_field_cmp_glue)]);
        llfn = r.bcx.build.Load(llfnptr);
      }
      some(sti) { llfn = option::get(sti.cmp_glue); }
    }

    let llcmpresultptr = alloca(r.bcx, T_i1());
    let llargs: ValueRef[] =
        ~[llcmpresultptr, r.bcx.fcx.lltaskptr, C_null(T_ptr(T_nil())),
          lltydescs, llrawlhsptr, llrawrhsptr, llop];
    r.bcx.build.Call(llfn, llargs);
    ret rslt(r.bcx, r.bcx.build.Load(llcmpresultptr));
}

// Compares two values. Performs the simple scalar comparison if the types are
// scalar and calls to comparison glue otherwise.
fn compare(cx: &@block_ctxt, lhs: ValueRef, rhs: ValueRef, t: &ty::t,
           llop: ValueRef) -> result {
    if ty::type_is_scalar(bcx_tcx(cx), t) {
        ret compare_scalar_types(cx, lhs, rhs, t, llop);
    }
    ret call_cmp_glue(cx, lhs, rhs, t, llop);
}

fn copy_ty(cx: &@block_ctxt, v: ValueRef, t: ty::t) -> result {
    if ty::type_has_pointers(bcx_tcx(cx), t) ||
           ty::type_owns_heap_mem(bcx_tcx(cx), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_copy_glue);
    }
    ret rslt(cx, C_nil());
}

fn drop_slot(cx: &@block_ctxt, slot: ValueRef, t: &ty::t) -> result {
    let llptr = load_if_immediate(cx, slot, t);
    let re = drop_ty(cx, llptr, t);
    let llty = val_ty(slot);
    let llelemty = lib::llvm::llvm::LLVMGetElementType(llty);
    re.bcx.build.Store(C_null(llelemty), slot);
    ret re;
}

fn drop_ty(cx: &@block_ctxt, v: ValueRef, t: ty::t) -> result {
    if ty::type_needs_drop(bcx_tcx(cx), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_drop_glue);
    }
    ret rslt(cx, C_nil());
}

fn free_ty(cx: &@block_ctxt, v: ValueRef, t: ty::t) -> result {
    if ty::type_has_pointers(bcx_tcx(cx), t) {
        ret call_tydesc_glue(cx, v, t, abi::tydesc_field_free_glue);
    }
    ret rslt(cx, C_nil());
}

fn call_memmove(cx: &@block_ctxt, dst: ValueRef, src: ValueRef,
                n_bytes: ValueRef) -> result {
    // FIXME: switch to the 64-bit variant when on such a platform.
    // TODO: Provide LLVM with better alignment information when the alignment
    // is statically known (it must be nothing more than a constant int, or
    // LLVM complains -- not even a constant element of a tydesc works).

    let i = bcx_ccx(cx).intrinsics;
    assert (i.contains_key("llvm.memmove.p0i8.p0i8.i32"));
    let memmove = i.get("llvm.memmove.p0i8.p0i8.i32");
    let src_ptr = cx.build.PointerCast(src, T_ptr(T_i8()));
    let dst_ptr = cx.build.PointerCast(dst, T_ptr(T_i8()));
    let size = cx.build.IntCast(n_bytes, T_i32());
    let align = C_int(0);
    let volatile = C_bool(false);
    ret rslt(cx,
             cx.build.Call(memmove,
                           ~[dst_ptr, src_ptr, size, align, volatile]));
}

fn call_bzero(cx: &@block_ctxt, dst: ValueRef, n_bytes: ValueRef,
              align_bytes: ValueRef) -> result {
    // FIXME: switch to the 64-bit variant when on such a platform.

    let i = bcx_ccx(cx).intrinsics;
    assert (i.contains_key("llvm.memset.p0i8.i32"));
    let memset = i.get("llvm.memset.p0i8.i32");
    let dst_ptr = cx.build.PointerCast(dst, T_ptr(T_i8()));
    let size = cx.build.IntCast(n_bytes, T_i32());
    let align =
        if lib::llvm::llvm::LLVMIsConstant(align_bytes) == True {
            cx.build.IntCast(align_bytes, T_i32())
        } else { cx.build.IntCast(C_int(0), T_i32()) };
    let volatile = C_bool(false);
    ret rslt(cx,
             cx.build.Call(memset,
                           ~[dst_ptr, C_u8(0u), size, align, volatile]));
}

fn memmove_ty(cx: &@block_ctxt, dst: ValueRef, src: ValueRef, t: &ty::t) ->
   result {
    if ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        let llsz = size_of(cx, t);
        ret call_memmove(llsz.bcx, dst, src, llsz.val);
    } else { ret rslt(cx, cx.build.Store(cx.build.Load(src), dst)); }
}

// Duplicates any heap-owned memory owned by a value of the given type.
fn duplicate_heap_parts_if_necessary(cx: &@block_ctxt, vptr: ValueRef,
                                     typ: ty::t) -> result {
    alt ty::struct(bcx_tcx(cx), typ) {
      ty::ty_ivec(tm) { ret ivec::duplicate_heap_part(cx, vptr, tm.ty); }
      ty::ty_istr. {
        ret ivec::duplicate_heap_part(cx, vptr,
                                      ty::mk_mach(bcx_tcx(cx), ast::ty_u8));
      }
      _ { ret rslt(cx, C_nil()); }
    }
}

tag copy_action { INIT; DROP_EXISTING; }

fn copy_val(cx: &@block_ctxt, action: copy_action, dst: ValueRef,
            src: ValueRef, t: &ty::t) -> result {
    let ccx = bcx_ccx(cx);
    // FIXME this is just a clunky stopgap. we should do proper checking in an
    // earlier pass.
    if !ty::type_is_copyable(ccx.tcx, t) {
        ccx.sess.span_fatal(cx.sp, "Copying a non-copyable type.");
    }

    if ty::type_is_scalar(ccx.tcx, t) || ty::type_is_native(ccx.tcx, t) {
        ret rslt(cx, cx.build.Store(src, dst));
    } else if (ty::type_is_nil(ccx.tcx, t) || ty::type_is_bot(ccx.tcx, t)) {
        ret rslt(cx, C_nil());
    } else if (ty::type_is_boxed(ccx.tcx, t)) {
        let bcx;
        if action == DROP_EXISTING {
            bcx = drop_ty(cx, cx.build.Load(dst), t).bcx;
        } else { bcx = cx; }
        bcx = copy_ty(bcx, src, t).bcx;
        ret rslt(bcx, bcx.build.Store(src, dst));
    } else if (ty::type_is_structural(ccx.tcx, t) ||
                   ty::type_has_dynamic_size(ccx.tcx, t)) {
        // Check for self-assignment.
        let do_copy_cx = new_sub_block_ctxt(cx, "do_copy");
        let next_cx = new_sub_block_ctxt(cx, "next");
        let self_assigning =
            cx.build.ICmp(lib::llvm::LLVMIntNE,
                          cx.build.PointerCast(dst, val_ty(src)), src);
        cx.build.CondBr(self_assigning, do_copy_cx.llbb, next_cx.llbb);

        if action == DROP_EXISTING {
            do_copy_cx = drop_ty(do_copy_cx, dst, t).bcx;
        }
        do_copy_cx = memmove_ty(do_copy_cx, dst, src, t).bcx;
        do_copy_cx = copy_ty(do_copy_cx, dst, t).bcx;
        do_copy_cx.build.Br(next_cx.llbb);

        ret rslt(next_cx, C_nil());
    }
    ccx.sess.bug("unexpected type in trans::copy_val: " +
                     ty_to_str(ccx.tcx, t));
}


// This works like copy_val, except that it deinitializes the source.
// Since it needs to zero out the source, src also needs to be an lval.
// FIXME: We always zero out the source. Ideally we would detect the
// case where a variable is always deinitialized by block exit and thus
// doesn't need to be dropped.
fn move_val(cx: @block_ctxt, action: copy_action, dst: ValueRef,
            src: &lval_result, t: &ty::t) -> result {
    let src_val = src.res.val;
    if ty::type_is_scalar(bcx_tcx(cx), t) ||
           ty::type_is_native(bcx_tcx(cx), t) {
        if src.is_mem { src_val = cx.build.Load(src_val); }
        cx.build.Store(src_val, dst);
        ret rslt(cx, C_nil());
    } else if (ty::type_is_nil(bcx_tcx(cx), t) ||
                   ty::type_is_bot(bcx_tcx(cx), t)) {
        ret rslt(cx, C_nil());
    } else if (ty::type_is_boxed(bcx_tcx(cx), t)) {
        if src.is_mem { src_val = cx.build.Load(src_val); }
        if action == DROP_EXISTING {
            cx = drop_ty(cx, cx.build.Load(dst), t).bcx;
        }
        cx.build.Store(src_val, dst);
        if src.is_mem {
            ret zero_alloca(cx, src.res.val, t);
        } else { // It must be a temporary
            revoke_clean(cx, src_val);
            ret rslt(cx, C_nil());
        }
    } else if (ty::type_is_structural(bcx_tcx(cx), t) ||
                   ty::type_has_dynamic_size(bcx_tcx(cx), t)) {
        if action == DROP_EXISTING { cx = drop_ty(cx, dst, t).bcx; }
        cx = memmove_ty(cx, dst, src_val, t).bcx;
        if src.is_mem {
            ret zero_alloca(cx, src_val, t);
        } else { // Temporary value
            revoke_clean(cx, src_val);
            ret rslt(cx, C_nil());
        }
    }
    bcx_ccx(cx).sess.bug("unexpected type in trans::move_val: " +
                             ty_to_str(bcx_tcx(cx), t));
}

fn move_val_if_temp(cx: @block_ctxt, action: copy_action, dst: ValueRef,
                    src: &lval_result, t: &ty::t) -> result {

    // Lvals in memory are not temporaries. Copy them.
    if src.is_mem {
        ret copy_val(cx, action, dst, load_if_immediate(cx, src.res.val, t),
                     t);
    } else { ret move_val(cx, action, dst, src, t); }
}

fn trans_lit_istr(cx: &@block_ctxt, s: str) -> result {
    let llstackpart = alloca(cx, T_ivec(T_i8()));
    let len = str::byte_len(s);

    let bcx;
    if len < 3u { // 3 because of the \0
        cx.build.Store(C_uint(len + 1u),
                       cx.build.InBoundsGEP(llstackpart,
                                            ~[C_int(0), C_int(0)]));
        cx.build.Store(C_int(4),
                       cx.build.InBoundsGEP(llstackpart,
                                            ~[C_int(0), C_int(1)]));
        let i = 0u;
        while i < len {
            cx.build.Store(C_u8(s.(i) as uint),
                           cx.build.InBoundsGEP(llstackpart,
                                                ~[C_int(0), C_int(2),
                                                  C_uint(i)]));
            i += 1u;
        }
        cx.build.Store(C_u8(0u),
                       cx.build.InBoundsGEP(llstackpart,
                                            ~[C_int(0), C_int(2),
                                              C_uint(len)]));

        bcx = cx;
    } else {
        let r =
            trans_shared_malloc(cx, T_ptr(T_ivec_heap_part(T_i8())),
                                llsize_of(T_struct(~[T_int(),
                                                     T_array(T_i8(),
                                                             len + 1u)])));
        bcx = r.bcx;
        let llheappart = r.val;

        bcx.build.Store(C_uint(len + 1u),
                        bcx.build.InBoundsGEP(llheappart,
                                              ~[C_int(0), C_int(0)]));
        bcx.build.Store(llvm::LLVMConstString(str::buf(s), len, False),
                        bcx.build.InBoundsGEP(llheappart,
                                              ~[C_int(0), C_int(1)]));

        let llspilledstackpart =
            bcx.build.PointerCast(llstackpart, T_ptr(T_ivec_heap(T_i8())));
        bcx.build.Store(C_int(0),
                        bcx.build.InBoundsGEP(llspilledstackpart,
                                              ~[C_int(0), C_int(0)]));
        bcx.build.Store(C_uint(len + 1u),
                        bcx.build.InBoundsGEP(llspilledstackpart,
                                              ~[C_int(0), C_int(1)]));
        bcx.build.Store(llheappart,
                        bcx.build.InBoundsGEP(llspilledstackpart,
                                              ~[C_int(0), C_int(2)]));
    }

    ret rslt(bcx, llstackpart);
}

fn trans_crate_lit(cx: &@crate_ctxt, lit: &ast::lit) -> ValueRef {
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
      ast::lit_str(s, ast::sk_rc.) { ret C_str(cx, s); }
      ast::lit_str(s, ast::sk_unique.) {
        cx.sess.span_unimpl(lit.span, "unique string in this context");
      }
    }
}

fn trans_lit(cx: &@block_ctxt, lit: &ast::lit) -> result {
    alt lit.node {
      ast::lit_str(s, ast::sk_unique.) { ret trans_lit_istr(cx, s); }
      _ { ret rslt(cx, trans_crate_lit(bcx_ccx(cx), lit)); }
    }
}


// Converts an annotation to a type
fn node_id_type(cx: &@crate_ctxt, id: ast::node_id) -> ty::t {
    ret ty::node_id_to_monotype(cx.tcx, id);
}

fn node_type(cx: &@crate_ctxt, sp: &span, id: ast::node_id) -> TypeRef {
    ret type_of(cx, sp, node_id_type(cx, id));
}

fn trans_unary(cx: &@block_ctxt, op: ast::unop, e: &@ast::expr,
               id: ast::node_id) -> result {
    let e_ty = ty::expr_ty(bcx_tcx(cx), e);
    alt op {
      ast::not. {
        let sub = trans_expr(cx, e);
        let dr = autoderef(sub.bcx, sub.val, ty::expr_ty(bcx_tcx(cx), e));
        ret rslt(dr.bcx, dr.bcx.build.Not(dr.val));
      }
      ast::neg. {
        let sub = trans_expr(cx, e);
        let dr = autoderef(sub.bcx, sub.val, ty::expr_ty(bcx_tcx(cx), e));
        if ty::struct(bcx_tcx(cx), e_ty) == ty::ty_float {
            ret rslt(dr.bcx, dr.bcx.build.FNeg(dr.val));
        } else { ret rslt(dr.bcx, sub.bcx.build.Neg(dr.val)); }
      }
      ast::box(_) {
        let lv = trans_lval(cx, e);
        let box_ty = node_id_type(bcx_ccx(lv.res.bcx), id);
        let sub = trans_malloc_boxed(lv.res.bcx, e_ty);
        let body = sub.body;
        add_clean_temp(cx, sub.box, box_ty);

        // Cast the body type to the type of the value. This is needed to
        // make tags work, since tags have a different LLVM type depending
        // on whether they're boxed or not.
        if !ty::type_has_dynamic_size(bcx_tcx(cx), e_ty) {
            let llety = T_ptr(type_of(bcx_ccx(sub.bcx), e.span, e_ty));
            body = sub.bcx.build.PointerCast(body, llety);
        }
        let res = move_val_if_temp(sub.bcx, INIT, body, lv, e_ty);
        ret rslt(res.bcx, sub.box);
      }
      ast::deref. {
        bcx_ccx(cx).sess.bug("deref expressions should have been \
                                 translated using trans_lval(), not \
                                 trans_unary()");
      }
    }
}

fn trans_compare(cx0: &@block_ctxt, op: ast::binop,
                 lhs0: ValueRef, lhs_t: ty::t, rhs0: ValueRef,
                rhs_t: ty::t) -> result {
    // Autoderef both sides.

    let cx = cx0;
    let lhs_r = autoderef(cx, lhs0, lhs_t);
    let lhs = lhs_r.val;
    cx = lhs_r.bcx;
    let rhs_r = autoderef(cx, rhs0, rhs_t);
    let rhs = rhs_r.val;
    cx = rhs_r.bcx;
    // Determine the operation we need.

    let llop;
    alt op {
      ast::eq. | ast::ne. { llop = C_u8(abi::cmp_glue_op_eq); }
      ast::lt. | ast::ge. { llop = C_u8(abi::cmp_glue_op_lt); }
      ast::le. | ast::gt. { llop = C_u8(abi::cmp_glue_op_le); }
    }

    let rs = compare(cx, lhs, rhs, rhs_r.ty, llop);

    // Invert the result if necessary.
    alt op {
      ast::eq. | ast::lt. | ast::le. { ret rslt(rs.bcx, rs.val); }
      ast::ne. | ast::ge. | ast::gt. {
        ret rslt(rs.bcx, rs.bcx.build.Not(rs.val));
      }
    }
}

fn trans_vec_append(cx: &@block_ctxt, t: &ty::t, lhs: ValueRef, rhs: ValueRef)
   -> result {
    let elt_ty = ty::sequence_element_type(bcx_tcx(cx), t);
    let skip_null = C_bool(false);
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_str. { skip_null = C_bool(true); }
      _ { }
    }
    let bcx = cx;
    let ti = none[@tydesc_info];
    let llvec_tydesc = get_tydesc(bcx, t, false, ti);
    bcx = llvec_tydesc.bcx;
    ti = none[@tydesc_info];
    let llelt_tydesc = get_tydesc(bcx, elt_ty, false, ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_copy_glue, ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_drop_glue, ti);
    lazily_emit_tydesc_glue(cx, abi::tydesc_field_free_glue, ti);
    bcx = llelt_tydesc.bcx;
    let dst = bcx.build.PointerCast(lhs, T_ptr(T_opaque_vec_ptr()));
    let src = bcx.build.PointerCast(rhs, T_opaque_vec_ptr());
    ret rslt(bcx,
             bcx.build.Call(bcx_ccx(cx).upcalls.vec_append,
                            ~[cx.fcx.lltaskptr, llvec_tydesc.val,
                              llelt_tydesc.val, dst, src, skip_null]));
}

mod ivec {

    // Returns the length of an interior vector and a pointer to its first
    // element, in that order.
    fn get_len_and_data(bcx: &@block_ctxt, orig_v: ValueRef, unit_ty: ty::t)
       -> {len: ValueRef, data: ValueRef, bcx: @block_ctxt} {
        // If this interior vector has dynamic size, we can't assume anything
        // about the LLVM type of the value passed in, so we cast it to an
        // opaque vector type.
        let v;
        if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
            v = bcx.build.PointerCast(orig_v, T_ptr(T_opaque_ivec()));
        } else { v = orig_v; }

        let llunitty = type_of_or_i8(bcx, unit_ty);
        let stack_len =
            load_inbounds(bcx, v, ~[C_int(0), C_uint(abi::ivec_elt_len)]);
        let stack_elem =
            bcx.build.InBoundsGEP(v,
                                  ~[C_int(0), C_uint(abi::ivec_elt_elems),
                                    C_int(0)]);
        let on_heap =
            bcx.build.ICmp(lib::llvm::LLVMIntEQ, stack_len, C_int(0));
        let on_heap_cx = new_sub_block_ctxt(bcx, "on_heap");
        let next_cx = new_sub_block_ctxt(bcx, "next");
        bcx.build.CondBr(on_heap, on_heap_cx.llbb, next_cx.llbb);
        let heap_stub =
            on_heap_cx.build.PointerCast(v, T_ptr(T_ivec_heap(llunitty)));
        let heap_ptr =
            load_inbounds(on_heap_cx, heap_stub,
                          ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)]);

        // Check whether the heap pointer is null. If it is, the vector length
        // is truly zero.

        let llstubty = T_ivec_heap(llunitty);
        let llheapptrty = struct_elt(llstubty, abi::ivec_heap_stub_elt_ptr);
        let heap_ptr_is_null =
            on_heap_cx.build.ICmp(lib::llvm::LLVMIntEQ, heap_ptr,
                                  C_null(T_ptr(llheapptrty)));
        let zero_len_cx = new_sub_block_ctxt(bcx, "zero_len");
        let nonzero_len_cx = new_sub_block_ctxt(bcx, "nonzero_len");
        on_heap_cx.build.CondBr(heap_ptr_is_null, zero_len_cx.llbb,
                                nonzero_len_cx.llbb);
        // Technically this context is unnecessary, but it makes this function
        // clearer.

        let zero_len = C_int(0);
        let zero_elem = C_null(T_ptr(llunitty));
        zero_len_cx.build.Br(next_cx.llbb);
        // If we're here, then we actually have a heapified vector.

        let heap_len =
            load_inbounds(nonzero_len_cx, heap_ptr,
                          ~[C_int(0), C_uint(abi::ivec_heap_elt_len)]);
        let heap_elem =
            {
                let v =
                    ~[C_int(0), C_uint(abi::ivec_heap_elt_elems), C_int(0)];
                nonzero_len_cx.build.InBoundsGEP(heap_ptr, v)
            };

        nonzero_len_cx.build.Br(next_cx.llbb);
        // Now we can figure out the length of `v` and get a pointer to its
        // first element.

        let len =
            next_cx.build.Phi(T_int(), ~[stack_len, zero_len, heap_len],
                              ~[bcx.llbb, zero_len_cx.llbb,
                                nonzero_len_cx.llbb]);
        let elem =
            next_cx.build.Phi(T_ptr(llunitty),
                              ~[stack_elem, zero_elem, heap_elem],
                              ~[bcx.llbb, zero_len_cx.llbb,
                                nonzero_len_cx.llbb]);
        ret {len: len, data: elem, bcx: next_cx};
    }

    // Returns a tuple consisting of a pointer to the newly-reserved space and
    // a block context. Updates the length appropriately.
    fn reserve_space(cx: &@block_ctxt, llunitty: TypeRef, v: ValueRef,
                     len_needed: ValueRef) -> result {
        let stack_len_ptr =
            cx.build.InBoundsGEP(v, ~[C_int(0), C_uint(abi::ivec_elt_len)]);
        let stack_len = cx.build.Load(stack_len_ptr);
        let alen =
            load_inbounds(cx, v, ~[C_int(0), C_uint(abi::ivec_elt_alen)]);
        // There are four cases we have to consider:
        // (1) On heap, no resize necessary.
        // (2) On heap, need to resize.
        // (3) On stack, no resize necessary.
        // (4) On stack, need to spill to heap.

        let maybe_on_heap =
            cx.build.ICmp(lib::llvm::LLVMIntEQ, stack_len, C_int(0));
        let maybe_on_heap_cx = new_sub_block_ctxt(cx, "maybe_on_heap");
        let on_stack_cx = new_sub_block_ctxt(cx, "on_stack");
        cx.build.CondBr(maybe_on_heap, maybe_on_heap_cx.llbb,
                        on_stack_cx.llbb);
        let next_cx = new_sub_block_ctxt(cx, "next");
        // We're possibly on the heap, unless the vector is zero-length.

        let stub_p = ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];
        let stub_ptr =
            maybe_on_heap_cx.build.PointerCast(v,
                                               T_ptr(T_ivec_heap(llunitty)));
        let heap_ptr = load_inbounds(maybe_on_heap_cx, stub_ptr, stub_p);
        let on_heap =
            maybe_on_heap_cx.build.ICmp(lib::llvm::LLVMIntNE, heap_ptr,
                                        C_null(val_ty(heap_ptr)));
        let on_heap_cx = new_sub_block_ctxt(cx, "on_heap");
        maybe_on_heap_cx.build.CondBr(on_heap, on_heap_cx.llbb,
                                      on_stack_cx.llbb);
        // We're definitely on the heap. Check whether we need to resize.

        let heap_len_ptr =
            on_heap_cx.build.InBoundsGEP(heap_ptr,
                                         ~[C_int(0),
                                           C_uint(abi::ivec_heap_elt_len)]);
        let heap_len = on_heap_cx.build.Load(heap_len_ptr);
        let new_heap_len = on_heap_cx.build.Add(heap_len, len_needed);
        let heap_len_unscaled =
            on_heap_cx.build.UDiv(heap_len, llsize_of(llunitty));
        let heap_no_resize_needed =
            on_heap_cx.build.ICmp(lib::llvm::LLVMIntULE, new_heap_len, alen);
        let heap_no_resize_cx = new_sub_block_ctxt(cx, "heap_no_resize");
        let heap_resize_cx = new_sub_block_ctxt(cx, "heap_resize");
        on_heap_cx.build.CondBr(heap_no_resize_needed, heap_no_resize_cx.llbb,
                                heap_resize_cx.llbb);
        // Case (1): We're on the heap and don't need to resize.

        let heap_data_no_resize =
            {
                let v =
                    ~[C_int(0), C_uint(abi::ivec_heap_elt_elems),
                      heap_len_unscaled];
                heap_no_resize_cx.build.InBoundsGEP(heap_ptr, v)
            };
        heap_no_resize_cx.build.Store(new_heap_len, heap_len_ptr);
        heap_no_resize_cx.build.Br(next_cx.llbb);
        // Case (2): We're on the heap and need to resize. This path is rare,
        // so we delegate to cold glue.

        {
            let p =
                heap_resize_cx.build.PointerCast(v, T_ptr(T_opaque_ivec()));
            let upcall = bcx_ccx(cx).upcalls.ivec_resize_shared;
            heap_resize_cx.build.Call(upcall,
                                      ~[cx.fcx.lltaskptr, p, new_heap_len]);
        }
        let heap_ptr_resize = load_inbounds(heap_resize_cx, stub_ptr, stub_p);

        let heap_data_resize =
            {
                let v =
                    ~[C_int(0), C_uint(abi::ivec_heap_elt_elems),
                      heap_len_unscaled];
                heap_resize_cx.build.InBoundsGEP(heap_ptr_resize, v)
            };
        heap_resize_cx.build.Br(next_cx.llbb);
        // We're on the stack. Check whether we need to spill to the heap.

        let new_stack_len = on_stack_cx.build.Add(stack_len, len_needed);
        let stack_no_spill_needed =
            on_stack_cx.build.ICmp(lib::llvm::LLVMIntULE, new_stack_len,
                                   alen);
        let stack_len_unscaled =
            on_stack_cx.build.UDiv(stack_len, llsize_of(llunitty));
        let stack_no_spill_cx = new_sub_block_ctxt(cx, "stack_no_spill");
        let stack_spill_cx = new_sub_block_ctxt(cx, "stack_spill");
        on_stack_cx.build.CondBr(stack_no_spill_needed,
                                 stack_no_spill_cx.llbb, stack_spill_cx.llbb);
        // Case (3): We're on the stack and don't need to spill.

        let stack_data_no_spill =
            stack_no_spill_cx.build.InBoundsGEP(v,
                                                ~[C_int(0),
                                                  C_uint(abi::ivec_elt_elems),
                                                  stack_len_unscaled]);
        stack_no_spill_cx.build.Store(new_stack_len, stack_len_ptr);
        stack_no_spill_cx.build.Br(next_cx.llbb);
        // Case (4): We're on the stack and need to spill. Like case (2), this
        // path is rare, so we delegate to cold glue.

        {
            let p =
                stack_spill_cx.build.PointerCast(v, T_ptr(T_opaque_ivec()));
            let upcall = bcx_ccx(cx).upcalls.ivec_spill_shared;
            stack_spill_cx.build.Call(upcall,
                                      ~[cx.fcx.lltaskptr, p, new_stack_len]);
        }
        let spill_stub =
            stack_spill_cx.build.PointerCast(v, T_ptr(T_ivec_heap(llunitty)));

        let heap_ptr_spill =
            load_inbounds(stack_spill_cx, spill_stub, stub_p);

        let heap_data_spill =
            {
                let v =
                    ~[C_int(0), C_uint(abi::ivec_heap_elt_elems),
                      stack_len_unscaled];
                stack_spill_cx.build.InBoundsGEP(heap_ptr_spill, v)
            };
        stack_spill_cx.build.Br(next_cx.llbb);
        // Phi together the different data pointers to get the result.

        let data_ptr =
            next_cx.build.Phi(T_ptr(llunitty),
                              ~[heap_data_no_resize, heap_data_resize,
                                stack_data_no_spill, heap_data_spill],
                              ~[heap_no_resize_cx.llbb, heap_resize_cx.llbb,
                                stack_no_spill_cx.llbb, stack_spill_cx.llbb]);
        ret rslt(next_cx, data_ptr);
    }
    fn trans_append(cx: &@block_ctxt, t: &ty::t, orig_lhs: ValueRef,
                    orig_rhs: ValueRef) -> result {
        // Cast to opaque interior vector types if necessary.
        let lhs;
        let rhs;
        if ty::type_has_dynamic_size(bcx_tcx(cx), t) {
            lhs = cx.build.PointerCast(orig_lhs, T_ptr(T_opaque_ivec()));
            rhs = cx.build.PointerCast(orig_rhs, T_ptr(T_opaque_ivec()));
        } else { lhs = orig_lhs; rhs = orig_rhs; }

        let unit_ty = ty::sequence_element_type(bcx_tcx(cx), t);
        let llunitty = type_of_or_i8(cx, unit_ty);
        alt ty::struct(bcx_tcx(cx), t) {
          ty::ty_istr. { }
          ty::ty_ivec(_) { }
          _ { bcx_tcx(cx).sess.bug("non-istr/ivec in trans_append"); }
        }

        let rs = size_of(cx, unit_ty);
        let bcx = rs.bcx;
        let unit_sz = rs.val;

        // Gather the various type descriptors we'll need.

        // FIXME (issue #511): This is needed to prevent a leak.
        let no_tydesc_info = none;

        rs = get_tydesc(bcx, t, false, no_tydesc_info);
        bcx = rs.bcx;
        rs = get_tydesc(bcx, unit_ty, false, no_tydesc_info);
        bcx = rs.bcx;
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_copy_glue, none);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, none);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, none);
        let rhs_len_and_data = get_len_and_data(bcx, rhs, unit_ty);
        let rhs_len = rhs_len_and_data.len;
        let rhs_data = rhs_len_and_data.data;
        bcx = rhs_len_and_data.bcx;
        rs = reserve_space(bcx, llunitty, lhs, rhs_len);
        let lhs_data = rs.val;
        bcx = rs.bcx;
        // Work out the end pointer.

        let lhs_unscaled_idx = bcx.build.UDiv(rhs_len, llsize_of(llunitty));
        let lhs_end = bcx.build.InBoundsGEP(lhs_data, ~[lhs_unscaled_idx]);
        // Now emit the copy loop.

        let dest_ptr = alloca(bcx, T_ptr(llunitty));
        bcx.build.Store(lhs_data, dest_ptr);
        let src_ptr = alloca(bcx, T_ptr(llunitty));
        bcx.build.Store(rhs_data, src_ptr);
        let copy_loop_header_cx = new_sub_block_ctxt(bcx, "copy_loop_header");
        bcx.build.Br(copy_loop_header_cx.llbb);
        let copy_dest_ptr = copy_loop_header_cx.build.Load(dest_ptr);
        let not_yet_at_end =
            copy_loop_header_cx.build.ICmp(lib::llvm::LLVMIntNE,
                                           copy_dest_ptr, lhs_end);
        let copy_loop_body_cx = new_sub_block_ctxt(bcx, "copy_loop_body");
        let next_cx = new_sub_block_ctxt(bcx, "next");
        copy_loop_header_cx.build.CondBr(not_yet_at_end,
                                         copy_loop_body_cx.llbb,
                                         next_cx.llbb);

        let copy_src_ptr = copy_loop_body_cx.build.Load(src_ptr);
        let copy_src =
            load_if_immediate(copy_loop_body_cx, copy_src_ptr, unit_ty);

        rs =
            copy_val(copy_loop_body_cx, INIT, copy_dest_ptr, copy_src,
                     unit_ty);
        let post_copy_cx = rs.bcx;
        // Increment both pointers.
        if ty::type_has_dynamic_size(bcx_tcx(cx), t) {
            // We have to increment by the dynamically-computed size.
            incr_ptr(post_copy_cx, copy_dest_ptr, unit_sz, dest_ptr);
            incr_ptr(post_copy_cx, copy_src_ptr, unit_sz, src_ptr);
        } else {
            incr_ptr(post_copy_cx, copy_dest_ptr, C_int(1), dest_ptr);
            incr_ptr(post_copy_cx, copy_src_ptr, C_int(1), src_ptr);
        }

        post_copy_cx.build.Br(copy_loop_header_cx.llbb);
        ret rslt(next_cx, C_nil());
    }

    type alloc_result =
        {bcx: @block_ctxt,
         llptr: ValueRef,
         llunitsz: ValueRef,
         llalen: ValueRef};

    fn alloc(cx: &@block_ctxt, unit_ty: ty::t) -> alloc_result {
        let dynamic = ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty);

        let bcx;
        if dynamic {
            bcx = llderivedtydescs_block_ctxt(cx.fcx);
        } else { bcx = cx; }

        let llunitsz;
        let rslt = size_of(bcx, unit_ty);
        bcx = rslt.bcx;
        llunitsz = rslt.val;

        if dynamic { cx.fcx.llderivedtydescs = bcx.llbb; }

        let llalen =
            bcx.build.Mul(llunitsz, C_uint(abi::ivec_default_length));

        let llptr;
        let llunitty = type_of_or_i8(bcx, unit_ty);
        let bcx_result;
        if dynamic {
            let llarraysz = bcx.build.Add(llsize_of(T_opaque_ivec()), llalen);
            let llvecptr = array_alloca(bcx, T_i8(), llarraysz);

            bcx_result = cx;
            llptr =
                bcx_result.build.PointerCast(llvecptr,
                                             T_ptr(T_opaque_ivec()));
        } else { llptr = alloca(bcx, T_ivec(llunitty)); bcx_result = bcx; }

        ret {bcx: bcx_result,
             llptr: llptr,
             llunitsz: llunitsz,
             llalen: llalen};
    }

    fn trans_add(cx: &@block_ctxt, vec_ty: ty::t, lhs: ValueRef,
                 rhs: ValueRef) -> result {
        let bcx = cx;
        let unit_ty = ty::sequence_element_type(bcx_tcx(bcx), vec_ty);

        let ares = alloc(bcx, unit_ty);
        bcx = ares.bcx;
        let llvecptr = ares.llptr;
        let unit_sz = ares.llunitsz;
        let llalen = ares.llalen;

        add_clean_temp(bcx, llvecptr, vec_ty);

        let llunitty = type_of_or_i8(bcx, unit_ty);
        let llheappartty = T_ivec_heap_part(llunitty);
        let lhs_len_and_data = get_len_and_data(bcx, lhs, unit_ty);
        let lhs_len = lhs_len_and_data.len;
        let lhs_data = lhs_len_and_data.data;
        bcx = lhs_len_and_data.bcx;
        let rhs_len_and_data = get_len_and_data(bcx, rhs, unit_ty);
        let rhs_len = rhs_len_and_data.len;
        let rhs_data = rhs_len_and_data.data;
        bcx = rhs_len_and_data.bcx;
        let lllen = bcx.build.Add(lhs_len, rhs_len);
        // We have three cases to handle here:
        // (1) Length is zero ([] + []).
        // (2) Copy onto stack.
        // (3) Allocate on heap and copy there.

        let len_is_zero =
            bcx.build.ICmp(lib::llvm::LLVMIntEQ, lllen, C_int(0));
        let zero_len_cx = new_sub_block_ctxt(bcx, "zero_len");
        let nonzero_len_cx = new_sub_block_ctxt(bcx, "nonzero_len");
        bcx.build.CondBr(len_is_zero, zero_len_cx.llbb, nonzero_len_cx.llbb);
        // Case (1): Length is zero.

        let stub_z = ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_zero)];
        let stub_a = ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_alen)];
        let stub_p = ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];

        let vec_l = ~[C_int(0), C_uint(abi::ivec_elt_len)];
        let vec_a = ~[C_int(0), C_uint(abi::ivec_elt_alen)];

        let stub_ptr_zero =
            zero_len_cx.build.PointerCast(llvecptr,
                                          T_ptr(T_ivec_heap(llunitty)));
        zero_len_cx.build.Store(C_int(0),
                                zero_len_cx.build.InBoundsGEP(stub_ptr_zero,
                                                              stub_z));
        zero_len_cx.build.Store(llalen,
                                zero_len_cx.build.InBoundsGEP(stub_ptr_zero,
                                                              stub_a));
        zero_len_cx.build.Store(C_null(T_ptr(llheappartty)),
                                zero_len_cx.build.InBoundsGEP(stub_ptr_zero,
                                                              stub_p));
        let next_cx = new_sub_block_ctxt(bcx, "next");
        zero_len_cx.build.Br(next_cx.llbb);
        // Determine whether we need to spill to the heap.

        let on_stack =
            nonzero_len_cx.build.ICmp(lib::llvm::LLVMIntULE, lllen, llalen);
        let stack_cx = new_sub_block_ctxt(bcx, "stack");
        let heap_cx = new_sub_block_ctxt(bcx, "heap");
        nonzero_len_cx.build.CondBr(on_stack, stack_cx.llbb, heap_cx.llbb);
        // Case (2): Copy onto stack.

        stack_cx.build.Store(lllen,
                             stack_cx.build.InBoundsGEP(llvecptr, vec_l));
        stack_cx.build.Store(llalen,
                             stack_cx.build.InBoundsGEP(llvecptr, vec_a));
        let dest_ptr_stack =
            stack_cx.build.InBoundsGEP(llvecptr,
                                       ~[C_int(0),
                                         C_uint(abi::ivec_elt_elems),
                                         C_int(0)]);
        let copy_cx = new_sub_block_ctxt(bcx, "copy");
        stack_cx.build.Br(copy_cx.llbb);
        // Case (3): Allocate on heap and copy there.

        let stub_ptr_heap =
            heap_cx.build.PointerCast(llvecptr, T_ptr(T_ivec_heap(llunitty)));
        heap_cx.build.Store(C_int(0),
                            heap_cx.build.InBoundsGEP(stub_ptr_heap, stub_z));
        heap_cx.build.Store(lllen,
                            heap_cx.build.InBoundsGEP(stub_ptr_heap, stub_a));
        let heap_sz = heap_cx.build.Add(llsize_of(llheappartty), lllen);
        let rs = trans_shared_malloc(heap_cx, T_ptr(llheappartty), heap_sz);
        let heap_part = rs.val;
        heap_cx = rs.bcx;
        heap_cx.build.Store(heap_part,
                            heap_cx.build.InBoundsGEP(stub_ptr_heap, stub_p));
        {
            let v = ~[C_int(0), C_uint(abi::ivec_heap_elt_len)];
            heap_cx.build.Store(lllen,
                                heap_cx.build.InBoundsGEP(heap_part, v));
        }
        let dest_ptr_heap =
            heap_cx.build.InBoundsGEP(heap_part,
                                      ~[C_int(0),
                                        C_uint(abi::ivec_heap_elt_elems),
                                        C_int(0)]);
        heap_cx.build.Br(copy_cx.llbb);
        // Emit the copy loop.

        let first_dest_ptr =
            copy_cx.build.Phi(T_ptr(llunitty),
                              ~[dest_ptr_stack, dest_ptr_heap],
                              ~[stack_cx.llbb, heap_cx.llbb]);

        let lhs_end_ptr;
        let rhs_end_ptr;
        if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
            lhs_end_ptr = copy_cx.build.InBoundsGEP(lhs_data, ~[lhs_len]);
            rhs_end_ptr = copy_cx.build.InBoundsGEP(rhs_data, ~[rhs_len]);
        } else {
            let lhs_len_unscaled = copy_cx.build.UDiv(lhs_len, unit_sz);
            lhs_end_ptr =
                copy_cx.build.InBoundsGEP(lhs_data, ~[lhs_len_unscaled]);
            let rhs_len_unscaled = copy_cx.build.UDiv(rhs_len, unit_sz);
            rhs_end_ptr =
                copy_cx.build.InBoundsGEP(rhs_data, ~[rhs_len_unscaled]);
        }

        let dest_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
        copy_cx.build.Store(first_dest_ptr, dest_ptr_ptr);
        let lhs_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
        copy_cx.build.Store(lhs_data, lhs_ptr_ptr);
        let rhs_ptr_ptr = alloca(copy_cx, T_ptr(llunitty));
        copy_cx.build.Store(rhs_data, rhs_ptr_ptr);
        let lhs_copy_cx = new_sub_block_ctxt(bcx, "lhs_copy");
        copy_cx.build.Br(lhs_copy_cx.llbb);
        // Copy in elements from the LHS.

        let lhs_ptr = lhs_copy_cx.build.Load(lhs_ptr_ptr);
        let not_at_end_lhs =
            lhs_copy_cx.build.ICmp(lib::llvm::LLVMIntNE, lhs_ptr,
                                   lhs_end_ptr);
        let lhs_do_copy_cx = new_sub_block_ctxt(bcx, "lhs_do_copy");
        let rhs_copy_cx = new_sub_block_ctxt(bcx, "rhs_copy");
        lhs_copy_cx.build.CondBr(not_at_end_lhs, lhs_do_copy_cx.llbb,
                                 rhs_copy_cx.llbb);
        let dest_ptr_lhs_copy = lhs_do_copy_cx.build.Load(dest_ptr_ptr);
        let lhs_val = load_if_immediate(lhs_do_copy_cx, lhs_ptr, unit_ty);
        rs =
            copy_val(lhs_do_copy_cx, INIT, dest_ptr_lhs_copy, lhs_val,
                     unit_ty);
        lhs_do_copy_cx = rs.bcx;

        // Increment both pointers.
        if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
            // We have to increment by the dynamically-computed size.
            incr_ptr(lhs_do_copy_cx, dest_ptr_lhs_copy, unit_sz,
                     dest_ptr_ptr);
            incr_ptr(lhs_do_copy_cx, lhs_ptr, unit_sz, lhs_ptr_ptr);
        } else {
            incr_ptr(lhs_do_copy_cx, dest_ptr_lhs_copy, C_int(1),
                     dest_ptr_ptr);
            incr_ptr(lhs_do_copy_cx, lhs_ptr, C_int(1), lhs_ptr_ptr);
        }

        lhs_do_copy_cx.build.Br(lhs_copy_cx.llbb);
        // Copy in elements from the RHS.

        let rhs_ptr = rhs_copy_cx.build.Load(rhs_ptr_ptr);
        let not_at_end_rhs =
            rhs_copy_cx.build.ICmp(lib::llvm::LLVMIntNE, rhs_ptr,
                                   rhs_end_ptr);
        let rhs_do_copy_cx = new_sub_block_ctxt(bcx, "rhs_do_copy");
        rhs_copy_cx.build.CondBr(not_at_end_rhs, rhs_do_copy_cx.llbb,
                                 next_cx.llbb);
        let dest_ptr_rhs_copy = rhs_do_copy_cx.build.Load(dest_ptr_ptr);
        let rhs_val = load_if_immediate(rhs_do_copy_cx, rhs_ptr, unit_ty);
        rs =
            copy_val(rhs_do_copy_cx, INIT, dest_ptr_rhs_copy, rhs_val,
                     unit_ty);
        rhs_do_copy_cx = rs.bcx;

        // Increment both pointers.
        if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
            // We have to increment by the dynamically-computed size.
            incr_ptr(rhs_do_copy_cx, dest_ptr_rhs_copy, unit_sz,
                     dest_ptr_ptr);
            incr_ptr(rhs_do_copy_cx, rhs_ptr, unit_sz, rhs_ptr_ptr);
        } else {
            incr_ptr(rhs_do_copy_cx, dest_ptr_rhs_copy, C_int(1),
                     dest_ptr_ptr);
            incr_ptr(rhs_do_copy_cx, rhs_ptr, C_int(1), rhs_ptr_ptr);
        }

        rhs_do_copy_cx.build.Br(rhs_copy_cx.llbb);
        // Finally done!

        ret rslt(next_cx, llvecptr);
    }

    // NB: This does *not* adjust reference counts. The caller must have done
    // this via copy_ty() beforehand.
    fn duplicate_heap_part(cx: &@block_ctxt, orig_vptr: ValueRef,
                           unit_ty: ty::t) -> result {
        // Cast to an opaque interior vector if we can't trust the pointer
        // type.
        let vptr;
        if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
            vptr = cx.build.PointerCast(orig_vptr, T_ptr(T_opaque_ivec()));
        } else { vptr = orig_vptr; }

        let llunitty = type_of_or_i8(cx, unit_ty);
        let llheappartty = T_ivec_heap_part(llunitty);

        // Check to see if the vector is heapified.
        let stack_len_ptr =
            cx.build.InBoundsGEP(vptr,
                                 ~[C_int(0), C_uint(abi::ivec_elt_len)]);
        let stack_len = cx.build.Load(stack_len_ptr);
        let stack_len_is_zero =
            cx.build.ICmp(lib::llvm::LLVMIntEQ, stack_len, C_int(0));
        let maybe_on_heap_cx = new_sub_block_ctxt(cx, "maybe_on_heap");
        let next_cx = new_sub_block_ctxt(cx, "next");
        cx.build.CondBr(stack_len_is_zero, maybe_on_heap_cx.llbb,
                        next_cx.llbb);

        let stub_ptr =
            maybe_on_heap_cx.build.PointerCast(vptr,
                                               T_ptr(T_ivec_heap(llunitty)));
        let heap_ptr_ptr =
            maybe_on_heap_cx.build.InBoundsGEP
            (stub_ptr, ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)]);
        let heap_ptr = maybe_on_heap_cx.build.Load(heap_ptr_ptr);
        let heap_ptr_is_nonnull =
            maybe_on_heap_cx.build.ICmp(lib::llvm::LLVMIntNE, heap_ptr,
                                        C_null(T_ptr(llheappartty)));
        let on_heap_cx = new_sub_block_ctxt(cx, "on_heap");
        maybe_on_heap_cx.build.CondBr(heap_ptr_is_nonnull, on_heap_cx.llbb,
                                      next_cx.llbb);

        // Ok, the vector is on the heap. Copy the heap part.
        let alen_ptr =
            on_heap_cx.build.InBoundsGEP
            (stub_ptr, ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_alen)]);
        let alen = on_heap_cx.build.Load(alen_ptr);

        let heap_part_sz =
            on_heap_cx.build.Add(alen, llsize_of(T_opaque_ivec_heap_part()));
        let rs =
            trans_shared_malloc(on_heap_cx, T_ptr(llheappartty),
                                heap_part_sz);
        on_heap_cx = rs.bcx;
        let new_heap_ptr = rs.val;

        rs = call_memmove(on_heap_cx, new_heap_ptr, heap_ptr, heap_part_sz);
        on_heap_cx = rs.bcx;

        on_heap_cx.build.Store(new_heap_ptr, heap_ptr_ptr);
        on_heap_cx.build.Br(next_cx.llbb);

        ret rslt(next_cx, C_nil());
    }
}

fn trans_vec_add(cx: &@block_ctxt, t: &ty::t, lhs: ValueRef, rhs: ValueRef) ->
   result {
    let r = alloc_ty(cx, t);
    let tmp = r.val;
    r = copy_val(r.bcx, INIT, tmp, lhs, t);
    let bcx = trans_vec_append(r.bcx, t, tmp, rhs).bcx;
    tmp = load_if_immediate(bcx, tmp, t);
    add_clean_temp(cx, tmp, t);
    ret rslt(bcx, tmp);
}

// Important to get types for both lhs and rhs, because one might be _|_
// and the other not.
fn trans_eager_binop(cx: &@block_ctxt, op: ast::binop, lhs: ValueRef,
                     lhs_t: ty::t, rhs: ValueRef, rhs_t: ty::t) -> result {

    // If either is bottom, it diverges. So no need to do the
    // operation.
    if (ty::type_is_bot(bcx_tcx(cx), lhs_t) ||
        ty::type_is_bot(bcx_tcx(cx), rhs_t)) {
        ret rslt(cx, cx.build.Unreachable());
    }

    let is_float = false;
    let intype = lhs_t;
    if ty::type_is_bot(bcx_tcx(cx), intype) {
        intype = rhs_t;
    }

    alt ty::struct(bcx_tcx(cx), intype) {
      ty::ty_float. { is_float = true; }
      _ { is_float = false; }
    }
    alt op {
      ast::add. {
        if ty::type_is_sequence(bcx_tcx(cx), intype) {
            if ty::sequence_is_interior(bcx_tcx(cx), intype) {
                ret ivec::trans_add(cx, intype, lhs, rhs);
            }
            ret trans_vec_add(cx, intype, lhs, rhs);
        }
        if is_float {
            ret rslt(cx, cx.build.FAdd(lhs, rhs));
        } else { ret rslt(cx, cx.build.Add(lhs, rhs)); }
      }
      ast::sub. {
        if is_float {
            ret rslt(cx, cx.build.FSub(lhs, rhs));
        } else { ret rslt(cx, cx.build.Sub(lhs, rhs)); }
      }
      ast::mul. {
        if is_float {
            ret rslt(cx, cx.build.FMul(lhs, rhs));
        } else { ret rslt(cx, cx.build.Mul(lhs, rhs)); }
      }
      ast::div. {
        if is_float { ret rslt(cx, cx.build.FDiv(lhs, rhs)); }
        if ty::type_is_signed(bcx_tcx(cx), intype) {
            ret rslt(cx, cx.build.SDiv(lhs, rhs));
        } else { ret rslt(cx, cx.build.UDiv(lhs, rhs)); }
      }
      ast::rem. {
        if is_float { ret rslt(cx, cx.build.FRem(lhs, rhs)); }
        if ty::type_is_signed(bcx_tcx(cx), intype) {
            ret rslt(cx, cx.build.SRem(lhs, rhs));
        } else { ret rslt(cx, cx.build.URem(lhs, rhs)); }
      }
      ast::bitor. { ret rslt(cx, cx.build.Or(lhs, rhs)); }
      ast::bitand. { ret rslt(cx, cx.build.And(lhs, rhs)); }
      ast::bitxor. { ret rslt(cx, cx.build.Xor(lhs, rhs)); }
      ast::lsl. { ret rslt(cx, cx.build.Shl(lhs, rhs)); }
      ast::lsr. { ret rslt(cx, cx.build.LShr(lhs, rhs)); }
      ast::asr. { ret rslt(cx, cx.build.AShr(lhs, rhs)); }
      _ { ret trans_compare(cx, op, lhs, lhs_t, rhs, rhs_t); }
    }
}

fn autoderef(cx: &@block_ctxt, v: ValueRef, t: &ty::t) -> result_t {
    let v1: ValueRef = v;
    let t1: ty::t = t;
    let ccx = bcx_ccx(cx);
    while true {
        alt ty::struct(ccx.tcx, t1) {
          ty::ty_box(mt) {
            let body =
                cx.build.GEP(v1, ~[C_int(0), C_int(abi::box_rc_field_body)]);
            t1 = mt.ty;

            // Since we're changing levels of box indirection, we may have
            // to cast this pointer, since statically-sized tag types have
            // different types depending on whether they're behind a box
            // or not.
            if !ty::type_has_dynamic_size(ccx.tcx, mt.ty) {
                let llty = type_of(ccx, cx.sp, mt.ty);
                v1 = cx.build.PointerCast(body, T_ptr(llty));
            } else { v1 = body; }
          }
          ty::ty_res(did, inner, tps) {
            t1 = ty::substitute_type_params(ccx.tcx, tps, inner);
            v1 = cx.build.GEP(v1, ~[C_int(0), C_int(1)]);
          }
          ty::ty_tag(did, tps) {
            let variants = ty::tag_variants(ccx.tcx, did);
            if std::ivec::len(variants) != 1u ||
                   std::ivec::len(variants.(0).args) != 1u {
                break;
            }
            t1 =
                ty::substitute_type_params(ccx.tcx, tps,
                                           variants.(0).args.(0));
            if !ty::type_has_dynamic_size(ccx.tcx, t1) {
                v1 = cx.build.PointerCast(v1, T_ptr(type_of(ccx, cx.sp, t1)));
            }
          }
          _ { break; }
        }
        v1 = load_if_immediate(cx, v1, t1);
    }
    ret {bcx: cx, val: v1, ty: t1};
}

fn trans_binary(cx: &@block_ctxt, op: ast::binop, a: &@ast::expr,
                b: &@ast::expr) -> result {


    // First couple cases are lazy:
    alt op {
      ast::and. {
        // Lazy-eval and
        let lhs_expr = trans_expr(cx, a);
        let lhs_res =
            autoderef(lhs_expr.bcx, lhs_expr.val,
                      ty::expr_ty(bcx_tcx(cx), a));
        let rhs_cx = new_scope_block_ctxt(cx, "rhs");
        let rhs_expr = trans_expr(rhs_cx, b);
        let rhs_res =
            autoderef(rhs_expr.bcx, rhs_expr.val,
                      ty::expr_ty(bcx_tcx(cx), b));
        let lhs_false_cx = new_scope_block_ctxt(cx, "lhs false");
        let lhs_false_res = rslt(lhs_false_cx, C_bool(false));
        // The following line ensures that any cleanups for rhs
        // are done within the block for rhs. This is necessary
        // because and/or are lazy. So the rhs may never execute,
        // and the cleanups can't be pushed into later code.

        let rhs_bcx = trans_block_cleanups(rhs_res.bcx, rhs_cx);
        lhs_res.bcx.build.CondBr(lhs_res.val, rhs_cx.llbb, lhs_false_cx.llbb);
        ret join_results(cx, T_bool(),
                         ~[lhs_false_res, {bcx: rhs_bcx, val: rhs_res.val}]);
      }
      ast::or. {
        // Lazy-eval or
        let lhs_expr = trans_expr(cx, a);
        let lhs_res =
            autoderef(lhs_expr.bcx, lhs_expr.val,
                      ty::expr_ty(bcx_tcx(cx), a));
        let rhs_cx = new_scope_block_ctxt(cx, "rhs");
        let rhs_expr = trans_expr(rhs_cx, b);
        let rhs_res =
            autoderef(rhs_expr.bcx, rhs_expr.val,
                      ty::expr_ty(bcx_tcx(cx), b));
        let lhs_true_cx = new_scope_block_ctxt(cx, "lhs true");
        let lhs_true_res = rslt(lhs_true_cx, C_bool(true));
        // see the and case for an explanation

        let rhs_bcx = trans_block_cleanups(rhs_res.bcx, rhs_cx);
        lhs_res.bcx.build.CondBr(lhs_res.val, lhs_true_cx.llbb, rhs_cx.llbb);
        ret join_results(cx, T_bool(),
                         ~[lhs_true_res, {bcx: rhs_bcx, val: rhs_res.val}]);
      }
      _ {
        // Remaining cases are eager:

        let lhs_expr = trans_expr(cx, a);
        let lhty = ty::expr_ty(bcx_tcx(cx), a);
        let lhs = autoderef(lhs_expr.bcx, lhs_expr.val, lhty);
        let rhs_expr = trans_expr(lhs.bcx, b);
        let rhty = ty::expr_ty(bcx_tcx(cx), b);
        let rhs = autoderef(rhs_expr.bcx, rhs_expr.val, rhty);

        ret trans_eager_binop(rhs.bcx, op, lhs.val, lhs.ty, rhs.val, rhs.ty);
      }
    }
}

fn join_results(parent_cx: &@block_ctxt, t: TypeRef, ins: &result[]) ->
   result {
    let live: result[] = ~[];
    let vals: ValueRef[] = ~[];
    let bbs: BasicBlockRef[] = ~[];
    for r: result  in ins {
        if !is_terminated(r.bcx) {
            live += ~[r];
            vals += ~[r.val];
            bbs += ~[r.bcx.llbb];
        }
    }
    alt std::ivec::len[result](live) {
      0u {
        // No incoming edges are live, so we're in dead-code-land.
        // Arbitrarily pick the first dead edge, since the caller
        // is just going to propagate it outward.

        assert (std::ivec::len[result](ins) >= 1u);
        ret ins.(0);
      }
      _ {/* fall through */ }
    }
    // We have >1 incoming edges. Make a join block and br+phi them into it.

    let join_cx = new_sub_block_ctxt(parent_cx, "join");
    for r: result  in live { r.bcx.build.Br(join_cx.llbb); }
    let phi = join_cx.build.Phi(t, vals, bbs);
    ret rslt(join_cx, phi);
}

fn join_branches(parent_cx: &@block_ctxt, ins: &result[]) -> @block_ctxt {
    let out = new_sub_block_ctxt(parent_cx, "join");
    for r: result  in ins {
        if !is_terminated(r.bcx) { r.bcx.build.Br(out.llbb); }
    }
    ret out;
}

tag out_method { return; save_in(ValueRef); }

fn trans_if(cx: &@block_ctxt, cond: &@ast::expr, thn: &ast::blk,
            els: &option::t[@ast::expr], id: ast::node_id,
            output: &out_method) -> result {
    let cond_res = trans_expr(cx, cond);

    if (ty::type_is_bot(bcx_tcx(cx), ty::expr_ty(bcx_tcx(cx), cond))) {
        // No need to generate code for comparison,
        // since the cond diverges.
        if (!cx.build.is_terminated()) {
            ret rslt(cx, cx.build.Unreachable());
        }
        else {
            ret cond_res;
        }
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
                let elseif_blk = ast::block_from_expr(elexpr);
                trans_block(else_cx, elseif_blk, output)
              }
              ast::expr_block(blk) { trans_block(else_cx, blk, output) }
            }
          }
          _ { rslt(else_cx, C_nil()) }
        };
    cond_res.bcx.build.CondBr(cond_res.val, then_cx.llbb, else_cx.llbb);
    ret rslt(join_branches(cx, ~[then_res, else_res]), C_nil());
}

fn trans_for(cx: &@block_ctxt, local: &@ast::local, seq: &@ast::expr,
             body: &ast::blk) -> result {
    // FIXME: We bind to an alias here to avoid a segfault... this is
    // obviously a bug.
    fn inner(cx: &@block_ctxt, local: @ast::local, curr: ValueRef, t: ty::t,
             body: &ast::blk, outer_next_cx: @block_ctxt) -> result {
        let next_cx = new_sub_block_ctxt(cx, "next");
        let scope_cx =
            new_loop_scope_block_ctxt(cx, option::some[@block_ctxt](next_cx),
                                      outer_next_cx, "for loop scope");
        cx.build.Br(scope_cx.llbb);
        let local_res = alloc_local(scope_cx, local);
        let loc_r = copy_val(local_res.bcx, INIT, local_res.val, curr, t);
        add_clean(scope_cx, local_res.val, t);
        let bcx = trans_alt::bind_irrefutable_pat
           (loc_r.bcx, local.node.pat, local_res.val, cx.fcx.lllocals, false);
        bcx = trans_block(bcx, body, return).bcx;
        if !bcx.build.is_terminated() {
            bcx.build.Br(next_cx.llbb);
            // otherwise, this code is unreachable
        }
        ret rslt(next_cx, C_nil());
    }
    let next_cx = new_sub_block_ctxt(cx, "next");
    let seq_ty = ty::expr_ty(bcx_tcx(cx), seq);
    let seq_res = trans_expr(cx, seq);
    let it =
        iter_sequence(seq_res.bcx, seq_res.val, seq_ty,
                      bind inner(_, local, _, _, body, next_cx));
    it.bcx.build.Br(next_cx.llbb);
    ret rslt(next_cx, it.val);
}


// Iterator translation

// Given a block context and a list of tydescs and values to bind
// construct a closure out of them. If copying is true, it is a
// heap allocated closure that copies the upvars into environment.
// Otherwise, it is stack allocated and copies pointers to the upvars.
fn build_environment(bcx: @block_ctxt, lltydescs: ValueRef[],
                     bound_tys: ty::t[], bound_vals: lval_result[],
                     copying: bool)
    -> {ptr: ValueRef, ptrty: ty::t, bcx: @block_ctxt} {
    // Synthesize a closure type.

    // First, synthesize a tuple type containing the types of all the
    // bound expressions.
    // bindings_ty = ~[bound_ty1, bound_ty2, ...]
    let bindings_ty: ty::t = ty::mk_imm_tup(bcx_tcx(bcx), bound_tys);

    // NB: keep this in sync with T_closure_ptr; we're making
    // a ty::t structure that has the same "shape" as the LLVM type
    // it constructs.

    // Make a vector that contains ty_param_count copies of tydesc_ty.
    // (We'll need room for that many tydescs in the closure.)
    let ty_param_count = std::ivec::len(lltydescs);
    let tydesc_ty: ty::t = ty::mk_type(bcx_tcx(bcx));
    let captured_tys: ty::t[] =
        std::ivec::init_elt(tydesc_ty, ty_param_count);

    // Get all the types we've got (some of which we synthesized
    // ourselves) into a vector.  The whole things ends up looking
    // like:

    // closure_tys = [tydesc_ty, [bound_ty1, bound_ty2, ...], [tydesc_ty,
    // tydesc_ty, ...]]
    let closure_tys: ty::t[] =
        ~[tydesc_ty, bindings_ty, ty::mk_imm_tup(bcx_tcx(bcx), captured_tys)];

    // Finally, synthesize a type for that whole vector.
    let closure_ty: ty::t = ty::mk_imm_tup(bcx_tcx(bcx), closure_tys);

    // Allocate a box that can hold something closure-sized.
    let r = if copying {
        trans_malloc_boxed(bcx, closure_ty)
    } else {
        // We need to dummy up a box on the stack
        let ty = ty::mk_imm_tup(bcx_tcx(bcx),
                                ~[ty::mk_int(bcx_tcx(bcx)), closure_ty]);
        let r = alloc_ty(bcx, ty);
        let body = GEPi(bcx, r.val, ~[0, abi::box_rc_field_body]);
        {bcx: r.bcx, box: r.val, body: body}
    };
    bcx = r.bcx;
    let closure = r.body;

    // Store bindings tydesc.
    if copying {
        let bound_tydesc = GEPi(bcx, closure, ~[0, abi::closure_elt_tydesc]);
        let ti = none;
        let bindings_tydesc = get_tydesc(bcx, bindings_ty, true, ti);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, ti);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, ti);
        bcx = bindings_tydesc.bcx;
        bcx.build.Store(bindings_tydesc.val, bound_tydesc);
    }

    // Copy expr values into boxed bindings.
    let i = 0u;
    let bindings =
        GEP_tup_like(bcx, closure_ty, closure,
                     ~[0, abi::closure_elt_bindings]);
    bcx = bindings.bcx;
    for lv: lval_result  in bound_vals {
        let bound = GEP_tup_like(bcx, bindings_ty, bindings.val,
                                 ~[0, i as int]);
        bcx = bound.bcx;
        if copying {
            bcx = move_val_if_temp(bcx, INIT,
                                   bound.val, lv, bound_tys.(i)).bcx;
        } else {
            bcx.build.Store(lv.res.val, bound.val);
        }
        i += 1u;
    }

    // If necessary, copy tydescs describing type parameters into the
    // appropriate slot in the closure.
    let ty_params_slot =
        GEP_tup_like(bcx, closure_ty, closure,
                     ~[0, abi::closure_elt_ty_params]);
    bcx = ty_params_slot.bcx;
    i = 0u;
    for td: ValueRef  in lltydescs {
        let ty_param_slot = GEPi(bcx, ty_params_slot.val, ~[0, i as int]);
        bcx.build.Store(td, ty_param_slot);
        i += 1u;
    }

    ret {ptr: r.box, ptrty: closure_ty, bcx: bcx};
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for build_environment.
fn build_closure(cx: &@block_ctxt, upvars: &@ast::node_id[], copying: bool)
    -> {ptr: ValueRef, ptrty: ty::t, bcx: @block_ctxt} {
        let closure_vals: lval_result[] = ~[];
        let closure_tys: ty::t[] = ~[];
        // If we need to, package up the iterator body to call
        if !copying && !option::is_none(cx.fcx.lliterbody) {
            closure_vals += ~[lval_mem(cx, option::get(cx.fcx.lliterbody))];
            closure_tys += ~[option::get(cx.fcx.iterbodyty)];
        }
        // Package up the upvars
        for nid: ast::node_id  in *upvars {
            closure_vals += ~[trans_var(cx, cx.sp, nid)];
            let ty = ty::node_id_to_monotype(bcx_tcx(cx), nid);
            if !copying { ty = ty::mk_mut_ptr(bcx_tcx(cx), ty); }
            closure_tys += ~[ty];
        }

        ret build_environment(cx, cx.fcx.lltydescs,
                              closure_tys, closure_vals, copying);
}

// Return a pointer to the stored typarams in a closure.
// This is awful. Since the size of the bindings stored in the closure might
// be dynamically sized, we can't skip past them to get to the tydescs until
// we have loaded the tydescs. Thus we use the stored size of the bindings
// in the tydesc for the closure to skip over them. Ugh.
fn find_environment_tydescs(bcx: &@block_ctxt, envty: &ty::t,
                            closure: ValueRef) -> ValueRef {
    ret if !ty::type_has_dynamic_size(bcx_tcx(bcx), envty) {
        // If we can find the typarams statically, do it
        GEPi(bcx, closure,
             ~[0, abi::box_rc_field_body, abi::closure_elt_ty_params])
    } else {
        // Ugh. We need to load the size of the bindings out of the
        // closure's tydesc and use that to skip over the bindings.
        let descsty =
            ty::get_element_type(bcx_tcx(bcx), envty,
                                 abi::closure_elt_ty_params as uint);
        let llenv = GEPi(bcx, closure, ~[0, abi::box_rc_field_body]);
        // Load the tydesc and find the size of the body
        let lldesc =
            bcx.build.Load(GEPi(bcx, llenv, ~[0, abi::closure_elt_tydesc]));
        let llsz = bcx.build.Load(
            GEPi(bcx, lldesc, ~[0, abi::tydesc_field_size]));

        // Get the bindings pointer and add the size to it
        let llbinds = GEPi(bcx, llenv, ~[0, abi::closure_elt_bindings]);
        bump_ptr(bcx, descsty, llbinds, llsz)
    }
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
fn load_environment(enclosing_cx: &@block_ctxt, fcx: &@fn_ctxt,
                    envty: &ty::t, upvars: &@ast::node_id[],
                    copying: bool) {
    let bcx = new_raw_block_ctxt(fcx, fcx.llcopyargs);

    let ty = ty::mk_imm_box(bcx_tcx(bcx), envty);
    let llty = type_of(bcx_ccx(bcx), bcx.sp, ty);
    let llclosure = bcx.build.PointerCast(fcx.llenv, llty);

    // Populate the type parameters from the environment. We need to
    // do this first because the tydescs are needed to index into
    // the bindings if they are dynamically sized.
    let tydesc_count = std::ivec::len(enclosing_cx.fcx.lltydescs);
    let lltydescs = find_environment_tydescs(bcx, envty, llclosure);
    let i = 0u;
    while i < tydesc_count {
        let lltydescptr = GEPi(bcx, lltydescs, ~[0, i as int]);
        fcx.lltydescs += ~[bcx.build.Load(lltydescptr)];
        i += 1u;
    }

    // Populate the upvars from the environment.
    let path = ~[0, abi::box_rc_field_body, abi::closure_elt_bindings];
    i = 0u;
    // If this is an aliasing closure/for-each body, we need to load
    // the iterbody.
    if !copying && !option::is_none(enclosing_cx.fcx.lliterbody) {
        let iterbodyptr = GEP_tup_like(bcx, ty, llclosure, path + ~[0]);
        fcx.lliterbody = some(bcx.build.Load(iterbodyptr.val));
        bcx = iterbodyptr.bcx;
        i += 1u;
    }
    // Load the acutal upvars.
    for upvar_id: ast::node_id  in *upvars {
        let upvarptr =
            GEP_tup_like(bcx, ty, llclosure, path + ~[i as int]);
        bcx = upvarptr.bcx;
        let llupvarptr = upvarptr.val;
        if !copying { llupvarptr = bcx.build.Load(llupvarptr); }
        let def_id = ast::def_id_of_def(bcx_tcx(bcx).def_map.get(upvar_id));
        fcx.llupvars.insert(def_id.node, llupvarptr);
        i += 1u;
    }
}

fn trans_for_each(cx: &@block_ctxt, local: &@ast::local, seq: &@ast::expr,
                  body: &ast::blk) -> result {
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

    // FIXME: possibly support alias-mode here?
    let decl_ty = node_id_type(lcx.ccx, local.node.id);
    let upvars = get_freevars(lcx.ccx.tcx, body.node.id);

    let llenv = build_closure(cx, upvars, false);

    // Step 2: Declare foreach body function.
    let s: str =
        mangle_internal_name_by_path_and_seq(lcx.ccx, lcx.path, "foreach");

    // The 'env' arg entering the body function is a fake env member (as in
    // the env-part of the normal rust calling convention) that actually
    // points to a stack allocated env in this frame. We bundle that env
    // pointer along with the foreach-body-fn pointer into a 'normal' fn pair
    // and pass it in as a first class fn-arg to the iterator.
    let iter_body_llty =
        type_of_fn_from_ty(lcx.ccx, cx.sp,
                           ty::mk_iter_body_fn(lcx.ccx.tcx, decl_ty), 0u);
    let lliterbody: ValueRef =
        decl_internal_fastcall_fn(lcx.ccx.llmod, s, iter_body_llty);
    let fcx = new_fn_ctxt_w_id(lcx, cx.sp, lliterbody, body.node.id);
    fcx.iterbodyty = cx.fcx.iterbodyty;

    // Generate code to load the environment out of the
    // environment pointer.
    load_environment(cx, fcx, llenv.ptrty, upvars, false);

    let bcx = new_top_block_ctxt(fcx);
    // Add bindings for the loop variable alias.
    bcx = trans_alt::bind_irrefutable_pat
        (bcx, local.node.pat, llvm::LLVMGetParam(fcx.llfn, 3u),
         bcx.fcx.lllocals, false);
    let lltop = bcx.llbb;
    let r = trans_block(bcx, body, return);
    finish_fn(fcx, lltop);

    if !r.bcx.build.is_terminated() {
        // if terminated is true, no need for the ret-fail
        r.bcx.build.RetVoid();
    }

    // Step 3: Call iter passing [lliterbody, llenv], plus other args.
    alt seq.node {
      ast::expr_call(f, args) {
        let pair =
            create_real_fn_pair(cx, iter_body_llty, lliterbody, llenv.ptr);
        r = trans_call(cx, f, some[ValueRef](cx.build.Load(pair)), args,
                       seq.id);
        ret rslt(r.bcx, C_nil());
      }
    }
}

fn trans_while(cx: &@block_ctxt, cond: &@ast::expr, body: &ast::blk) ->
   result {
    let cond_cx = new_scope_block_ctxt(cx, "while cond");
    let next_cx = new_sub_block_ctxt(cx, "next");
    let body_cx =
        new_loop_scope_block_ctxt(cx, option::none[@block_ctxt], next_cx,
                                  "while loop body");
    let body_res = trans_block(body_cx, body, return);
    let cond_res = trans_expr(cond_cx, cond);
    body_res.bcx.build.Br(cond_cx.llbb);
    let cond_bcx = trans_block_cleanups(cond_res.bcx, cond_cx);
    cond_bcx.build.CondBr(cond_res.val, body_cx.llbb, next_cx.llbb);
    cx.build.Br(cond_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn trans_do_while(cx: &@block_ctxt, body: &ast::blk, cond: &@ast::expr) ->
   result {
    let next_cx = new_sub_block_ctxt(cx, "next");
    let body_cx =
        new_loop_scope_block_ctxt(cx, option::none[@block_ctxt], next_cx,
                                  "do-while loop body");
    let body_res = trans_block(body_cx, body, return);
    let cond_res = trans_expr(body_res.bcx, cond);
    cond_res.bcx.build.CondBr(cond_res.val, body_cx.llbb, next_cx.llbb);
    cx.build.Br(body_cx.llbb);
    ret rslt(next_cx, body_res.val);
}

type generic_info =
    {item_type: ty::t,
     static_tis: (option::t[@tydesc_info])[],
     tydescs: ValueRef[]};

type lval_result =
    {res: result,
     is_mem: bool,
     generic: option::t[generic_info],
     llobj: option::t[ValueRef],
     method_ty: option::t[ty::t]};

fn lval_mem(cx: &@block_ctxt, val: ValueRef) -> lval_result {
    ret {res: rslt(cx, val),
         is_mem: true,
         generic: none[generic_info],
         llobj: none[ValueRef],
         method_ty: none[ty::t]};
}

fn lval_val(cx: &@block_ctxt, val: ValueRef) -> lval_result {
    ret {res: rslt(cx, val),
         is_mem: false,
         generic: none[generic_info],
         llobj: none[ValueRef],
         method_ty: none[ty::t]};
}

fn trans_external_path(cx: &@block_ctxt, did: &ast::def_id,
                       tpt: &ty::ty_param_kinds_and_ty) -> ValueRef {
    let lcx = cx.fcx.lcx;
    let name = csearch::get_symbol(lcx.ccx.sess.get_cstore(), did);
    ret get_extern_const(lcx.ccx.externs, lcx.ccx.llmod, name,
                         type_of_ty_param_kinds_and_ty(lcx, cx.sp, tpt));
}

fn lval_generic_fn(cx: &@block_ctxt, tpt: &ty::ty_param_kinds_and_ty,
                   fn_id: &ast::def_id, id: ast::node_id) -> lval_result {
    let lv;
    if fn_id.crate == ast::local_crate {
        // Internal reference.
        assert (bcx_ccx(cx).fn_pairs.contains_key(fn_id.node));
        lv = lval_val(cx, bcx_ccx(cx).fn_pairs.get(fn_id.node));
    } else {
        // External reference.
        lv = lval_val(cx, trans_external_path(cx, fn_id, tpt));
    }
    let tys = ty::node_id_to_type_params(bcx_tcx(cx), id);
    if std::ivec::len[ty::t](tys) != 0u {
        let bcx = lv.res.bcx;
        let tydescs: ValueRef[] = ~[];
        let tis: (option::t[@tydesc_info])[] = ~[];
        for t: ty::t  in tys {
            // TODO: Doesn't always escape.

            let ti = none[@tydesc_info];
            let td = get_tydesc(bcx, t, true, ti);
            tis += ~[ti];
            bcx = td.bcx;
            tydescs += ~[td.val];
        }
        let gen = {item_type: tpt.ty, static_tis: tis, tydescs: tydescs};
        lv = {res: rslt(bcx, lv.res.val), generic: some(gen) with lv};
    }
    ret lv;
}

fn lookup_discriminant(lcx: &@local_ctxt, tid: &ast::def_id,
                       vid: &ast::def_id) -> ValueRef {
    alt lcx.ccx.discrims.find(vid.node) {
      none. {
        // It's an external discriminant that we haven't seen yet.
        assert (vid.crate != ast::local_crate);
        let sym = csearch::get_symbol(lcx.ccx.sess.get_cstore(), vid);
        let gvar = llvm::LLVMAddGlobal(lcx.ccx.llmod, T_int(), str::buf(sym));
        llvm::LLVMSetLinkage(gvar,
                             lib::llvm::LLVMExternalLinkage as llvm::Linkage);
        llvm::LLVMSetGlobalConstant(gvar, True);
        lcx.ccx.discrims.insert(vid.node, gvar);
        ret gvar;
      }
      some(llval) { ret llval; }
    }
}

fn trans_var(cx: &@block_ctxt, sp: &span, id: ast::node_id) ->
   lval_result {
    let ccx = bcx_ccx(cx);
    alt freevars::def_lookup(bcx_tcx(cx), cx.fcx.id, id) {
      some(ast::def_upvar(did, _)) {
        assert (cx.fcx.llupvars.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.llupvars.get(did.node));
      }
      some(ast::def_arg(did)) {
        assert (cx.fcx.llargs.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.llargs.get(did.node));
      }
      some(ast::def_local(did)) {
        assert (cx.fcx.lllocals.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.lllocals.get(did.node));
      }
      some(ast::def_binding(did)) {
        assert (cx.fcx.lllocals.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.lllocals.get(did.node));
      }
      some(ast::def_obj_field(did)) {
        assert (cx.fcx.llobjfields.contains_key(did.node));
        ret lval_mem(cx, cx.fcx.llobjfields.get(did.node));
      }
      some(ast::def_fn(did, _)) {
        let tyt = ty::lookup_item_type(ccx.tcx, did);
        ret lval_generic_fn(cx, tyt, did, id);
      }
      some(ast::def_variant(tid, vid)) {
        let v_tyt = ty::lookup_item_type(ccx.tcx, vid);
        alt ty::struct(ccx.tcx, v_tyt.ty) {
          ty::ty_fn(_, _, _, _, _) {
            // N-ary variant.

            ret lval_generic_fn(cx, v_tyt, vid, id);
          }
          _ {
            // Nullary variant.
            let tag_ty = node_id_type(ccx, id);
            let alloc_result = alloc_ty(cx, tag_ty);
            let lltagblob = alloc_result.val;
            let lltagty = type_of_tag(ccx, sp, tid, tag_ty);
            let bcx = alloc_result.bcx;
            let lltagptr = bcx.build.PointerCast(lltagblob, T_ptr(lltagty));
            if std::ivec::len(ty::tag_variants(ccx.tcx, tid)) != 1u {
                let lldiscrim_gv = lookup_discriminant(bcx.fcx.lcx, tid, vid);
                let lldiscrim = bcx.build.Load(lldiscrim_gv);
                let lldiscrimptr =
                    bcx.build.GEP(lltagptr, ~[C_int(0), C_int(0)]);
                bcx.build.Store(lldiscrim, lldiscrimptr);
            }
            ret lval_val(bcx, lltagptr);
          }
        }
      }
      some(ast::def_const(did)) {
        if did.crate == ast::local_crate {
            assert (ccx.consts.contains_key(did.node));
            ret lval_mem(cx, ccx.consts.get(did.node));
        } else {
            let tp = ty::node_id_to_monotype(ccx.tcx, id);
            let k: ast::kind[] = ~[];
            ret lval_val(cx,
                         load_if_immediate(cx,
                                           trans_external_path(cx, did,
                                                               {kinds: k,
                                                                ty: tp}),
                                           tp));
        }
      }
      some(ast::def_native_fn(did)) {
        let tyt = ty::lookup_item_type(ccx.tcx, did);
        ret lval_generic_fn(cx, tyt, did, id);
      }
      _ { ccx.sess.span_unimpl(cx.sp, "def variant in trans"); }
    }
}

fn trans_path(cx: &@block_ctxt, p: &ast::path, id: ast::node_id) ->
   lval_result {
    ret trans_var(cx, p.span, id);
}

fn trans_field(cx: &@block_ctxt, sp: &span, v: ValueRef, t0: &ty::t,
               field: &ast::ident, id: ast::node_id) -> lval_result {
    let r = autoderef(cx, v, t0);
    let t = r.ty;
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_rec(fields) {
        let ix: uint = ty::field_idx(bcx_ccx(cx).sess, sp, field, fields);
        let v = GEP_tup_like(r.bcx, t, r.val, ~[0, ix as int]);
        ret lval_mem(v.bcx, v.val);
      }
      ty::ty_obj(methods) {
        let ix: uint = ty::method_idx(bcx_ccx(cx).sess, sp, field, methods);
        let vtbl =
            r.bcx.build.GEP(r.val, ~[C_int(0), C_int(abi::obj_field_vtbl)]);
        vtbl = r.bcx.build.Load(vtbl);

        let vtbl_type = T_ptr(T_array(T_ptr(T_nil()), ix + 1u));
        vtbl = cx.build.PointerCast(vtbl, vtbl_type);

        let v = r.bcx.build.GEP(vtbl, ~[C_int(0), C_int(ix as int)]);
        let fn_ty: ty::t = ty::method_ty_to_fn_ty(bcx_tcx(cx), methods.(ix));
        let tcx = bcx_tcx(cx);
        let ll_fn_ty =
            type_of_fn_full(bcx_ccx(cx), sp, ty::ty_fn_proto(tcx, fn_ty),
                            true, ty::ty_fn_args(tcx, fn_ty),
                            ty::ty_fn_ret(tcx, fn_ty), 0u);
        v = r.bcx.build.PointerCast(v, T_ptr(T_ptr(ll_fn_ty)));
        let lvo = lval_mem(r.bcx, v);
        ret {llobj: some[ValueRef](r.val), method_ty: some[ty::t](fn_ty)
                with lvo};
      }
      _ { bcx_ccx(cx).sess.unimpl("field variant in trans_field"); }
    }
}

fn trans_index(cx: &@block_ctxt, sp: &span, base: &@ast::expr,
               idx: &@ast::expr, id: ast::node_id) -> lval_result {
    // Is this an interior vector?

    let base_ty = ty::expr_ty(bcx_tcx(cx), base);
    let exp = trans_expr(cx, base);
    let lv = autoderef(exp.bcx, exp.val, base_ty);
    let base_ty_no_boxes = lv.ty;
    let is_interior = ty::sequence_is_interior(bcx_tcx(cx), base_ty_no_boxes);
    let ix = trans_expr(lv.bcx, idx);
    let v = lv.val;
    let bcx = ix.bcx;
    // Cast to an LLVM integer. Rust is less strict than LLVM in this regard.

    let ix_val;
    let ix_size = llsize_of_real(bcx_ccx(cx), val_ty(ix.val));
    let int_size = llsize_of_real(bcx_ccx(cx), T_int());
    if ix_size < int_size {
        ix_val = bcx.build.ZExt(ix.val, T_int());
    } else if (ix_size > int_size) {
        ix_val = bcx.build.Trunc(ix.val, T_int());
    } else { ix_val = ix.val; }
    let unit_ty = node_id_type(bcx_ccx(cx), id);
    let unit_sz = size_of(bcx, unit_ty);
    bcx = unit_sz.bcx;
    maybe_name_value(bcx_ccx(cx), unit_sz.val, "unit_sz");
    let scaled_ix = bcx.build.Mul(ix_val, unit_sz.val);
    maybe_name_value(bcx_ccx(cx), scaled_ix, "scaled_ix");
    let interior_len_and_data;
    if is_interior {
        let rslt = ivec::get_len_and_data(bcx, v, unit_ty);
        interior_len_and_data = some({len: rslt.len, data: rslt.data});
        bcx = rslt.bcx;
    } else { interior_len_and_data = none; }
    let lim;
    alt interior_len_and_data {
      some(lad) { lim = lad.len; }
      none. {
        lim = bcx.build.GEP(v, ~[C_int(0), C_int(abi::vec_elt_fill)]);
        lim = bcx.build.Load(lim);
      }
    }
    let bounds_check = bcx.build.ICmp(lib::llvm::LLVMIntULT, scaled_ix, lim);
    let fail_cx = new_sub_block_ctxt(bcx, "fail");
    let next_cx = new_sub_block_ctxt(bcx, "next");
    bcx.build.CondBr(bounds_check, next_cx.llbb, fail_cx.llbb);
    // fail: bad bounds check.

    trans_fail(fail_cx, some[span](sp), "bounds check");
    let body;
    alt interior_len_and_data {
      some(lad) { body = lad.data; }
      none. {
        body =
            next_cx.build.GEP(v,
                              ~[C_int(0), C_int(abi::vec_elt_data),
                                C_int(0)]);
      }
    }
    let elt;
    if ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
        body = next_cx.build.PointerCast(body, T_ptr(T_i8()));
        elt = next_cx.build.GEP(body, ~[scaled_ix]);
    } else {
        elt = next_cx.build.GEP(body, ~[ix_val]);
        // We're crossing a box boundary here, so we may need to pointer cast.

        let llunitty = type_of(bcx_ccx(next_cx), sp, unit_ty);
        elt = next_cx.build.PointerCast(elt, T_ptr(llunitty));
    }
    ret lval_mem(next_cx, elt);
}


// The additional bool returned indicates whether it's mem (that is
// represented as an alloca or heap, hence needs a 'load' to be used as an
// immediate).
fn trans_lval_gen(cx: &@block_ctxt, e: &@ast::expr) -> lval_result {
    alt e.node {
      ast::expr_path(p) { ret trans_path(cx, p, e.id); }
      ast::expr_field(base, ident) {
        let r = trans_expr(cx, base);
        let t = ty::expr_ty(bcx_tcx(cx), base);
        ret trans_field(r.bcx, e.span, r.val, t, ident, e.id);
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
                sub.bcx.build.InBoundsGEP(sub.val,
                                          ~[C_int(0),
                                            C_int(abi::box_rc_field_body)])
              }
              ty::ty_res(_, _, _) {
                sub.bcx.build.InBoundsGEP(sub.val, ~[C_int(0), C_int(1)])
              }
              ty::ty_tag(_, _) {
                let ety = ty::expr_ty(ccx.tcx, e);
                let ellty;
                if ty::type_has_dynamic_size(ccx.tcx, ety) {
                    ellty = T_typaram_ptr(ccx.tn);
                } else { ellty = T_ptr(type_of(ccx, e.span, ety)); }
                sub.bcx.build.PointerCast(sub.val, ellty)
              }
              ty::ty_ptr(_) { sub.val }
            };
        ret lval_mem(sub.bcx, val);
      }
      ast::expr_self_method(ident) {
        alt { cx.fcx.llself } {
          some(pair) {
            let r = pair.v;
            let t = pair.t;
            ret trans_field(cx, e.span, r, t, ident, e.id);
          }
          _ {
            // Shouldn't happen.
            bcx_ccx(cx).sess.bug("trans_lval called on \
                                         expr_self_method in \
                                         a context without llself");
          }
        }
      }
      _ {
        ret {res: trans_expr(cx, e),
             is_mem: false,
             generic: none,
             llobj: none,
             method_ty: none};
      }
    }
}

fn trans_lval(cx: &@block_ctxt, e: &@ast::expr) -> lval_result {
    let lv = trans_lval_gen(cx, e);
    alt lv.generic {
      some(gi) {
        let t = ty::expr_ty(bcx_tcx(cx), e);
        let n_args = std::ivec::len(ty::ty_fn_args(bcx_tcx(cx), t));
        let args = std::ivec::init_elt(none[@ast::expr], n_args);
        let bound = trans_bind_1(lv.res.bcx, e, lv, args, e.id);
        ret lval_val(bound.bcx, bound.val);
      }
      none. { ret lv; }
    }
}

fn int_cast(bcx: &@block_ctxt, lldsttype: TypeRef, llsrctype: TypeRef,
            llsrc: ValueRef, signed: bool) -> ValueRef {
    let srcsz = llvm::LLVMGetIntTypeWidth(llsrctype);
    let dstsz = llvm::LLVMGetIntTypeWidth(lldsttype);
    ret if dstsz == srcsz {
            bcx.build.BitCast(llsrc, lldsttype)
        } else if (srcsz > dstsz) {
            bcx.build.TruncOrBitCast(llsrc, lldsttype)
        } else if (signed) {
            bcx.build.SExtOrBitCast(llsrc, lldsttype)
        } else { bcx.build.ZExtOrBitCast(llsrc, lldsttype) };
}

fn float_cast(bcx: &@block_ctxt, lldsttype: TypeRef, llsrctype: TypeRef,
              llsrc: ValueRef) -> ValueRef {
    let srcsz = lib::llvm::float_width(llsrctype);
    let dstsz = lib::llvm::float_width(lldsttype);
    ret if dstsz > srcsz {
            bcx.build.FPExt(llsrc, lldsttype)
        } else if (srcsz > dstsz) {
            bcx.build.FPTrunc(llsrc, lldsttype)
        } else { llsrc };
}

fn trans_cast(cx: &@block_ctxt, e: &@ast::expr, id: ast::node_id) -> result {
    let ccx = bcx_ccx(cx);
    let e_res = trans_expr(cx, e);
    let ll_t_in = val_ty(e_res.val);
    let t_in = ty::expr_ty(ccx.tcx, e);
    let t_out = node_id_type(ccx, id);
    let ll_t_out = type_of(ccx, e.span, t_out);

    tag kind { native_; integral; float; other; }
    fn t_kind(tcx: &ty::ctxt, t: ty::t) -> kind {
        ret if ty::type_is_fp(tcx, t) {
                float
            } else if (ty::type_is_native(tcx, t)) {
                native_
            } else if (ty::type_is_integral(tcx, t)) {
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
                e_res.bcx.build.SIToFP(e_res.val, ll_t_out)
            } else { e_res.bcx.build.UIToFP(e_res.val, ll_t_out) }
          }
          {in: float., out: integral.} {
            if ty::type_is_signed(ccx.tcx, t_out) {
                e_res.bcx.build.FPToSI(e_res.val, ll_t_out)
            } else { e_res.bcx.build.FPToUI(e_res.val, ll_t_out) }
          }
          {in: integral., out: native_.} {
            e_res.bcx.build.IntToPtr(e_res.val, ll_t_out)
          }
          {in: native_., out: integral.} {
            e_res.bcx.build.PtrToInt(e_res.val, ll_t_out)
          }
          {in: native_., out: native_.} {
            e_res.bcx.build.PointerCast(e_res.val, ll_t_out)
          }
          _ { ccx.sess.bug("Translating unsupported cast.") }
        };
    ret rslt(e_res.bcx, newval);
}

fn trans_bind_thunk(cx: &@local_ctxt, sp: &span, incoming_fty: &ty::t,
                    outgoing_fty: &ty::t, args: &(option::t[@ast::expr])[],
                    env_ty: &ty::t, bound_tys: &ty::t[],
                    ty_param_count: uint) -> {val: ValueRef, ty: TypeRef} {

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
    let s: str =
        mangle_internal_name_by_path_and_seq(cx.ccx, cx.path, "thunk");
    let llthunk_ty: TypeRef =
        get_pair_fn_ty(type_of(cx.ccx, sp, incoming_fty));
    let llthunk: ValueRef =
        decl_internal_fastcall_fn(cx.ccx.llmod, s, llthunk_ty);

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
    let closure_ty = ty::mk_imm_box(cx.ccx.tcx, env_ty);
    let llclosure_ptr_ty = type_of(cx.ccx, sp, closure_ty);
    let llclosure =
        copy_args_bcx.build.PointerCast(fcx.llenv, llclosure_ptr_ty);

    // "target", in this context, means the function that's having some of its
    // arguments bound and that will be called inside the thunk we're
    // creating.  (In our running example, target is the function f.)  Pick
    // out the pointer to the target function from the environment. The
    // target function lives in the first binding spot.
    let lltarget =
        GEP_tup_like(bcx, closure_ty, llclosure,
                     ~[0, abi::box_rc_field_body,
                       abi::closure_elt_bindings, 0]);
    bcx = lltarget.bcx;

    // And then, pick out the target function's own environment.  That's what
    // we'll use as the environment the thunk gets.
    let lltargetclosure =
        bcx.build.GEP(lltarget.val, ~[C_int(0), C_int(abi::fn_field_box)]);
    lltargetclosure = bcx.build.Load(lltargetclosure);

    // Get f's return type, which will also be the return type of the entire
    // bind expression.
    let outgoing_ret_ty = ty::ty_fn_ret(cx.ccx.tcx, outgoing_fty);

    // Get the types of the arguments to f.
    let outgoing_args = ty::ty_fn_args(cx.ccx.tcx, outgoing_fty);

    // The 'llretptr' that will arrive in the thunk we're creating also needs
    // to be the correct type.  Cast it to f's return type, if necessary.
    let llretptr = fcx.llretptr;
    if ty::type_contains_params(cx.ccx.tcx, outgoing_ret_ty) {
        let llretty = type_of_inner(cx.ccx, sp, outgoing_ret_ty);
        llretptr = bcx.build.PointerCast(llretptr, T_ptr(llretty));
    }

    // Set up the three implicit arguments to the thunk.
    let llargs: ValueRef[] = ~[llretptr, fcx.lltaskptr, lltargetclosure];

    // Copy in the type parameters.
    let i: uint = 0u;
    while i < ty_param_count {
        let lltyparam_ptr =
            GEP_tup_like(copy_args_bcx, closure_ty, llclosure,
                         ~[0, abi::box_rc_field_body,
                           abi::closure_elt_ty_params, i as int]);
        copy_args_bcx = lltyparam_ptr.bcx;
        let td = copy_args_bcx.build.Load(lltyparam_ptr.val);
        llargs += ~[td];
        fcx.lltydescs += ~[td];
        i += 1u;
    }

    let a: uint = 3u; // retptr, task ptr, env come first
    let b: int = 1;
    let outgoing_arg_index: uint = 0u;
    let llout_arg_tys: TypeRef[] =
        type_of_explicit_args(cx.ccx, sp, outgoing_args);
    for arg: option::t[@ast::expr]  in args {
        let out_arg = outgoing_args.(outgoing_arg_index);
        let llout_arg_ty = llout_arg_tys.(outgoing_arg_index);
        let is_val = out_arg.mode == ty::mo_val;
        alt arg {
          // Arg provided at binding time; thunk copies it from
          // closure.
          some(e) {
            let e_ty = ty::expr_ty(cx.ccx.tcx, e);
            let bound_arg =
                GEP_tup_like(bcx, closure_ty, llclosure,
                             ~[0, abi::box_rc_field_body,
                               abi::closure_elt_bindings, b]);
            bcx = bound_arg.bcx;
            let val = bound_arg.val;
            // If the type is parameterized, then we need to cast the
            // type we actually have to the parameterized out type.
            if ty::type_contains_params(cx.ccx.tcx, out_arg.ty) {
                let ty = if is_val
                         { T_ptr(llout_arg_ty) } else { llout_arg_ty };
                val = bcx.build.PointerCast(val, ty);
            }
            if is_val {
                if type_is_immediate(cx.ccx, e_ty) {
                    val = bcx.build.Load(val);
                    bcx = copy_ty(bcx, val, e_ty).bcx;
                } else {
                    bcx = copy_ty(bcx, val, e_ty).bcx;
                    val = bcx.build.Load(val);
                }
            }
            llargs += ~[val];
            b += 1;
          }

          // Arg will be provided when the thunk is invoked.
          none. {
            let arg: ValueRef = llvm::LLVMGetParam(llthunk, a);
            if ty::type_contains_params(cx.ccx.tcx, out_arg.ty) {
                // If the argument was passed by value and isn't a
                // pointer type, we need to spill it to an alloca in
                // order to do a pointer cast. Argh.
                if is_val && !ty::type_is_boxed(cx.ccx.tcx, out_arg.ty) {
                    let argp = do_spill(bcx, arg);
                    argp = bcx.build.PointerCast(argp, T_ptr(llout_arg_ty));
                    arg = bcx.build.Load(argp);
                } else {
                    arg = bcx.build.PointerCast(arg, llout_arg_ty);
                }
            }
            llargs += ~[arg];
            a += 1u;
          }
        }
        outgoing_arg_index += 1u;
    }
    // FIXME: turn this call + ret into a tail call.
    let lltargetfn =
        bcx.build.GEP(lltarget.val, ~[C_int(0), C_int(abi::fn_field_code)]);

    // Cast the outgoing function to the appropriate type.
    // This is necessary because the type of the function that we have
    // in the closure does not know how many type descriptors the function
    // needs to take.
    let lltargetty =
        type_of_fn_from_ty(bcx_ccx(bcx), sp, outgoing_fty, ty_param_count);
    lltargetfn = bcx.build.PointerCast(lltargetfn, T_ptr(T_ptr(lltargetty)));
    lltargetfn = bcx.build.Load(lltargetfn);
    bcx.build.FastCall(lltargetfn, llargs);
    bcx.build.RetVoid();
    finish_fn(fcx, lltop);
    ret {val: llthunk, ty: llthunk_ty};
}

fn trans_bind(cx: &@block_ctxt, f: &@ast::expr,
              args: &(option::t[@ast::expr])[], id: ast::node_id) -> result {
    let f_res = trans_lval_gen(cx, f);
    ret trans_bind_1(cx, f, f_res, args, id);
}

fn trans_bind_1(cx: &@block_ctxt, f: &@ast::expr, f_res: &lval_result,
                args: &(option::t[@ast::expr])[], id: ast::node_id) ->
   result {
    let bound: (@ast::expr)[] = ~[];
    for argopt: option::t[@ast::expr]  in args {
        alt argopt { none. { } some(e) { bound += ~[e]; } }
    }

    // Figure out which tydescs we need to pass, if any.
    let outgoing_fty: ty::t = ty::expr_ty(bcx_tcx(cx), f);
    let outgoing_fty_real; // the type with typarams still in it
    let lltydescs: ValueRef[];
    alt f_res.generic {
      none. { outgoing_fty_real = outgoing_fty; lltydescs = ~[]; }
      some(ginfo) {
        lazily_emit_all_generic_info_tydesc_glues(cx, ginfo);
        outgoing_fty_real = ginfo.item_type;
        lltydescs = ginfo.tydescs;
      }
    }

    let ty_param_count = std::ivec::len(lltydescs);
    if std::ivec::len(bound) == 0u && ty_param_count == 0u {
        // Trivial 'binding': just return the static pair-ptr.
        ret f_res.res;
    }
    let bcx = f_res.res.bcx;

    // Cast the function we are binding to be the type that the closure
    // will expect it to have. The type the closure knows about has the
    // type parameters substituted with the real types.
    let llclosurety = T_ptr(type_of(bcx_ccx(cx), cx.sp, outgoing_fty));
    let src_loc = bcx.build.PointerCast(f_res.res.val, llclosurety);
    let bound_f = {res: {bcx: bcx, val: src_loc} with f_res};

    // Arrange for the bound function to live in the first binding spot.
    let bound_tys: ty::t[] = ~[outgoing_fty];
    let bound_vals: lval_result[] = ~[bound_f];
    // Translate the bound expressions.
    for e: @ast::expr  in bound {
        let lv = trans_lval(bcx, e);
        bcx = lv.res.bcx;
        bound_vals += ~[lv];
        bound_tys += ~[ty::expr_ty(bcx_tcx(cx), e)];
    }

    // Actually construct the closure
    let closure = build_environment(bcx, lltydescs,
                                    bound_tys, bound_vals, true);
    bcx = closure.bcx;

    // Make thunk
    // The type of the entire bind expression.
    let pair_ty = node_id_type(bcx_ccx(cx), id);
    let llthunk =
        trans_bind_thunk(cx.fcx.lcx, cx.sp, pair_ty, outgoing_fty_real,
                         args, closure.ptrty, bound_tys, ty_param_count);

    // Construct the function pair
    let pair_v = create_real_fn_pair(bcx, llthunk.ty, llthunk.val,
                                     closure.ptr);
    add_clean_temp(cx, pair_v, pair_ty);
    ret rslt(bcx, pair_v);
}

fn trans_arg_expr(cx: &@block_ctxt, arg: &ty::arg, lldestty0: TypeRef,
                  e: &@ast::expr) -> result {
    let ccx = bcx_ccx(cx);
    let e_ty = ty::expr_ty(ccx.tcx, e);
    let is_bot = ty::type_is_bot(ccx.tcx, e_ty);
    let lv = trans_lval(cx, e);
    let bcx = lv.res.bcx;
    let val = lv.res.val;
    if is_bot {
        // For values of type _|_, we generate an
        // "undef" value, as such a value should never
        // be inspected. It's important for the value
        // to have type lldestty0 (the callee's expected type).
        val = llvm::LLVMGetUndef(lldestty0);
    } else if (arg.mode == ty::mo_val) {
        if ty::type_owns_heap_mem(ccx.tcx, e_ty) {
            let dst = alloc_ty(bcx, e_ty);
            val = dst.val;
            bcx = move_val_if_temp(dst.bcx, INIT, val, lv, e_ty).bcx;
        } else if (lv.is_mem) {
            val = load_if_immediate(bcx, val, e_ty);
            bcx = copy_ty(bcx, val, e_ty).bcx;
        } else {
            // Eliding take/drop for appending of external vectors currently
            // corrupts memory. I can't figure out why, and external vectors
            // are on the way out anyway, so this simply turns off the
            // optimization for that case.
            let is_ext_vec_plus =
                alt e.node {
                  ast::expr_binary(_, _, _) {
                    ty::type_is_sequence(ccx.tcx, e_ty) &&
                        !ty::sequence_is_interior(ccx.tcx, e_ty)
                  }
                  _ { false }
                };
            if is_ext_vec_plus {
                bcx = copy_ty(bcx, val, e_ty).bcx;
            } else { revoke_clean(bcx, val); }
        }
    } else if (type_is_immediate(ccx, e_ty) && !lv.is_mem) {
        val = do_spill(bcx, val);
    }

    if !is_bot && ty::type_contains_params(ccx.tcx, arg.ty) {
        let lldestty = lldestty0;
        if arg.mode == ty::mo_val && ty::type_is_structural(ccx.tcx, e_ty) {
            lldestty = T_ptr(lldestty);
        }
        val = bcx.build.PointerCast(val, lldestty);
    }
    if arg.mode == ty::mo_val && ty::type_is_structural(ccx.tcx, e_ty) {
        // Until here we've been treating structures by pointer;
        // we are now passing it as an arg, so need to load it.
        val = bcx.build.Load(val);
    }
    ret rslt(bcx, val);
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn_full
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn trans_args(cx: &@block_ctxt, llenv: ValueRef, llobj: &option::t[ValueRef],
              gen: &option::t[generic_info], lliterbody: &option::t[ValueRef],
              es: &(@ast::expr)[], fn_ty: &ty::t) ->
   {bcx: @block_ctxt, args: ValueRef[], retslot: ValueRef} {
    let args: ty::arg[] = ty::ty_fn_args(bcx_tcx(cx), fn_ty);
    let llargs: ValueRef[] = ~[];
    let lltydescs: ValueRef[] = ~[];
    let bcx: @block_ctxt = cx;
    // Arg 0: Output pointer.

    // FIXME: test case looks like
    // f(1, fail, @42);
    if bcx.build.is_terminated() {
        // This means an earlier arg was divergent.
        // So this arg can't be evaluated.
        ret {bcx: bcx, args: ~[], retslot: C_nil()};
    }

    let retty = ty::ty_fn_ret(bcx_tcx(cx), fn_ty);
    let llretslot_res = alloc_ty(bcx, retty);
    bcx = llretslot_res.bcx;
    let llretslot = llretslot_res.val;
    alt gen {
      some(g) {
        lazily_emit_all_generic_info_tydesc_glues(cx, g);
        lltydescs = g.tydescs;
        args = ty::ty_fn_args(bcx_tcx(cx), g.item_type);
        retty = ty::ty_fn_ret(bcx_tcx(cx), g.item_type);
      }
      _ { }
    }
    if (ty::type_contains_params(bcx_tcx(cx), retty)) {
        // It's possible that the callee has some generic-ness somewhere in
        // its return value -- say a method signature within an obj or a fn
        // type deep in a structure -- which the caller has a concrete view
        // of. If so, cast the caller's view of the restlot to the callee's
        // view, for the sake of making a type-compatible call.
        let llretty = T_ptr(type_of_inner(bcx_ccx(bcx), bcx.sp, retty));
        llargs += ~[cx.build.PointerCast(llretslot, llretty)];
    } else { llargs += ~[llretslot]; }

    // Arg 1: task pointer.
    llargs += ~[bcx.fcx.lltaskptr];

    // Arg 2: Env (closure-bindings / self-obj)
    alt llobj {
      some(ob) {
        // Every object is always found in memory,
        // and not-yet-loaded (as part of an lval x.y
        // doted method-call).

        llargs += ~[bcx.build.Load(ob)];
      }
      _ { llargs += ~[llenv]; }
    }

    // Args >3: ty_params ...
    llargs += lltydescs;

    // ... then possibly an lliterbody argument.
    alt lliterbody { none. { } some(lli) { llargs += ~[lli]; } }

    // ... then explicit args.

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    let arg_tys = type_of_explicit_args(bcx_ccx(cx), cx.sp, args);
    let i = 0u;
    for e: @ast::expr  in es {
        if bcx.build.is_terminated() {
            // This means an earlier arg was divergent.
            // So this arg can't be evaluated.
            break;
        }
        let r = trans_arg_expr(bcx, args.(i), arg_tys.(i), e);
        bcx = r.bcx;
        llargs += ~[r.val];
        i += 1u;
    }
    ret {bcx: bcx, args: llargs, retslot: llretslot};
}

fn trans_call(cx: &@block_ctxt, f: &@ast::expr,
              lliterbody: &option::t[ValueRef], args: &(@ast::expr)[],
              id: ast::node_id) -> result {
    // NB: 'f' isn't necessarily a function; it might be an entire self-call
    // expression because of the hack that allows us to process self-calls
    // with trans_call.

    let f_res = trans_lval_gen(cx, f);
    let fn_ty: ty::t;
    alt f_res.method_ty {
      some(meth) {
        // self-call
        fn_ty = meth;
      }
      _ { fn_ty = ty::expr_ty(bcx_tcx(cx), f); }
    }

    let bcx = f_res.res.bcx;

    let faddr = f_res.res.val;
    let llenv = C_null(T_opaque_closure_ptr(*bcx_ccx(cx)));
    alt f_res.llobj {
      some(_) {
        // It's a vtbl entry.
        faddr = bcx.build.Load(faddr);
      }
      none. {
        // It's a closure. We have to autoderef.
        if f_res.is_mem { faddr = load_if_immediate(bcx, faddr, fn_ty); }
        let res = autoderef(bcx, faddr, fn_ty);
        bcx = res.bcx;
        fn_ty = res.ty;

        let pair = res.val;
        faddr = bcx.build.GEP(pair, ~[C_int(0), C_int(abi::fn_field_code)]);
        faddr = bcx.build.Load(faddr);
        let llclosure =
            bcx.build.GEP(pair, ~[C_int(0), C_int(abi::fn_field_box)]);
        llenv = bcx.build.Load(llclosure);
      }
    }

    let ret_ty = ty::node_id_to_type(bcx_tcx(cx), id);
    let args_res =
        trans_args(bcx, llenv, f_res.llobj, f_res.generic, lliterbody, args,
                   fn_ty);
    bcx = args_res.bcx;
    let llargs = args_res.args;
    let llretslot = args_res.retslot;
    /*
    log "calling: " + val_str(bcx_ccx(cx).tn, faddr);

    for (ValueRef arg in llargs) {
        log "arg: " + val_str(bcx_ccx(cx).tn, arg);
    }
    */

    /* If the block is terminated,
       then one or more of the args has
       type _|_. Since that means it diverges, the code
       for the call itself is unreachable. */
    let retval = C_nil();
    if !bcx.build.is_terminated() {
        bcx.build.FastCall(faddr, llargs);
        alt lliterbody {
          none. {
            if !ty::type_is_nil(bcx_tcx(cx), ret_ty) {
                retval = load_if_immediate(bcx, llretslot, ret_ty);
                // Retval doesn't correspond to anything really tangible
                // in the frame, but it's a ref all the same, so we put a
                // note here to drop it when we're done in this scope.
                add_clean_temp(cx, retval, ret_ty);
            }
          }
          some(_) {
            // If there was an lliterbody, it means we were calling an
            // iter, and we are *not* the party using its 'output' value,
            // we should ignore llretslot.
          }
        }
    }
    ret rslt(bcx, retval);
}

fn trans_vec(cx: &@block_ctxt, args: &(@ast::expr)[], id: ast::node_id) ->
   result {
    let t = node_id_type(bcx_ccx(cx), id);
    let unit_ty = t;
    alt ty::struct(bcx_tcx(cx), t) {
      ty::ty_vec(mt) { unit_ty = mt.ty; }
      _ { bcx_ccx(cx).sess.bug("non-vec type in trans_vec"); }
    }
    let bcx = cx;
    let unit_sz = size_of(bcx, unit_ty);
    bcx = unit_sz.bcx;
    let data_sz =
        bcx.build.Mul(C_uint(std::ivec::len[@ast::expr](args)), unit_sz.val);
    // FIXME: pass tydesc properly.

    let vec_val =
        bcx.build.Call(bcx_ccx(bcx).upcalls.new_vec,
                       ~[bcx.fcx.lltaskptr, data_sz,
                         C_null(T_ptr(bcx_ccx(bcx).tydesc_type))]);
    let llty = type_of(bcx_ccx(bcx), bcx.sp, t);
    vec_val = bcx.build.PointerCast(vec_val, llty);
    add_clean_temp(bcx, vec_val, t);
    let body = bcx.build.GEP(vec_val, ~[C_int(0), C_int(abi::vec_elt_data)]);
    let pseudo_tup_ty =
        ty::mk_imm_tup(bcx_tcx(cx),
                       std::ivec::init_elt[ty::t](unit_ty,
                                                  std::ivec::len(args)));
    let i: int = 0;
    for e: @ast::expr  in args {
        let src = trans_lval(bcx, e);
        bcx = src.res.bcx;
        let dst_res = GEP_tup_like(bcx, pseudo_tup_ty, body, ~[0, i]);
        bcx = dst_res.bcx;
        // Cast the destination type to the source type. This is needed to
        // make tags work, for a subtle combination of reasons:
        //
        // (1) "dst_res" above is derived from "body", which is in turn
        //     derived from "vec_val".
        // (2) "vec_val" has the LLVM type "llty".
        // (3) "llty" is the result of calling type_of() on a vector type.
        // (4) For tags, type_of() returns a different type depending on
        //     on whether the tag is behind a box or not. Vector types are
        //     considered boxes.
        // (5) "src_res" is derived from "unit_ty", which is not behind a box.

        let dst_val;
        if !ty::type_has_dynamic_size(bcx_tcx(cx), unit_ty) {
            let llunit_ty = type_of(bcx_ccx(cx), bcx.sp, unit_ty);
            dst_val = bcx.build.PointerCast(dst_res.val, T_ptr(llunit_ty));
        } else { dst_val = dst_res.val; }
        bcx = move_val_if_temp(bcx, INIT, dst_val, src, unit_ty).bcx;
        i += 1;
    }
    let fill = bcx.build.GEP(vec_val, ~[C_int(0), C_int(abi::vec_elt_fill)]);
    bcx.build.Store(data_sz, fill);
    ret rslt(bcx, vec_val);
}


// TODO: Move me to ivec::
fn trans_ivec(bcx: @block_ctxt, args: &(@ast::expr)[], id: ast::node_id) ->
   result {
    let typ = node_id_type(bcx_ccx(bcx), id);
    let unit_ty;
    alt ty::struct(bcx_tcx(bcx), typ) {
      ty::ty_ivec(mt) { unit_ty = mt.ty; }
      _ { bcx_ccx(bcx).sess.bug("non-ivec type in trans_ivec"); }
    }
    let llunitty = type_of_or_i8(bcx, unit_ty);

    let ares = ivec::alloc(bcx, unit_ty);
    bcx = ares.bcx;
    let llvecptr = ares.llptr;
    let unit_sz = ares.llunitsz;
    let llalen = ares.llalen;

    add_clean_temp(bcx, llvecptr, typ);

    let lllen = bcx.build.Mul(C_uint(std::ivec::len(args)), unit_sz);
    // Allocate the vector pieces and store length and allocated length.

    let llfirsteltptr;
    if std::ivec::len(args) > 0u &&
           std::ivec::len(args) <= abi::ivec_default_length {
        // Interior case.

        bcx.build.Store(lllen,
                        bcx.build.InBoundsGEP(llvecptr,
                                              ~[C_int(0),
                                                C_uint(abi::ivec_elt_len)]));
        bcx.build.Store(llalen,
                        bcx.build.InBoundsGEP(llvecptr,
                                              ~[C_int(0),
                                                C_uint(abi::ivec_elt_alen)]));
        llfirsteltptr =
            bcx.build.InBoundsGEP(llvecptr,
                                  ~[C_int(0), C_uint(abi::ivec_elt_elems),
                                    C_int(0)]);
    } else {
        // Heap case.

        let stub_z = ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_zero)];
        let stub_a = ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_alen)];
        let stub_p = ~[C_int(0), C_uint(abi::ivec_heap_stub_elt_ptr)];
        let llstubty = T_ivec_heap(llunitty);
        let llstubptr = bcx.build.PointerCast(llvecptr, T_ptr(llstubty));
        bcx.build.Store(C_int(0), bcx.build.InBoundsGEP(llstubptr, stub_z));
        let llheapty = T_ivec_heap_part(llunitty);
        if std::ivec::len(args) == 0u {
            // Null heap pointer indicates a zero-length vector.

            bcx.build.Store(llalen, bcx.build.InBoundsGEP(llstubptr, stub_a));
            bcx.build.Store(C_null(T_ptr(llheapty)),
                            bcx.build.InBoundsGEP(llstubptr, stub_p));
            llfirsteltptr = C_null(T_ptr(llunitty));
        } else {
            bcx.build.Store(lllen, bcx.build.InBoundsGEP(llstubptr, stub_a));

            let llheapsz = bcx.build.Add(llsize_of(llheapty), lllen);
            let rslt = trans_shared_malloc(bcx, T_ptr(llheapty), llheapsz);
            bcx = rslt.bcx;
            let llheapptr = rslt.val;
            bcx.build.Store(llheapptr,
                            bcx.build.InBoundsGEP(llstubptr, stub_p));
            let heap_l = ~[C_int(0), C_uint(abi::ivec_heap_elt_len)];
            bcx.build.Store(lllen, bcx.build.InBoundsGEP(llheapptr, heap_l));
            llfirsteltptr =
                bcx.build.InBoundsGEP(llheapptr,
                                      ~[C_int(0),
                                        C_uint(abi::ivec_heap_elt_elems),
                                        C_int(0)]);
        }
    }
    // Store the individual elements.

    let i = 0u;
    for e: @ast::expr  in args {
        let lv = trans_lval(bcx, e);
        bcx = lv.res.bcx;
        let lleltptr;
        if ty::type_has_dynamic_size(bcx_tcx(bcx), unit_ty) {
            lleltptr =
                bcx.build.InBoundsGEP(llfirsteltptr,
                                      ~[bcx.build.Mul(C_uint(i), unit_sz)]);
        } else {
            lleltptr = bcx.build.InBoundsGEP(llfirsteltptr, ~[C_uint(i)]);
        }
        bcx = move_val_if_temp(bcx, INIT, lleltptr, lv, unit_ty).bcx;
        i += 1u;
    }
    ret rslt(bcx, llvecptr);
}

fn trans_rec(cx: &@block_ctxt, fields: &ast::field[],
             base: &option::t[@ast::expr], id: ast::node_id) -> result {
    let bcx = cx;
    let t = node_id_type(bcx_ccx(bcx), id);
    let rec_res = alloc_ty(bcx, t);
    let rec_val = rec_res.val;
    bcx = rec_res.bcx;
    add_clean_temp(cx, rec_val, t);
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
    let ty_fields: ty::field[] = ~[];
    alt ty::struct(bcx_tcx(cx), t) { ty::ty_rec(flds) { ty_fields = flds; } }
    for tf: ty::field  in ty_fields {
        let e_ty = tf.mt.ty;
        let dst_res = GEP_tup_like(bcx, t, rec_val, ~[0, i]);
        bcx = dst_res.bcx;
        let expr_provided = false;
        for f: ast::field  in fields {
            if str::eq(f.node.ident, tf.ident) {
                expr_provided = true;
                let lv = trans_lval(bcx, f.node.expr);
                bcx =
                    move_val_if_temp(lv.res.bcx, INIT, dst_res.val, lv,
                                     e_ty).bcx;
                break;
            }
        }
        if !expr_provided {
            let src_res = GEP_tup_like(bcx, t, base_val, ~[0, i]);
            src_res =
                rslt(src_res.bcx, load_if_immediate(bcx, src_res.val, e_ty));
            bcx =
                copy_val(src_res.bcx, INIT, dst_res.val, src_res.val,
                         e_ty).bcx;
        }
        i += 1;
    }
    ret rslt(bcx, rec_val);
}

fn trans_expr(cx: &@block_ctxt, e: &@ast::expr) -> result {
    ret trans_expr_out(cx, e, return);
}

fn trans_expr_out(cx: &@block_ctxt, e: &@ast::expr, output: out_method) ->
   result {
    // FIXME Fill in cx.sp
    alt e.node {
      ast::expr_lit(lit) { ret trans_lit(cx, *lit); }
      ast::expr_unary(op, x) {
        if op != ast::deref { ret trans_unary(cx, op, x, e.id); }
      }
      ast::expr_binary(op, x, y) { ret trans_binary(cx, op, x, y); }
      ast::expr_if(cond, thn, els) {
        ret with_out_method(bind trans_if(cx, cond, thn, els, e.id, _), cx,
                            e.id, output);
      }
      ast::expr_if_check(cond, thn, els) {
        ret with_out_method(bind trans_if(cx, cond, thn, els, e.id, _), cx,
                            e.id, output);
      }
      ast::expr_ternary(_, _, _) {
        ret trans_expr_out(cx, ast::ternary_to_if(e), output);
      }
      ast::expr_for(decl, seq, body) { ret trans_for(cx, decl, seq, body); }
      ast::expr_for_each(decl, seq, body) {
        ret trans_for_each(cx, decl, seq, body);
      }
      ast::expr_while(cond, body) { ret trans_while(cx, cond, body); }
      ast::expr_do_while(body, cond) { ret trans_do_while(cx, body, cond); }
      ast::expr_alt(expr, arms) {
        ret with_out_method(bind trans_alt::trans_alt(cx, expr, arms, e.id,
                                                      _), cx, e.id, output);
      }
      ast::expr_fn(f) {
        let ccx = bcx_ccx(cx);
        let llfnty: TypeRef =
            type_of_fn_from_ty(ccx, e.span, node_id_type(ccx, e.id), 0u);
        let sub_cx = extend_path(cx.fcx.lcx, ccx.names.next("anon"));
        let s = mangle_internal_name_by_path(ccx, sub_cx.path);
        let llfn = decl_internal_fastcall_fn(ccx.llmod, s, llfnty);

        let fn_res =
            trans_closure(some(cx), some(llfnty), sub_cx, e.span, f, llfn,
                          none, ~[], e.id);
        let fn_pair =
            alt fn_res {
              some(fn_pair) { fn_pair }
              none. { {fn_pair: create_fn_pair(ccx, s, llfnty, llfn, false),
                       bcx: cx} }
            };
        ret rslt(fn_pair.bcx, fn_pair.fn_pair);
      }
      ast::expr_block(blk) {
        let sub_cx = new_scope_block_ctxt(cx, "block-expr body");
        let next_cx = new_sub_block_ctxt(cx, "next");
        let sub =
            with_out_method(bind trans_block(sub_cx, blk, _), cx, e.id,
                            output);
        cx.build.Br(sub_cx.llbb);
        sub.bcx.build.Br(next_cx.llbb);
        ret rslt(next_cx, sub.val);
      }
      ast::expr_move(dst, src) {
        let lhs_res = trans_lval(cx, dst);
        assert (lhs_res.is_mem);
        // FIXME Fill in lhs_res.res.bcx.sp

        let rhs_res = trans_lval(lhs_res.res.bcx, src);
        let t = ty::expr_ty(bcx_tcx(cx), src);
        // FIXME: calculate copy init-ness in typestate.

        let move_res =
            move_val(rhs_res.res.bcx, DROP_EXISTING, lhs_res.res.val, rhs_res,
                     t);
        ret rslt(move_res.bcx, C_nil());
      }
      ast::expr_assign(dst, src) {
        let lhs_res = trans_lval(cx, dst);
        assert (lhs_res.is_mem);
        // FIXME Fill in lhs_res.res.bcx.sp
        let rhs = trans_lval(lhs_res.res.bcx, src);
        let t = ty::expr_ty(bcx_tcx(cx), src);
        // FIXME: calculate copy init-ness in typestate.
        let copy_res =
            move_val_if_temp(rhs.res.bcx, DROP_EXISTING, lhs_res.res.val, rhs,
                             t);
        ret rslt(copy_res.bcx, C_nil());
      }
      ast::expr_swap(dst, src) {
        let lhs_res = trans_lval(cx, dst);
        assert (lhs_res.is_mem);
        // FIXME Fill in lhs_res.res.bcx.sp

        let rhs_res = trans_lval(lhs_res.res.bcx, src);
        let t = ty::expr_ty(bcx_tcx(cx), src);
        let tmp_res = alloc_ty(rhs_res.res.bcx, t);
        // Swap through a temporary.

        let move1_res =
            memmove_ty(tmp_res.bcx, tmp_res.val, lhs_res.res.val, t);
        let move2_res =
            memmove_ty(move1_res.bcx, lhs_res.res.val, rhs_res.res.val, t);
        let move3_res =
            memmove_ty(move2_res.bcx, rhs_res.res.val, tmp_res.val, t);
        ret rslt(move3_res.bcx, C_nil());
      }
      ast::expr_assign_op(op, dst, src) {
        let t = ty::expr_ty(bcx_tcx(cx), src);
        let lhs_res = trans_lval(cx, dst);
        assert (lhs_res.is_mem);
        // FIXME Fill in lhs_res.res.bcx.sp

        let rhs_res = trans_expr(lhs_res.res.bcx, src);
        if ty::type_is_sequence(bcx_tcx(cx), t) {
            alt op {
              ast::add. {
                if ty::sequence_is_interior(bcx_tcx(cx), t) {
                    ret ivec::trans_append(rhs_res.bcx, t, lhs_res.res.val,
                                           rhs_res.val);
                }
                ret trans_vec_append(rhs_res.bcx, t, lhs_res.res.val,
                                     rhs_res.val);
              }
              _ { }
            }
        }
        let lhs_val = load_if_immediate(rhs_res.bcx, lhs_res.res.val, t);
        let v = trans_eager_binop(rhs_res.bcx, op, lhs_val, t,
                                  rhs_res.val, t);
        // FIXME: calculate copy init-ness in typestate.
        // This is always a temporary, so can always be safely moved
        let move_res =
            move_val(v.bcx, DROP_EXISTING, lhs_res.res.val,
                     lval_val(v.bcx, v.val), t);
        ret rslt(move_res.bcx, C_nil());
      }
      ast::expr_bind(f, args) { ret trans_bind(cx, f, args, e.id); }
      ast::expr_call(f, args) {
        ret trans_call(cx, f, none[ValueRef], args, e.id);
      }
      ast::expr_cast(val, _) { ret trans_cast(cx, val, e.id); }
      ast::expr_vec(args, _, ast::sk_rc.) { ret trans_vec(cx, args, e.id); }
      ast::expr_vec(args, _, ast::sk_unique.) {
        ret trans_ivec(cx, args, e.id);
      }
      ast::expr_rec(args, base) { ret trans_rec(cx, args, base, e.id); }
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
        let cond = cx.build.Load(c);

        let then_cx = new_scope_block_ctxt(cx, "claim_then");
        let check_res = trans_check_expr(then_cx, a, "Claim");
        let else_cx = new_scope_block_ctxt(cx, "else");
        let els = rslt(else_cx, C_nil());

        cx.build.CondBr(cond, then_cx.llbb, else_cx.llbb);
        ret rslt(join_branches(cx, ~[check_res, els]), C_nil());
      }
      ast::expr_break. { ret trans_break(e.span, cx); }
      ast::expr_cont. { ret trans_cont(e.span, cx); }
      ast::expr_ret(ex) { ret trans_ret(cx, ex); }
      ast::expr_put(ex) { ret trans_put(cx, ex); }
      ast::expr_be(ex) { ret trans_be(cx, ex); }
      ast::expr_port(_) { ret trans_port(cx, e.id); }
      ast::expr_chan(ex) { ret trans_chan(cx, ex, e.id); }
      ast::expr_send(lhs, rhs) { ret trans_send(cx, lhs, rhs, e.id); }
      ast::expr_recv(lhs, rhs) { ret trans_recv(cx, lhs, rhs, e.id); }
      ast::expr_spawn(dom, name, func, args) {
        ret trans_spawn(cx, dom, name, func, args, e.id);
      }
      ast::expr_anon_obj(anon_obj) {
        ret trans_anon_obj(cx, e.span, anon_obj, e.id);
      }
      _ {
        // The expression is an lvalue. Fall through.
        assert (ty::is_lval(e));
        // make sure it really is and that we
        // didn't forget to add a case for a new expr!
      }
    }
    // lval cases fall through to trans_lval and then
    // possibly load the result (if it's non-structural).

    let t = ty::expr_ty(bcx_tcx(cx), e);
    let sub = trans_lval(cx, e);
    let v = sub.res.val;
    if sub.is_mem { v = load_if_immediate(sub.res.bcx, v, t); }
    ret rslt(sub.res.bcx, v);
}

fn with_out_method(work: fn(&out_method) -> result , cx: @block_ctxt,
                   id: ast::node_id, outer_output: &out_method) -> result {
    let ccx = bcx_ccx(cx);
    if outer_output != return {
        ret work(outer_output);
    } else {
        let tp = node_id_type(ccx, id);
        if ty::type_is_nil(ccx.tcx, tp) { ret work(return); }
        let res_alloca = alloc_ty(cx, tp);
        cx = zero_alloca(res_alloca.bcx, res_alloca.val, tp).bcx;
        fn drop_hoisted_ty(cx: &@block_ctxt, target: ValueRef, t: ty::t) ->
           result {
            let reg_val = load_if_immediate(cx, target, t);
            ret drop_ty(cx, reg_val, t);
        }
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
fn type_is_immediate(ccx: &@crate_ctxt, t: &ty::t) -> bool {
    ret ty::type_is_scalar(ccx.tcx, t) || ty::type_is_boxed(ccx.tcx, t) ||
            ty::type_is_native(ccx.tcx, t);
}

fn do_spill(cx: &@block_ctxt, v: ValueRef) -> ValueRef {
    // We have a value but we have to spill it to pass by alias.
    let llptr = alloca(cx, val_ty(v));
    cx.build.Store(v, llptr);
    ret llptr;
}

fn spill_if_immediate(cx: &@block_ctxt, v: ValueRef, t: &ty::t) -> ValueRef {
    if type_is_immediate(bcx_ccx(cx), t) { ret do_spill(cx, v); }
    ret v;
}

fn load_if_immediate(cx: &@block_ctxt, v: ValueRef, t: &ty::t) -> ValueRef {
    if type_is_immediate(bcx_ccx(cx), t) { ret cx.build.Load(v); }
    ret v;
}

fn trans_log(lvl: int, cx: &@block_ctxt, e: &@ast::expr) -> result {
    let lcx = cx.fcx.lcx;
    let modname = str::connect_ivec(lcx.module_path, "::");
    let global;
    if lcx.ccx.module_data.contains_key(modname) {
        global = lcx.ccx.module_data.get(modname);
    } else {
        let s =
            link::mangle_internal_name_by_path_and_seq(lcx.ccx,
                                                       lcx.module_path,
                                                       "loglevel");
        global = llvm::LLVMAddGlobal(lcx.ccx.llmod, T_int(), str::buf(s));
        llvm::LLVMSetGlobalConstant(global, False);
        llvm::LLVMSetInitializer(global, C_null(T_int()));
        llvm::LLVMSetLinkage(global,
                             lib::llvm::LLVMInternalLinkage as llvm::Linkage);
        lcx.ccx.module_data.insert(modname, global);
    }
    let log_cx = new_scope_block_ctxt(cx, "log");
    let after_cx = new_sub_block_ctxt(cx, "after");
    let load = cx.build.Load(global);
    let test = cx.build.ICmp(lib::llvm::LLVMIntSGE, load, C_int(lvl));
    cx.build.CondBr(test, log_cx.llbb, after_cx.llbb);
    let sub = trans_expr(log_cx, e);
    let e_ty = ty::expr_ty(bcx_tcx(cx), e);
    let log_bcx = sub.bcx;
    if ty::type_is_fp(bcx_tcx(cx), e_ty) {
        let tr: TypeRef;
        let is32bit: bool = false;
        alt ty::struct(bcx_tcx(cx), e_ty) {
          ty::ty_machine(ast::ty_f32.) { tr = T_f32(); is32bit = true; }
          ty::ty_machine(ast::ty_f64.) { tr = T_f64(); }
          _ { tr = T_float(); }
        }
        if is32bit {
            log_bcx.build.Call(bcx_ccx(log_bcx).upcalls.log_float,
                               ~[log_bcx.fcx.lltaskptr, C_int(lvl), sub.val]);
        } else {
            // FIXME: Eliminate this level of indirection.

            let tmp = alloca(log_bcx, tr);
            sub.bcx.build.Store(sub.val, tmp);
            log_bcx.build.Call(bcx_ccx(log_bcx).upcalls.log_double,
                               ~[log_bcx.fcx.lltaskptr, C_int(lvl), tmp]);
        }
    } else if (ty::type_is_integral(bcx_tcx(cx), e_ty) ||
                   ty::type_is_bool(bcx_tcx(cx), e_ty)) {
        // FIXME: Handle signedness properly.

        let llintval =
            int_cast(log_bcx, T_int(), val_ty(sub.val), sub.val, false);
        log_bcx.build.Call(bcx_ccx(log_bcx).upcalls.log_int,
                           ~[log_bcx.fcx.lltaskptr, C_int(lvl), llintval]);
    } else {
        alt ty::struct(bcx_tcx(cx), e_ty) {
          ty::ty_str. {
            log_bcx.build.Call(bcx_ccx(log_bcx).upcalls.log_str,
                               ~[log_bcx.fcx.lltaskptr, C_int(lvl), sub.val]);
          }
          _ {
            // FIXME: Support these types.

            bcx_ccx(cx).sess.span_fatal(e.span,
                                        "log called on unsupported type " +
                                            ty_to_str(bcx_tcx(cx), e_ty));
          }
        }
    }
    log_bcx = trans_block_cleanups(log_bcx, log_cx);
    log_bcx.build.Br(after_cx.llbb);
    ret rslt(after_cx, C_nil());
}

fn trans_check_expr(cx: &@block_ctxt, e: &@ast::expr, s: &str) -> result {
    let cond_res = trans_expr(cx, e);
    let expr_str = s + " " + expr_to_str(e) + " failed";
    let fail_cx = new_sub_block_ctxt(cx, "fail");
    trans_fail(fail_cx, some[span](e.span), expr_str);
    let next_cx = new_sub_block_ctxt(cx, "next");
    cond_res.bcx.build.CondBr(cond_res.val, next_cx.llbb, fail_cx.llbb);
    ret rslt(next_cx, C_nil());
}

fn trans_fail_expr(cx: &@block_ctxt, sp_opt: &option::t[span],
                   fail_expr: &option::t[@ast::expr]) -> result {
    let bcx = cx;
    alt fail_expr {
      some(expr) {
        let tcx = bcx_tcx(bcx);
        let expr_res = trans_expr(bcx, expr);
        let e_ty = ty::expr_ty(tcx, expr);
        bcx = expr_res.bcx;


        if ty::type_is_str(tcx, e_ty) {
            let elt =
                bcx.build.GEP(expr_res.val,
                              ~[C_int(0), C_int(abi::vec_elt_data)]);
            ret trans_fail_value(bcx, sp_opt, elt);
        } else {
            bcx_ccx(cx).sess.span_bug(expr.span,
                                      "fail called with unsupported \
                                             type "
                                          + ty_to_str(tcx, e_ty));
        }
      }
      _ { ret trans_fail(bcx, sp_opt, "explicit failure"); }
    }
}

fn trans_fail(cx: &@block_ctxt, sp_opt: &option::t[span], fail_str: &str) ->
   result {
    let V_fail_str = C_cstr(bcx_ccx(cx), fail_str);
    ret trans_fail_value(cx, sp_opt, V_fail_str);
}

fn trans_fail_value(cx: &@block_ctxt, sp_opt: &option::t[span],
                    V_fail_str: &ValueRef) -> result {
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
    let V_str = cx.build.PointerCast(V_fail_str, T_ptr(T_i8()));
    V_filename = cx.build.PointerCast(V_filename, T_ptr(T_i8()));
    let args = ~[cx.fcx.lltaskptr, V_str, V_filename, C_int(V_line)];
    cx.build.Call(bcx_ccx(cx).upcalls._fail, args);
    cx.build.Unreachable();
    ret rslt(cx, C_nil());
}

fn trans_put(cx: &@block_ctxt, e: &option::t[@ast::expr]) -> result {
    let llcallee = C_nil();
    let llenv = C_nil();
    alt { cx.fcx.lliterbody } {
      some(lli) {
        let slot = alloca(cx, val_ty(lli));
        cx.build.Store(lli, slot);
        llcallee = cx.build.GEP(slot, ~[C_int(0), C_int(abi::fn_field_code)]);
        llcallee = cx.build.Load(llcallee);
        llenv = cx.build.GEP(slot, ~[C_int(0), C_int(abi::fn_field_box)]);
        llenv = cx.build.Load(llenv);
      }
    }
    let bcx = cx;
    let dummy_retslot = alloca(bcx, T_nil());
    let llargs: ValueRef[] = ~[dummy_retslot, cx.fcx.lltaskptr, llenv];
    alt e {
      none. { }
      some(x) {
        let e_ty = ty::expr_ty(bcx_tcx(cx), x);
        let arg = {mode: ty::mo_alias(false), ty: e_ty};
        let arg_tys = type_of_explicit_args(bcx_ccx(cx), x.span, ~[arg]);
        let r = trans_arg_expr(bcx, arg, arg_tys.(0), x);
        bcx = r.bcx;
        llargs += ~[r.val];
      }
    }
    bcx.build.FastCall(llcallee, llargs);
    ret rslt(bcx, C_nil());
}

fn trans_break_cont(sp: &span, cx: &@block_ctxt, to_end: bool) -> result {
    let bcx = cx;
    // Locate closest loop block, outputting cleanup as we go.

    let cleanup_cx = cx;
    while true {
        bcx = trans_block_cleanups(bcx, cleanup_cx);
        alt { cleanup_cx.kind } {
          LOOP_SCOPE_BLOCK(_cont, _break) {
            if to_end {
                bcx.build.Br(_break.llbb);
            } else {
                alt _cont {
                  option::some(_cont) { bcx.build.Br(_cont.llbb); }
                  _ { bcx.build.Br(cleanup_cx.llbb); }
                }
            }
            ret rslt(new_sub_block_ctxt(bcx, "break_cont.unreachable"),
                     C_nil());
          }
          _ {
            alt { cleanup_cx.parent } {
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

fn trans_break(sp: &span, cx: &@block_ctxt) -> result {
    ret trans_break_cont(sp, cx, true);
}

fn trans_cont(sp: &span, cx: &@block_ctxt) -> result {
    ret trans_break_cont(sp, cx, false);
}

fn trans_ret(cx: &@block_ctxt, e: &option::t[@ast::expr]) -> result {
    let bcx = cx;
    alt e {
      some(x) {
        let t = ty::expr_ty(bcx_tcx(cx), x);
        let lv = trans_lval(cx, x);
        bcx = lv.res.bcx;
        bcx = move_val_if_temp(bcx, INIT, cx.fcx.llretptr, lv, t).bcx;
      }
      _ {
        let t = llvm::LLVMGetElementType(val_ty(cx.fcx.llretptr));
        bcx.build.Store(C_null(t), cx.fcx.llretptr);
      }
    }
    // run all cleanups and back out.

    let more_cleanups: bool = true;
    let cleanup_cx = cx;
    while more_cleanups {
        bcx = trans_block_cleanups(bcx, cleanup_cx);
        alt { cleanup_cx.parent } {
          parent_some(b) { cleanup_cx = b; }
          parent_none. { more_cleanups = false; }
        }
    }
    bcx.build.RetVoid();
    ret rslt(new_sub_block_ctxt(bcx, "ret.unreachable"), C_nil());
}

fn trans_be(cx: &@block_ctxt, e: &@ast::expr) -> result {
    // FIXME: This should be a typestate precondition

    assert (ast::is_call_expr(e));
    // FIXME: Turn this into a real tail call once
    // calling convention issues are settled

    ret trans_ret(cx, some(e));
}

/*

  Suppose we create an anonymous object my_b from a regular object a:

        obj a() {
            fn foo() -> int {
                ret 2;
            }
            fn bar() -> int {
                ret self.foo();
            }
        }

       auto my_a = a();
       auto my_b = obj { fn baz() -> int { ret self.foo() } with my_a };

  Here we're extending the my_a object with an additional method baz, creating
  an object my_b. Since it's an object, my_b is a pair of a vtable pointer and
  a body pointer:

  my_b: [vtbl* | body*]

  my_b's vtable has entries for foo, bar, and baz, whereas my_a's vtable has
  only foo and bar. my_b's 3-entry vtable consists of two forwarding functions
  and one real method.

  my_b's body just contains the pair a: [ a_vtable | a_body ], wrapped up with
  any additional fields that my_b added. None were added, so my_b is just the
  wrapped inner object.

*/

// trans_anon_obj: create and return a pointer to an object.  This code
// differs from trans_obj in that, rather than creating an object constructor
// function and putting it in the generated code as an object item, we are
// instead "inlining" the construction of the object and returning the object
// itself.
fn trans_anon_obj(bcx: @block_ctxt, sp: &span, anon_obj: &ast::anon_obj,
                  id: ast::node_id) -> result {


    let ccx = bcx_ccx(bcx);

    // Fields.
    // FIXME (part of issue #538): Where do we fill in the field *values* from
    // the outer object?
    let additional_fields: ast::anon_obj_field[] = ~[];
    let additional_field_vals: result[] = ~[];
    let additional_field_tys: ty::t[] = ~[];
    alt anon_obj.fields {
      none. { }
      some(fields) {
        additional_fields = fields;
        for f: ast::anon_obj_field  in fields {
            additional_field_tys += ~[node_id_type(ccx, f.id)];
            additional_field_vals += ~[trans_expr(bcx, f.expr)];
        }
      }
    }

    // Get the type of the eventual entire anonymous object, possibly with
    // extensions.  NB: This type includes both inner and outer methods.
    let outer_obj_ty = ty::node_id_to_type(ccx.tcx, id);

    // Create a vtable for the anonymous object.

    // create_vtbl() wants an ast::_obj and all we have is an ast::anon_obj,
    // so we need to roll our own.
    let wrapper_obj: ast::_obj =
        {fields:
             std::ivec::map(ast::obj_field_from_anon_obj_field,
                            additional_fields),
         methods: anon_obj.methods};

    let inner_obj_ty: ty::t;
    let vtbl;
    alt anon_obj.inner_obj {
      none. {
        // We need a dummy inner_obj_ty for setting up the object body
        // later.
        inner_obj_ty = ty::mk_type(ccx.tcx);

        // If there's no inner_obj -- that is, if we're just adding new
        // fields rather than extending an existing object -- then we just
        // pass the outer object to create_vtbl().  Our vtable won't need
        // to have any forwarding slots.
        vtbl =
            create_vtbl(bcx.fcx.lcx, sp, outer_obj_ty, wrapper_obj, ~[], none,
                        additional_field_tys);
      }
      some(e) {
        // TODO: What makes more sense to get the type of an expr --
        // calling ty::expr_ty(ccx.tcx, e) on it or calling
        // ty::node_id_to_type(ccx.tcx, id) on its id?
        inner_obj_ty = ty::expr_ty(ccx.tcx, e);
        //inner_obj_ty = ty::node_id_to_type(ccx.tcx, e.id);

        // If there's a inner_obj, we pass its type along to create_vtbl().
        // Part of what create_vtbl() will do is take the set difference
        // of methods defined on the original and methods being added.
        // For every method defined on the original that does *not* have
        // one with a matching name and type being added, we'll need to
        // create a forwarding slot.  And, of course, we need to create a
        // normal vtable entry for every method being added.
        vtbl =
            create_vtbl(bcx.fcx.lcx, sp, outer_obj_ty, wrapper_obj, ~[],
                        some(inner_obj_ty), additional_field_tys);
      }
    }

    // Allocate the object that we're going to return.
    let pair = alloca(bcx, ccx.rust_object_type);

    // Take care of cleanups.
    let t = node_id_type(ccx, id);
    add_clean_temp(bcx, pair, t);

    // Grab onto the first and second elements of the pair.
    // abi::obj_field_vtbl and abi::obj_field_box simply specify words 0 and 1
    // of 'pair'.
    let pair_vtbl =
        bcx.build.GEP(pair, ~[C_int(0), C_int(abi::obj_field_vtbl)]);
    let pair_box =
        bcx.build.GEP(pair, ~[C_int(0), C_int(abi::obj_field_box)]);

    vtbl = bcx.build.PointerCast(vtbl, T_ptr(T_empty_struct()));
    bcx.build.Store(vtbl, pair_vtbl);

    // Next we have to take care of the other half of the pair we're
    // returning: a boxed (reference-counted) tuple containing a tydesc,
    // typarams, fields, and a pointer to our inner_obj.
    let llbox_ty: TypeRef = T_ptr(T_empty_struct());

    if std::ivec::len[ast::anon_obj_field](additional_fields) == 0u &&
           anon_obj.inner_obj == none {
        // If the object we're translating has no fields and no inner_obj,
        // there's not much to do.
        bcx.build.Store(C_null(llbox_ty), pair_box);
    } else {

        // Synthesize a tuple type for fields: [field, ...]
        let fields_ty: ty::t = ty::mk_imm_tup(ccx.tcx, additional_field_tys);

        // Type for tydescs.
        let tydesc_ty: ty::t = ty::mk_type(ccx.tcx);

        // Placeholder for non-existent typarams, since anon objs don't have
        // them.
        let typarams_ty: ty::t = ty::mk_imm_tup(ccx.tcx, ~[]);

        // Tuple type for body:
        // [tydesc, [typaram, ...], [field, ...], inner_obj]
        let body_ty: ty::t =
            ty::mk_imm_tup(ccx.tcx,
                           ~[tydesc_ty, typarams_ty, fields_ty,
                             inner_obj_ty]);

        // Hand this type we've synthesized off to trans_malloc_boxed, which
        // allocates a box, including space for a refcount.
        let box = trans_malloc_boxed(bcx, body_ty);
        bcx = box.bcx;
        let body = box.body;

        // Put together a tydesc for the body, so that the object can later be
        // freed by calling through its tydesc.

        // Every object (not just those with type parameters) needs to have a
        // tydesc to describe its body, since all objects have unknown type to
        // the user of the object.  So the tydesc is needed to keep track of
        // the types of the object's fields, so that the fields can be freed
        // later.
        let body_tydesc =
            GEP_tup_like(bcx, body_ty, body,
                         ~[0, abi::obj_body_elt_tydesc]);
        bcx = body_tydesc.bcx;
        let ti = none[@tydesc_info];
        let body_td = get_tydesc(bcx, body_ty, true, ti);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, ti);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, ti);
        bcx = body_td.bcx;
        bcx.build.Store(body_td.val, body_tydesc.val);

        // Copy the object's fields into the space we allocated for the object
        // body.  (This is something like saving the lexical environment of a
        // function in its closure: the fields were passed to the object
        // constructor and are now available to the object's methods.
        let body_fields =
            GEP_tup_like(bcx, body_ty, body,
                         ~[0, abi::obj_body_elt_fields]);
        bcx = body_fields.bcx;
        let i: int = 0;
        for f: ast::anon_obj_field  in additional_fields {
            // FIXME (part of issue #538): make this work eventually, when we
            // have additional field exprs in the AST.
            load_if_immediate(bcx, additional_field_vals.(i).val,
                              additional_field_tys.(i));

            let field =
                GEP_tup_like(bcx, fields_ty, body_fields.val, ~[0, i]);
            bcx = field.bcx;
            bcx =
                copy_val(bcx, INIT, field.val, additional_field_vals.(i).val,
                         additional_field_tys.(i)).bcx;
            i += 1;
        }

        // If there's a inner_obj, copy a pointer to it into the object's
        // body.
        alt anon_obj.inner_obj {
          none. { }
          some(e) {
            // If inner_obj (the object being extended) exists, translate it.
            // Translating inner_obj returns a ValueRef (pointer to a 2-word
            // value) wrapped in a result.
            let inner_obj_val: result = trans_expr(bcx, e);

            let body_inner_obj =
                GEP_tup_like(bcx, body_ty, body,
                             ~[0, abi::obj_body_elt_inner_obj]);
            bcx = body_inner_obj.bcx;
            bcx =
                copy_val(bcx, INIT, body_inner_obj.val, inner_obj_val.val,
                         inner_obj_ty).bcx;
          }
        }

        // Store box ptr in outer pair.
        let p = bcx.build.PointerCast(box.box, llbox_ty);
        bcx.build.Store(p, pair_box);
    }

    // return the object we built.
    ret rslt(bcx, pair);
}

fn init_local(bcx: @block_ctxt, local: &@ast::local) -> result {
    let ty = node_id_type(bcx_ccx(bcx), local.node.id);
    let llptr = bcx.fcx.lllocals.get(local.node.id);
    // Make a note to drop this slot on the way out.
    add_clean(bcx, llptr, ty);
    alt local.node.init {
      some(init) {
        alt init.op {
          ast::init_assign. {
            // Use the type of the RHS because if it's _|_, the LHS
            // type might be something else, but we don't want to copy
            // the value.
            ty = node_id_type(bcx_ccx(bcx), init.expr.id);
            let sub = trans_lval(bcx, init.expr);
            bcx = move_val_if_temp(sub.res.bcx, INIT, llptr, sub, ty).bcx;
          }
          ast::init_move. {
            let sub = trans_lval(bcx, init.expr);
            bcx = move_val(sub.res.bcx, INIT, llptr, sub, ty).bcx;
          }
        }
      }
      _ { bcx = zero_alloca(bcx, llptr, ty).bcx; }
    }
    bcx = trans_alt::bind_irrefutable_pat(bcx, local.node.pat, llptr,
                                          bcx.fcx.lllocals, false);
    ret rslt(bcx, llptr);
}

fn zero_alloca(cx: &@block_ctxt, llptr: ValueRef, t: ty::t) -> result {
    let bcx = cx;
    if ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        let llsz = size_of(bcx, t);
        let llalign = align_of(llsz.bcx, t);
        bcx = call_bzero(llalign.bcx, llptr, llsz.val, llalign.val).bcx;
    } else {
        let llty = type_of(bcx_ccx(bcx), cx.sp, t);
        bcx.build.Store(C_null(llty), llptr);
    }
    ret rslt(bcx, llptr);
}

fn trans_stmt(cx: &@block_ctxt, s: &ast::stmt) -> result {
    // FIXME Fill in cx.sp

    let bcx = cx;
    alt s.node {
      ast::stmt_expr(e, _) { bcx = trans_expr(cx, e).bcx; }
      ast::stmt_decl(d, _) {
        alt d.node {
          ast::decl_local(locals) {
            for local: @ast::local  in locals {
                bcx = init_local(bcx, local).bcx;
            }
          }
          ast::decl_item(i) { trans_item(cx.fcx.lcx, *i); }
        }
      }
      _ { bcx_ccx(cx).sess.unimpl("stmt variant"); }
    }
    ret rslt(bcx, C_nil());
}

// You probably don't want to use this one. See the
// next three functions instead.
fn new_block_ctxt(cx: &@fn_ctxt, parent: &block_parent, kind: block_kind,
                  name: &str) -> @block_ctxt {
    let cleanups: cleanup[] = ~[];
    let s = str::buf("");
    let held_name; //HACK for str::buf, which doesn't keep its value alive
    if cx.lcx.ccx.sess.get_opts().save_temps ||
           cx.lcx.ccx.sess.get_opts().debuginfo {
        held_name = cx.lcx.ccx.names.next(name);
        s = str::buf(held_name);
    }
    let llbb: BasicBlockRef = llvm::LLVMAppendBasicBlock(cx.llfn, s);
    ret @{llbb: llbb,
          build: new_builder(llbb),
          parent: parent,
          kind: kind,
          mutable cleanups: cleanups,
          sp: cx.sp,
          fcx: cx};
}


// Use this when you're at the top block of a function or the like.
fn new_top_block_ctxt(fcx: &@fn_ctxt) -> @block_ctxt {
    ret new_block_ctxt(fcx, parent_none, SCOPE_BLOCK, "function top level");
}


// Use this when you're at a curly-brace or similar lexical scope.
fn new_scope_block_ctxt(bcx: &@block_ctxt, n: &str) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, parent_some(bcx), SCOPE_BLOCK, n);
}

fn new_loop_scope_block_ctxt(bcx: &@block_ctxt,
                             _cont: &option::t[@block_ctxt],
                             _break: &@block_ctxt, n: &str) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, parent_some(bcx),
                       LOOP_SCOPE_BLOCK(_cont, _break), n);
}


// Use this when you're making a general CFG BB within a scope.
fn new_sub_block_ctxt(bcx: &@block_ctxt, n: &str) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, parent_some(bcx), NON_SCOPE_BLOCK, n);
}

fn new_raw_block_ctxt(fcx: &@fn_ctxt, llbb: BasicBlockRef) -> @block_ctxt {
    let cleanups: cleanup[] = ~[];
    ret @{llbb: llbb,
          build: new_builder(llbb),
          parent: parent_none,
          kind: NON_SCOPE_BLOCK,
          mutable cleanups: cleanups,
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
fn trans_block_cleanups(cx: &@block_ctxt, cleanup_cx: &@block_ctxt) ->
   @block_ctxt {
    let bcx = cx;
    if cleanup_cx.kind == NON_SCOPE_BLOCK {
        assert (std::ivec::len[cleanup](cleanup_cx.cleanups) == 0u);
    }
    let i = std::ivec::len[cleanup](cleanup_cx.cleanups);
    while i > 0u {
        i -= 1u;
        let c = cleanup_cx.cleanups.(i);
        alt c {
          clean(cfn) { bcx = cfn(bcx).bcx; }
          clean_temp(_, cfn) { bcx = cfn(bcx).bcx; }
        }
    }
    ret bcx;
}

iter block_locals(b: &ast::blk) -> @ast::local {
    // FIXME: putting from inside an iter block doesn't work, so we can't
    // use the index here.
    for s: @ast::stmt  in b.node.stmts {
        alt s.node {
          ast::stmt_decl(d, _) {
            alt d.node {
              ast::decl_local(locals) {
                for local: @ast::local in locals { put local; }
              }
              _ {/* fall through */ }
            }
          }
          _ {/* fall through */ }
        }
    }
}

fn llstaticallocas_block_ctxt(fcx: &@fn_ctxt) -> @block_ctxt {
    let cleanups: cleanup[] = ~[];
    ret @{llbb: fcx.llstaticallocas,
          build: new_builder(fcx.llstaticallocas),
          parent: parent_none,
          kind: SCOPE_BLOCK,
          mutable cleanups: cleanups,
          sp: fcx.sp,
          fcx: fcx};
}

fn llderivedtydescs_block_ctxt(fcx: &@fn_ctxt) -> @block_ctxt {
    let cleanups: cleanup[] = ~[];
    ret @{llbb: fcx.llderivedtydescs,
          build: new_builder(fcx.llderivedtydescs),
          parent: parent_none,
          kind: SCOPE_BLOCK,
          mutable cleanups: cleanups,
          sp: fcx.sp,
          fcx: fcx};
}

fn lldynamicallocas_block_ctxt(fcx: &@fn_ctxt) -> @block_ctxt {
    let cleanups: cleanup[] = ~[];
    ret @{llbb: fcx.lldynamicallocas,
          build: new_builder(fcx.lldynamicallocas),
          parent: parent_none,
          kind: SCOPE_BLOCK,
          mutable cleanups: cleanups,
          sp: fcx.sp,
          fcx: fcx};
}



fn alloc_ty(cx: &@block_ctxt, t: &ty::t) -> result {
    let val = C_int(0);
    if ty::type_has_dynamic_size(bcx_tcx(cx), t) {
        // NB: we have to run this particular 'size_of' in a
        // block_ctxt built on the llderivedtydescs block for the fn,
        // so that the size dominates the array_alloca that
        // comes next.

        let n = size_of(llderivedtydescs_block_ctxt(cx.fcx), t);
        cx.fcx.llderivedtydescs = n.bcx.llbb;
        val = array_alloca(cx, T_i8(), n.val);
    } else { val = alloca(cx, type_of(bcx_ccx(cx), cx.sp, t)); }
    // NB: since we've pushed all size calculations in this
    // function up to the alloca block, we actually return the
    // block passed into us unmodified; it doesn't really
    // have to be passed-and-returned here, but it fits
    // past caller conventions and may well make sense again,
    // so we leave it as-is.

    ret rslt(cx, val);
}

fn alloc_local(cx: &@block_ctxt, local: &@ast::local) -> result {
    let t = node_id_type(bcx_ccx(cx), local.node.id);
    let r = alloc_ty(cx, t);
    alt local.node.pat.node {
      ast::pat_bind(ident) {
        if bcx_ccx(cx).sess.get_opts().debuginfo {
            llvm::LLVMSetValueName(r.val, str::buf(ident));
        }
      }
      _ {}
    }
    ret r;
}

fn trans_block(cx: &@block_ctxt, b: &ast::blk, output: &out_method) ->
   result {
    let bcx = cx;
    for each local: @ast::local in block_locals(b) {
        // FIXME Update bcx.sp
        let r = alloc_local(bcx, local);
        bcx = r.bcx;
        bcx.fcx.lllocals.insert(local.node.id, r.val);
    }
    let r = rslt(bcx, C_nil());
    for s: @ast::stmt  in b.node.stmts {
        r = trans_stmt(bcx, *s);
        bcx = r.bcx;

        // If we hit a terminator, control won't go any further so
        // we're in dead-code land. Stop here.
        if is_terminated(bcx) { ret r; }
    }
    fn accept_out_method(expr: &@ast::expr) -> bool {
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
            r = lv.res;
            bcx = r.bcx;
            if is_terminated(bcx) || ty::type_is_bot(ccx.tcx, r_ty) { ret r; }
            alt output {
              save_in(target) {
                // The output method is to save the value at target,
                // and we didn't pass it to the recursive trans_expr
                // call.
                bcx = move_val_if_temp(bcx, INIT, target, lv, r_ty).bcx;
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

fn new_local_ctxt(ccx: &@crate_ctxt) -> @local_ctxt {
    let pth: str[] = ~[];
    ret @{path: pth,
          module_path: ~[ccx.link_meta.name],
          obj_typarams: ~[],
          obj_fields: ~[],
          ccx: ccx};
}


// Creates the standard quartet of basic blocks: static allocas, copy args,
// derived tydescs, and dynamic allocas.
fn mk_standard_basic_blocks(llfn: ValueRef) ->
   {sa: BasicBlockRef,
    ca: BasicBlockRef,
    dt: BasicBlockRef,
    da: BasicBlockRef} {
    ret {sa: llvm::LLVMAppendBasicBlock(llfn, str::buf("static_allocas")),
         ca: llvm::LLVMAppendBasicBlock(llfn, str::buf("copy_args")),
         dt: llvm::LLVMAppendBasicBlock(llfn, str::buf("derived_tydescs")),
         da: llvm::LLVMAppendBasicBlock(llfn, str::buf("dynamic_allocas"))};
}


// NB: must keep 4 fns in sync:
//
//  - type_of_fn_full
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
fn new_fn_ctxt_w_id(cx: @local_ctxt, sp: &span, llfndecl: ValueRef,
                    id: ast::node_id) -> @fn_ctxt {
    let llretptr: ValueRef = llvm::LLVMGetParam(llfndecl, 0u);
    let lltaskptr: ValueRef = llvm::LLVMGetParam(llfndecl, 1u);
    let llenv: ValueRef = llvm::LLVMGetParam(llfndecl, 2u);
    let llargs: hashmap[ast::node_id, ValueRef] = new_int_hash[ValueRef]();
    let llobjfields: hashmap[ast::node_id, ValueRef] =
        new_int_hash[ValueRef]();
    let lllocals: hashmap[ast::node_id, ValueRef] = new_int_hash[ValueRef]();
    let llupvars: hashmap[ast::node_id, ValueRef] = new_int_hash[ValueRef]();
    let derived_tydescs =
        map::mk_hashmap[ty::t, derived_tydesc_info](ty::hash_ty, ty::eq_ty);
    let llbbs = mk_standard_basic_blocks(llfndecl);
    ret @{llfn: llfndecl,
          lltaskptr: lltaskptr,
          llenv: llenv,
          llretptr: llretptr,
          mutable llstaticallocas: llbbs.sa,
          mutable llcopyargs: llbbs.ca,
          mutable llderivedtydescs_first: llbbs.dt,
          mutable llderivedtydescs: llbbs.dt,
          mutable lldynamicallocas: llbbs.da,
          mutable llself: none[val_self_pair],
          mutable lliterbody: none[ValueRef],
          mutable iterbodyty: none[ty::t],
          llargs: llargs,
          llobjfields: llobjfields,
          lllocals: lllocals,
          llupvars: llupvars,
          mutable lltydescs: ~[],
          derived_tydescs: derived_tydescs,
          id: id,
          sp: sp,
          lcx: cx};
}

fn new_fn_ctxt(cx: @local_ctxt, sp: &span, llfndecl: ValueRef) -> @fn_ctxt {
    be new_fn_ctxt_w_id(cx, sp, llfndecl, -1);
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn_full
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
fn create_llargs_for_fn_args(cx: &@fn_ctxt, proto: ast::proto,
                             ty_self: option::t[ty::t], ret_ty: ty::t,
                             args: &ast::arg[], ty_params: &ast::ty_param[]) {
    // Skip the implicit arguments 0, 1, and 2.  TODO: Pull out 3u and define
    // it as a constant, since we're using it in several places in trans this
    // way.
    let arg_n = 3u;
    alt ty_self {
      some(tt) { cx.llself = some[val_self_pair]({v: cx.llenv, t: tt}); }
      none. {
        let i = 0u;
        for tp: ast::ty_param  in ty_params {
            let llarg = llvm::LLVMGetParam(cx.llfn, arg_n);
            assert (llarg as int != 0);
            cx.lltydescs += ~[llarg];
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
        cx.lliterbody = some[ValueRef](llarg);
        arg_n += 1u;
    }

    // Populate the llargs field of the function context with the ValueRefs
    // that we get from llvm::LLVMGetParam for each argument.
    for arg: ast::arg  in args {
        let llarg = llvm::LLVMGetParam(cx.llfn, arg_n);
        assert (llarg as int != 0);
        cx.llargs.insert(arg.id, llarg);
        arg_n += 1u;
    }
}


// Recommended LLVM style, strange though this is, is to copy from args to
// allocas immediately upon entry; this permits us to GEP into structures we
// were passed and whatnot. Apparently mem2reg will mop up.
fn copy_any_self_to_alloca(fcx: @fn_ctxt) {
    let bcx = llstaticallocas_block_ctxt(fcx);
    alt { fcx.llself } {
      some(pair) {
        let a = alloca(bcx, fcx.lcx.ccx.rust_object_type);
        bcx.build.Store(pair.v, a);
        fcx.llself = some[val_self_pair]({v: a, t: pair.t});
      }
      _ { }
    }
}

fn copy_args_to_allocas(fcx: @fn_ctxt, args: &ast::arg[],
                        arg_tys: &ty::arg[]) {
    let bcx = new_raw_block_ctxt(fcx, fcx.llcopyargs);
    let arg_n: uint = 0u;
    for aarg: ast::arg  in args {
        if aarg.mode == ast::val {
            let argval;
            alt bcx.fcx.llargs.find(aarg.id) {
              some(x) { argval = x; }
              _ {
                bcx_ccx(bcx).sess.span_fatal
                    (aarg.ty.span, "unbound arg ID in copy_args_to_allocas");
              }
            }
            let a = do_spill(bcx, argval);

            // Overwrite the llargs entry for this arg with its alloca.
            bcx.fcx.llargs.insert(aarg.id, a);
        }
        arg_n += 1u;
    }
}

fn add_cleanups_for_args(bcx: &@block_ctxt, args: &ast::arg[],
                         arg_tys: &ty::arg[]) {
    let arg_n: uint = 0u;
    for aarg: ast::arg  in args {
        if aarg.mode == ast::val {
            let argval;
            alt bcx.fcx.llargs.find(aarg.id) {
              some(x) { argval = x; }
              _ {
                bcx_ccx(bcx).sess.span_fatal
                    (aarg.ty.span, "unbound arg ID in copy_args_to_allocas");
              }
            }
            add_clean(bcx, argval, arg_tys.(arg_n).ty);
        }
        arg_n += 1u;
    }
}

fn is_terminated(cx: &@block_ctxt) -> bool {
    let inst = llvm::LLVMGetLastInstruction(cx.llbb);
    ret llvm::LLVMIsATerminatorInst(inst) as int != 0;
}

fn arg_tys_of_fn(ccx: &@crate_ctxt, id: ast::node_id) -> ty::arg[] {
    alt ty::struct(ccx.tcx, ty::node_id_to_type(ccx.tcx, id)) {
      ty::ty_fn(_, arg_tys, _, _, _) { ret arg_tys; }
    }
}

fn populate_fn_ctxt_from_llself(fcx: @fn_ctxt, llself: val_self_pair) {
    let bcx = llstaticallocas_block_ctxt(fcx);
    let field_tys: ty::t[] = ~[];
    for f: ast::obj_field  in bcx.fcx.lcx.obj_fields {
        field_tys += ~[node_id_type(bcx_ccx(bcx), f.id)];
    }
    // Synthesize a tuple type for the fields so that GEP_tup_like() can work
    // its magic.

    let fields_tup_ty = ty::mk_imm_tup(fcx.lcx.ccx.tcx, field_tys);
    let n_typarams = std::ivec::len[ast::ty_param](bcx.fcx.lcx.obj_typarams);
    let llobj_box_ty: TypeRef = T_obj_ptr(*bcx_ccx(bcx), n_typarams);
    let box_cell =
        bcx.build.GEP(llself.v, ~[C_int(0), C_int(abi::obj_field_box)]);
    let box_ptr = bcx.build.Load(box_cell);
    box_ptr = bcx.build.PointerCast(box_ptr, llobj_box_ty);
    let obj_typarams =
        bcx.build.GEP(box_ptr,
                      ~[C_int(0), C_int(abi::box_rc_field_body),
                        C_int(abi::obj_body_elt_typarams)]);
    // The object fields immediately follow the type parameters, so we skip
    // over them to get the pointer.

    let et = llvm::LLVMGetElementType(val_ty(obj_typarams));
    let obj_fields = bcx.build.Add(vp2i(bcx, obj_typarams), llsize_of(et));
    // If we can (i.e. the type is statically sized), then cast the resulting
    // fields pointer to the appropriate LLVM type. If not, just leave it as
    // i8 *.

    if !ty::type_has_dynamic_size(fcx.lcx.ccx.tcx, fields_tup_ty) {
        let llfields_ty = type_of(fcx.lcx.ccx, fcx.sp, fields_tup_ty);
        obj_fields = vi2p(bcx, obj_fields, T_ptr(llfields_ty));
    } else { obj_fields = vi2p(bcx, obj_fields, T_ptr(T_i8())); }
    let i: int = 0;
    for p: ast::ty_param  in fcx.lcx.obj_typarams {
        let lltyparam: ValueRef =
            bcx.build.GEP(obj_typarams, ~[C_int(0), C_int(i)]);
        lltyparam = bcx.build.Load(lltyparam);
        fcx.lltydescs += ~[lltyparam];
        i += 1;
    }
    i = 0;
    for f: ast::obj_field  in fcx.lcx.obj_fields {
        let rslt = GEP_tup_like(bcx, fields_tup_ty, obj_fields, ~[0, i]);
        bcx = llstaticallocas_block_ctxt(fcx);
        let llfield = rslt.val;
        fcx.llobjfields.insert(f.id, llfield);
        i += 1;
    }
    fcx.llstaticallocas = bcx.llbb;
}


// Ties up the llstaticallocas -> llcopyargs -> llderivedtydescs ->
// lldynamicallocas -> lltop edges.
fn finish_fn(fcx: &@fn_ctxt, lltop: BasicBlockRef) {
    new_builder(fcx.llstaticallocas).Br(fcx.llcopyargs);
    new_builder(fcx.llcopyargs).Br(fcx.llderivedtydescs_first);
    new_builder(fcx.llderivedtydescs).Br(fcx.lldynamicallocas);
    new_builder(fcx.lldynamicallocas).Br(lltop);
}

// trans_closure: Builds an LLVM function out of a source function.
// If the function closes over its environment a closure will be
// returned.
fn trans_closure(bcx_maybe: &option::t[@block_ctxt],
                 llfnty: &option::t[TypeRef], cx: @local_ctxt, sp: &span,
                 f: &ast::_fn, llfndecl: ValueRef, ty_self: option::t[ty::t],
                 ty_params: &ast::ty_param[], id: ast::node_id)
    -> option::t[{fn_pair: ValueRef, bcx: @block_ctxt}] {
    set_uwtable(llfndecl);

    // Set up arguments to the function.
    let fcx = new_fn_ctxt_w_id(cx, sp, llfndecl, id);
    create_llargs_for_fn_args(fcx, f.proto, ty_self,
                              ty::ret_ty_of_fn(cx.ccx.tcx, id), f.decl.inputs,
                              ty_params);
    copy_any_self_to_alloca(fcx);
    alt { fcx.llself } {
      some(llself) { populate_fn_ctxt_from_llself(fcx, llself); }
      _ { }
    }
    let arg_tys = arg_tys_of_fn(fcx.lcx.ccx, id);
    copy_args_to_allocas(fcx, f.decl.inputs, arg_tys);

    // Figure out if we need to build a closure and act accordingly
    let res = alt f.proto {
      ast::proto_block. | ast::proto_closure. {
        let bcx = option::get(bcx_maybe);
        let upvars = get_freevars(cx.ccx.tcx, id);

        let copying = f.proto == ast::proto_closure;
        let env = build_closure(bcx, upvars, copying);
        load_environment(bcx, fcx, env.ptrty, upvars, copying);

        let closure = create_real_fn_pair(env.bcx, option::get(llfnty),
                                          llfndecl, env.ptr);
        some({fn_pair: closure, bcx: env.bcx})
      }
      _ { none }
    };

    // Create the first basic block in the function and keep a handle on it to
    //  pass to finish_fn later.
    let bcx = new_top_block_ctxt(fcx);
    add_cleanups_for_args(bcx, f.decl.inputs, arg_tys);
    let lltop = bcx.llbb;
    let block_ty = node_id_type(cx.ccx, f.body.node.id);

    if cx.ccx.sess.get_opts().dps {
        // Call into the new destination-passing-style translation engine.
        let dest = trans_dps::dest_move(cx.ccx.tcx, fcx.llretptr, block_ty);
        bcx = trans_dps::trans_block(bcx, dest, f.body);
    } else {
        // This call to trans_block is the place where we bridge between
        // translation calls that don't have a return value (trans_crate,
        // trans_mod, trans_item, trans_obj, et cetera) and those that do
        // (trans_block, trans_expr, et cetera).
        let rslt =
            if !ty::type_is_nil(cx.ccx.tcx, block_ty) &&
               !ty::type_is_bot(cx.ccx.tcx, block_ty) &&
               f.proto != ast::proto_iter {
                trans_block(bcx, f.body, save_in(fcx.llretptr))
            } else { trans_block(bcx, f.body, return) };
        bcx = rslt.bcx;
    }

    if !is_terminated(bcx) {
        // FIXME: until LLVM has a unit type, we are moving around
        // C_nil values rather than their void type.
        bcx.build.RetVoid();
    }

    // Insert the mandatory first few basic blocks before lltop.
    finish_fn(fcx, lltop);

    ret res;
}

fn trans_fn_inner(cx: @local_ctxt, sp: &span, f: &ast::_fn,
                  llfndecl: ValueRef, ty_self: option::t[ty::t],
                  ty_params: &ast::ty_param[], id: ast::node_id) {
    trans_closure(none, none, cx, sp, f, llfndecl, ty_self, ty_params, id);
}


// trans_fn: creates an LLVM function corresponding to a source language
// function.
fn trans_fn(cx: @local_ctxt, sp: &span, f: &ast::_fn, llfndecl: ValueRef,
            ty_self: option::t[ty::t], ty_params: &ast::ty_param[],
            id: ast::node_id) {
    if !cx.ccx.sess.get_opts().stats {
        trans_fn_inner(cx, sp, f, llfndecl, ty_self, ty_params, id);
        ret;
    }

    let start = time::get_time();
    trans_fn_inner(cx, sp, f, llfndecl, ty_self, ty_params, id);
    let end = time::get_time();
    log_fn_time(cx.ccx, str::connect_ivec(cx.path, "::"), start, end);
}

// process_fwding_mthd: Create the forwarding function that appears in a
// vtable slot for method calls that need to forward to another object.  A
// helper function for create_vtbl.
//
// We use forwarding functions in two situations:
//
//  (1) Forwarding: For method calls that fall through to an inner object, For
//      example, suppose an inner object has method foo and we extend it with
//      a method bar.  The only version of 'foo' we have is on the inner
//      object, but we would like to be able to call outer.foo().  So we use a
//      forwarding function to make the foo method available on the outer
//      object.  It takes all the same arguments as the foo method on the
//      inner object does, calls inner.foo() with those arguments, and then
//      returns the value returned from that call.  (The inner object won't
//      exist until run-time, but we know its type statically.)
//
//  (2) Backwarding: For method calls that dispatch back through an outer
//      object.  For example, suppose an inner object has methods foo and bar,
//      and bar contains the call self.foo().  We extend that object with a
//      foo method that overrides the inner foo.  Now, a call to outer.bar()
//      should send us to to inner.bar() via a normal forwarding function, and
//      then to self.foo().  But inner.bar() was already compiled under the
//      assumption that self.foo() is inner.foo(), when we really want to
//      reach outer.foo().  So, we give 'self' a vtable of backwarding
//      functions, one for each method on inner, each of which takes all the
//      same arguments as the corresponding method on inner does, calls that
//      method on outer, and returns the value returned from that call.

fn process_fwding_mthd(cx: @local_ctxt, sp: &span, m: @ty::method,
                       ty_params: &ast::ty_param[], target_obj_ty: ty::t,
                       backwarding_vtbl: option::t[ValueRef],
                       additional_field_tys: &ty::t[]) -> ValueRef {

    // NB: target_obj_ty is the type of the object being forwarded to.
    // Depending on whether this is a forwarding or backwarding function, it
    // will be either the inner obj's type or the outer obj's type,
    // respectively.

    // Create a local context that's aware of the name of the method we're
    // creating.
    let mcx: @local_ctxt = @{path: cx.path + ~["method", m.ident] with *cx};

    // Make up a name for the forwarding function.
    let fn_name: str = "";
    alt (backwarding_vtbl) {
      // NB: If we have a backwarding_vtbl, that *doesn't* mean that we're
      // currently processing a backwarding fn.  It's the opposite: it means
      // that we have already processed them, and now we're creating
      // forwarding fns that *use* a vtable full of them.
      none. { fn_name = "backwarding_fn"; }
      some(_) { fn_name = "forwarding_fn"; }
    }

    let s: str = mangle_internal_name_by_path_and_seq(mcx.ccx, mcx.path,
                                                      fn_name);

    // Get the forwarding function's type and declare it.
    let llforwarding_fn_ty: TypeRef =
        type_of_fn_full(cx.ccx, sp, m.proto, true, m.inputs, m.output,
                        std::ivec::len[ast::ty_param](ty_params));
    let llforwarding_fn: ValueRef =
        decl_internal_fastcall_fn(cx.ccx.llmod, s, llforwarding_fn_ty);

    // Create a new function context and block context for the forwarding
    // function, holding onto a pointer to the first block.
    let fcx = new_fn_ctxt(cx, sp, llforwarding_fn);
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;

    // The outer object will arrive in the forwarding function via the llenv
    // argument.  Put it in an alloca so that we can GEP into it later.
    let llself_obj_ptr = alloca(bcx, fcx.lcx.ccx.rust_object_type);
    bcx.build.Store(fcx.llenv, llself_obj_ptr);

    // Do backwarding if necessary.
    alt (backwarding_vtbl) {
      none. {
        // NB: As before, this means that we are processing a backwarding fn
        // right now.
      }
      some(bv) {
        // NB: As before, this means that we are processing a forwarding fn
        // right now.

        // Grab the vtable out of the self-object and replace it with the
        // backwarding vtable.
        let llself_obj_vtbl =
            bcx.build.GEP(llself_obj_ptr, ~[C_int(0),
                                            C_int(abi::obj_field_vtbl)]);
        let llbv = bcx.build.PointerCast(bv, T_ptr(T_empty_struct()));
        bcx.build.Store(llbv, llself_obj_vtbl);

        // NB: llself_obj is now a freakish combination of outer object body
        // and backwarding (inner-object) vtable.
      }
    }

    // Grab hold of the outer object so we can pass it into the inner object,
    // in case that inner object needs to make any self-calls.  (Such calls
    // will need to dispatch back through the outer object.)
    let llself_obj = bcx.build.Load(llself_obj_ptr);

    // The 'llretptr' that will arrive in the forwarding function we're
    // creating also needs to be the correct type.  Cast it to the method's
    // return type, if necessary.
    let llretptr = fcx.llretptr;
    if ty::type_contains_params(cx.ccx.tcx, m.output) {
        let llretty = type_of_inner(cx.ccx, sp, m.output);
        llretptr = bcx.build.PointerCast(llretptr, T_ptr(llretty));
    }

    // Now, we have to get the the inner_obj's vtbl out of the self_obj.  This
    // is a multi-step process:

    // First, grab the box out of the self_obj.  It contains a refcount and a
    // body.
    let llself_obj_box =
        bcx.build.GEP(llself_obj_ptr, ~[C_int(0), C_int(abi::obj_field_box)]);
    llself_obj_box = bcx.build.Load(llself_obj_box);

    let ccx = bcx_ccx(bcx);
    let llbox_ty = T_opaque_obj_ptr(*ccx);
    llself_obj_box = bcx.build.PointerCast(llself_obj_box, llbox_ty);

    // Now, reach into the box and grab the body.
    let llself_obj_body =
        bcx.build.GEP(llself_obj_box,
                      ~[C_int(0), C_int(abi::box_rc_field_body)]);

    // Now, we need to figure out exactly what type the body is supposed to be
    // cast to.

    // NB: This next part is almost flat-out copypasta from trans_anon_obj.
    // It would be great to factor this out.

    // Synthesize a tuple type for fields: [field, ...]
    let fields_ty: ty::t = ty::mk_imm_tup(cx.ccx.tcx, additional_field_tys);

    // Type for tydescs.
    let tydesc_ty: ty::t = ty::mk_type(cx.ccx.tcx);

    // Placeholder for non-existent typarams, since anon objs don't have them.
    let typarams_ty: ty::t = ty::mk_imm_tup(cx.ccx.tcx, ~[]);

    // Tuple type for body:
    // [tydesc, [typaram, ...], [field, ...], inner_obj]

    // NB: When we're creating a forwarding fn, target_obj_ty is indeed the
    // type of the inner object, so it makes sense to have 'target_obj_ty'
    // appear here.  When we're creating a backwarding fn, though,
    // target_obj_ty is the outer object's type, so instead, we need to use
    // the extra inner type we passed along.

    let inner_obj_ty = target_obj_ty;

    let body_ty: ty::t =
        ty::mk_imm_tup(cx.ccx.tcx,
                       ~[tydesc_ty, typarams_ty, fields_ty, inner_obj_ty]);

    // And cast to that type.
    llself_obj_body =
        bcx.build.PointerCast(llself_obj_body,
                              T_ptr(type_of(cx.ccx, sp, body_ty)));

    // Now, reach into the body and grab the inner_obj.
    let llinner_obj =
        GEP_tup_like(bcx, body_ty, llself_obj_body,
                     ~[0, abi::obj_body_elt_inner_obj]);
    bcx = llinner_obj.bcx;

    // And, now, somewhere in inner_obj is a vtable with an entry for the
    // method we want.  First, pick out the vtable, and then pluck that
    // method's entry out of the vtable so that the forwarding function can
    // call it.
    let llinner_obj_vtbl =
        bcx.build.GEP(llinner_obj.val,
                      ~[C_int(0), C_int(abi::obj_field_vtbl)]);
    llinner_obj_vtbl = bcx.build.Load(llinner_obj_vtbl);

    // Get the index of the method we want.
    let ix: uint = 0u;
    alt ty::struct(bcx_tcx(bcx), target_obj_ty) {
      ty::ty_obj(methods) {
        ix = ty::method_idx(cx.ccx.sess, sp, m.ident, methods);
      }
      _ {
        // Shouldn't happen.
        cx.ccx.sess.bug("process_fwding_mthd(): non-object type passed \
                        as target_obj_ty");
      }
    }

    // Pick out the original method from the vtable.
    let vtbl_type = T_ptr(T_array(T_ptr(T_nil()), ix + 1u));
    llinner_obj_vtbl = bcx.build.PointerCast(llinner_obj_vtbl, vtbl_type);

    let llorig_mthd =
        bcx.build.GEP(llinner_obj_vtbl, ~[C_int(0), C_int(ix as int)]);

    // Set up the original method to be called.
    let orig_mthd_ty = ty::method_ty_to_fn_ty(cx.ccx.tcx, *m);
    let llorig_mthd_ty =
        type_of_fn_full(bcx_ccx(bcx), sp,
                        ty::ty_fn_proto(bcx_tcx(bcx), orig_mthd_ty), true,
                        m.inputs, m.output,
                        std::ivec::len[ast::ty_param](ty_params));
    llorig_mthd =
        bcx.build.PointerCast(llorig_mthd, T_ptr(T_ptr(llorig_mthd_ty)));
    llorig_mthd = bcx.build.Load(llorig_mthd);

    // Set up the three implicit arguments to the original method we'll need
    // to call.
    let self_arg = llself_obj;
    let llorig_mthd_args: ValueRef[] = ~[llretptr, fcx.lltaskptr, self_arg];

    // Copy the explicit arguments that are being passed into the forwarding
    // function (they're in fcx.llargs) to llorig_mthd_args.

    let a: uint = 3u; // retptr, task ptr, env come first
    let passed_arg: ValueRef = llvm::LLVMGetParam(llforwarding_fn, a);
    for arg: ty::arg  in m.inputs {
        if arg.mode == ty::mo_val {
            passed_arg = load_if_immediate(bcx, passed_arg, arg.ty);
        }
        llorig_mthd_args += ~[passed_arg];
        a += 1u;
    }

    // And, finally, call the original method.
    bcx.build.FastCall(llorig_mthd, llorig_mthd_args);

    bcx.build.RetVoid();
    finish_fn(fcx, lltop);

    ret llforwarding_fn;
}

// process_normal_mthd: Create the contents of a normal vtable slot.  A helper
// function for create_vtbl.
fn process_normal_mthd(cx: @local_ctxt, m: @ast::method, self_ty: ty::t,
                       ty_params: &ast::ty_param[]) -> ValueRef {

    let llfnty = T_nil();
    alt ty::struct(cx.ccx.tcx, node_id_type(cx.ccx, m.node.id)) {
      ty::ty_fn(proto, inputs, output, _, _) {
        llfnty =
            type_of_fn_full(cx.ccx, m.span, proto, true, inputs, output,
                            std::ivec::len[ast::ty_param](ty_params));
      }
    }
    let mcx: @local_ctxt =
        @{path: cx.path + ~["method", m.node.ident] with *cx};
    let s: str = mangle_internal_name_by_path(mcx.ccx, mcx.path);
    let llfn: ValueRef = decl_internal_fastcall_fn(cx.ccx.llmod, s, llfnty);

    // Every method on an object gets its node_id inserted into the
    // crate-wide item_ids map, together with the ValueRef that points to
    // where that method's definition will be in the executable.
    cx.ccx.item_ids.insert(m.node.id, llfn);
    cx.ccx.item_symbols.insert(m.node.id, s);
    trans_fn(mcx, m.span, m.node.meth, llfn, some(self_ty), ty_params,
             m.node.id);

    ret llfn;
}

// Used only inside create_vtbl and create_backwarding_vtbl to distinguish
// different kinds of slots we'll have to create.
tag vtbl_mthd {
    // Normal methods are complete AST nodes, but for forwarding methods,
    // the only information we'll have about them is their type.
    normal_mthd(@ast::method);
    fwding_mthd(@ty::method);
}

// Create a vtable for an object being translated.  Returns a pointer into
// read-only memory.
fn create_vtbl(cx: @local_ctxt, sp: &span, outer_obj_ty: ty::t,
               ob: &ast::_obj, ty_params: &ast::ty_param[],
               inner_obj_ty: option::t[ty::t],
               additional_field_tys: &ty::t[]) -> ValueRef {

    let llmethods: ValueRef[] = ~[];
    let meths: vtbl_mthd[] = ~[];
    let backwarding_vtbl: option::t[ValueRef] = none;

    alt inner_obj_ty {
      none. {

        // If there's no inner_obj, then we don't need any forwarding
        // slots.  Just use the object's regular methods.
        for m: @ast::method  in ob.methods { meths += ~[normal_mthd(m)]; }
      }
      some(inner_obj_ty) {
        // Handle forwarding slots.

        // If this vtable is being created for an extended object, then the
        // vtable needs to contain 'forwarding slots' for methods that were on
        // the original object and are not being overloaded by the extended
        // one.  So, to find the set of methods that we need forwarding slots
        // for, we need to take the set difference of inner_obj_methods
        // (methods on the original object) and ob.methods (methods on the
        // object being added).

        // If we're here, then inner_obj_ty and llinner_obj_ty are the type of
        // the inner object, and "ob" is the wrapper object.  We need to take
        // apart inner_obj_ty (it had better have an object type with
        // methods!) and put those original methods onto the list of methods
        // we need forwarding methods for.

        // Gather up methods on the original object in 'meths'.
        alt ty::struct(cx.ccx.tcx, inner_obj_ty) {
          ty::ty_obj(inner_obj_methods) {
            for m: ty::method  in inner_obj_methods {
                meths += ~[fwding_mthd(@m)];
            }
          }
          _ {
            // Shouldn't happen.
            cx.ccx.sess.bug("create_vtbl(): trying to extend a \
                            non-object");
          }
        }

        // Now, filter out any methods that we don't need forwarding slots
        // for, because they're being replaced.
        fn filtering_fn(cx: @local_ctxt, m: &vtbl_mthd,
                        addtl_meths: (@ast::method)[]) ->
           option::t[vtbl_mthd] {

            alt m {
              fwding_mthd(fm) {
                // Since fm is a fwding_mthd, and we're checking to see if
                // it's in addtl_meths (which only contains normal_mthds), we
                // can't just check if fm is a member of addtl_meths.
                // Instead, we have to go through addtl_meths and see if
                // there's some method in it that has the same name as fm.
                for am: @ast::method  in addtl_meths {
                    if str::eq(am.node.ident, fm.ident) { ret none; }
                }
                ret some(fwding_mthd(fm));
              }
              normal_mthd(_) {
                // Should never happen.
                cx.ccx.sess.bug("create_vtbl(): shouldn't be any \
                                normal_mthds in meths here");
              }
            }
        }
        let f = bind filtering_fn(cx, _, ob.methods);
        meths = std::ivec::filter_map[vtbl_mthd, vtbl_mthd](f, meths);


        // And now add the additional ones (both replacements and entirely new
        // ones).  These'll just be normal methods.
        for m: @ast::method  in ob.methods { meths += ~[normal_mthd(m)]; }
      }
    }

    // Sort all the methods.
    fn vtbl_mthd_lteq(a: &vtbl_mthd, b: &vtbl_mthd) -> bool {
        alt a {
          normal_mthd(ma) {
            alt b {
              normal_mthd(mb) { ret str::lteq(ma.node.ident, mb.node.ident); }
              fwding_mthd(mb) { ret str::lteq(ma.node.ident, mb.ident); }
            }
          }
          fwding_mthd(ma) {
            alt b {
              normal_mthd(mb) { ret str::lteq(ma.ident, mb.node.ident); }
              fwding_mthd(mb) { ret str::lteq(ma.ident, mb.ident); }
            }
          }
        }
    }
    meths =
        std::sort::ivector::merge_sort[vtbl_mthd](bind vtbl_mthd_lteq(_, _),
                                                  meths);

    // Now that we have our list of methods, we can process them in order.
    for m: vtbl_mthd in meths {
        alt m {
          normal_mthd(nm) {
            llmethods += ~[process_normal_mthd(cx, nm, outer_obj_ty,
                                               ty_params)];
          }

          // If we have to process a forwarding method, then we need to know
          // about the inner_obj's type as well as the outer object's type.
          fwding_mthd(fm) {
            alt inner_obj_ty {
              none. {
                // This shouldn't happen; if we're trying to process a
                // forwarding method, then we should always have a
                // inner_obj_ty.
                cx.ccx.sess.bug("create_vtbl(): trying to create \
                                forwarding method without a type \
                                of object to forward to");
              }
              some(t) {
                llmethods +=
                    ~[process_fwding_mthd(cx, sp, fm, ty_params, t,
                                          backwarding_vtbl,
                                          additional_field_tys)];
              }
            }
          }
        }
    }

    let vtbl = C_struct(llmethods);
    let vtbl_name = mangle_internal_name_by_path(cx.ccx, cx.path + ~["vtbl"]);
    let gvar =
        llvm::LLVMAddGlobal(cx.ccx.llmod, val_ty(vtbl), str::buf(vtbl_name));
    llvm::LLVMSetInitializer(gvar, vtbl);
    llvm::LLVMSetGlobalConstant(gvar, True);
    llvm::LLVMSetLinkage(gvar,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    ret gvar;
}

fn create_backwarding_vtbl(cx: @local_ctxt, sp: &span, inner_obj_ty: ty::t,
                           outer_obj_ty: ty::t) -> ValueRef {

    // This vtbl needs to have slots for all of the methods on an inner
    // object, and it needs to forward them to the corresponding slots on the
    // outer object.  All we know about either one are their types.

    let llmethods: ValueRef[] = ~[];
    let meths: ty::method[]= ~[];

    // Gather up methods on the inner object.
    alt ty::struct(cx.ccx.tcx, inner_obj_ty) {
        ty::ty_obj(inner_obj_methods) {
            for m: ty::method in inner_obj_methods {
                meths += ~[m];
            }
        }
        _ {
            // Shouldn't happen.
            cx.ccx.sess.bug("create_backwarding_vtbl(): trying to extend a \
                            non-object");
        }
    }

    // Methods should have already been sorted, so no need to do so again.
    for m: ty::method in meths {
        // We pass outer_obj_ty to process_fwding_mthd() because it's
        // the one being forwarded to.
        llmethods += ~[process_fwding_mthd(
            cx, sp, @m, ~[], outer_obj_ty, none, ~[])];
    }

    let vtbl = C_struct(llmethods);
    let vtbl_name =
        mangle_internal_name_by_path(cx.ccx,
                                     cx.path + ~["backwarding_vtbl"]);
    let gvar =
        llvm::LLVMAddGlobal(cx.ccx.llmod, val_ty(vtbl), str::buf(vtbl_name));
    llvm::LLVMSetInitializer(gvar, vtbl);
    llvm::LLVMSetGlobalConstant(gvar, True);
    llvm::LLVMSetLinkage(gvar,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);

    ret gvar;

}

// trans_obj: creates an LLVM function that is the object constructor for the
// object being translated.
fn trans_obj(cx: @local_ctxt, sp: &span, ob: &ast::_obj,
             ctor_id: ast::node_id, ty_params: &ast::ty_param[]) {
    // To make a function, we have to create a function context and, inside
    // that, a number of block contexts for which code is generated.

    let ccx = cx.ccx;
    let llctor_decl;
    alt ccx.item_ids.find(ctor_id) {
      some(x) { llctor_decl = x; }
      _ { cx.ccx.sess.span_fatal(sp, "unbound llctor_decl in trans_obj"); }
    }
    // Much like trans_fn, we must create an LLVM function, but since we're
    // starting with an ast::_obj rather than an ast::_fn, we have some setup
    // work to do.

    // The fields of our object will become the arguments to the function
    // we're creating.

    let fn_args: ast::arg[] = ~[];
    for f: ast::obj_field  in ob.fields {
        fn_args +=
            ~[{mode: ast::alias(false), ty: f.ty, ident: f.ident, id: f.id}];
    }
    let fcx = new_fn_ctxt(cx, sp, llctor_decl);

    // Both regular arguments and type parameters are handled here.
    create_llargs_for_fn_args(fcx, ast::proto_fn, none[ty::t],
                              ty::ret_ty_of_fn(ccx.tcx, ctor_id), fn_args,
                              ty_params);
    let arg_tys: ty::arg[] = arg_tys_of_fn(ccx, ctor_id);
    copy_args_to_allocas(fcx, fn_args, arg_tys);

    //  Create the first block context in the function and keep a handle on it
    //  to pass to finish_fn later.
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;

    // Pick up the type of this object by looking at our own output type, that
    // is, the output type of the object constructor we're building.
    let self_ty = ty::ret_ty_of_fn(ccx.tcx, ctor_id);

    // Set up the two-word pair that we're going to return from the object
    // constructor we're building.  The two elements of this pair will be a
    // vtable pointer and a body pointer.  (llretptr already points to the
    // place where this two-word pair should go; it was pre-allocated by the
    // caller of the function.)
    let pair = bcx.fcx.llretptr;

    // Grab onto the first and second elements of the pair.
    // abi::obj_field_vtbl and abi::obj_field_box simply specify words 0 and 1
    // of 'pair'.
    let pair_vtbl =
        bcx.build.GEP(pair, ~[C_int(0), C_int(abi::obj_field_vtbl)]);
    let pair_box =
        bcx.build.GEP(pair, ~[C_int(0), C_int(abi::obj_field_box)]);

    // Make a vtable for this object: a static array of pointers to functions.
    // It will be located in the read-only memory of the executable we're
    // creating and will contain ValueRefs for all of this object's methods.
    // create_vtbl returns a pointer to the vtable, which we store.
    let vtbl = create_vtbl(cx, sp, self_ty, ob, ty_params, none, ~[]);
    vtbl = bcx.build.PointerCast(vtbl, T_ptr(T_empty_struct()));

    bcx.build.Store(vtbl, pair_vtbl);

    // Next we have to take care of the other half of the pair we're
    // returning: a boxed (reference-counted) tuple containing a tydesc,
    // typarams, and fields.

    // FIXME: What about inner_obj?  Do we have to think about it here?
    // (Pertains to issues #538/#539/#540/#543.)

    let llbox_ty: TypeRef = T_ptr(T_empty_struct());

    if std::ivec::len[ast::ty_param](ty_params) == 0u &&
           std::ivec::len[ty::arg](arg_tys) == 0u {
        // If the object we're translating has no fields or type parameters,
        // there's not much to do.

        // Store null into pair, if no args or typarams.
        bcx.build.Store(C_null(llbox_ty), pair_box);
    } else {
        // Otherwise, we have to synthesize a big structural type for the
        // object body.
        let obj_fields: ty::t[] = ~[];
        for a: ty::arg  in arg_tys { obj_fields += ~[a.ty]; }

        // Tuple type for fields: [field, ...]
        let fields_ty: ty::t = ty::mk_imm_tup(ccx.tcx, obj_fields);

        let tydesc_ty = ty::mk_type(ccx.tcx);
        let tps: ty::t[] = ~[];
        for tp: ast::ty_param  in ty_params { tps += ~[tydesc_ty]; }

        // Tuple type for typarams: [typaram, ...]
        let typarams_ty: ty::t = ty::mk_imm_tup(ccx.tcx, tps);

        // Tuple type for body: [tydesc_ty, [typaram, ...], [field, ...]]
        let body_ty: ty::t =
            ty::mk_imm_tup(ccx.tcx, ~[tydesc_ty, typarams_ty, fields_ty]);

        // Hand this type we've synthesized off to trans_malloc_boxed, which
        // allocates a box, including space for a refcount.
        let box = trans_malloc_boxed(bcx, body_ty);
        bcx = box.bcx;
        let body = box.body;

        // Put together a tydesc for the body, so that the object can later be
        // freed by calling through its tydesc.

        // Every object (not just those with type parameters) needs to have a
        // tydesc to describe its body, since all objects have unknown type to
        // the user of the object.  So the tydesc is needed to keep track of
        // the types of the object's fields, so that the fields can be freed
        // later.

        let body_tydesc =
            GEP_tup_like(bcx, body_ty, body,
                         ~[0, abi::obj_body_elt_tydesc]);
        bcx = body_tydesc.bcx;
        let ti = none[@tydesc_info];
        let body_td = get_tydesc(bcx, body_ty, true, ti);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_drop_glue, ti);
        lazily_emit_tydesc_glue(bcx, abi::tydesc_field_free_glue, ti);
        bcx = body_td.bcx;
        bcx.build.Store(body_td.val, body_tydesc.val);

        // Copy the object's type parameters and fields into the space we
        // allocated for the object body.  (This is something like saving the
        // lexical environment of a function in its closure: the "captured
        // typarams" are any type parameters that are passed to the object
        // constructor and are then available to the object's methods.
        // Likewise for the object's fields.)

        // Copy typarams into captured typarams.
        let body_typarams =
            GEP_tup_like(bcx, body_ty, body,
                         ~[0, abi::obj_body_elt_typarams]);
        bcx = body_typarams.bcx;
        let i: int = 0;
        for tp: ast::ty_param  in ty_params {
            let typaram = bcx.fcx.lltydescs.(i);
            let capture =
                GEP_tup_like(bcx, typarams_ty, body_typarams.val, ~[0, i]);
            bcx = capture.bcx;
            bcx = copy_val(bcx, INIT, capture.val, typaram, tydesc_ty).bcx;
            i += 1;
        }

        // Copy args into body fields.
        let body_fields =
            GEP_tup_like(bcx, body_ty, body,
                         ~[0, abi::obj_body_elt_fields]);
        bcx = body_fields.bcx;
        i = 0;
        for f: ast::obj_field  in ob.fields {
            alt bcx.fcx.llargs.find(f.id) {
              some(arg1) {
                let arg = load_if_immediate(bcx, arg1, arg_tys.(i).ty);
                let field =
                    GEP_tup_like(bcx, fields_ty, body_fields.val, ~[0, i]);
                bcx = field.bcx;
                bcx = copy_val(bcx, INIT, field.val, arg, arg_tys.(i).ty).bcx;
                i += 1;
              }
              none. {
                bcx_ccx(bcx).sess.span_fatal(f.ty.span,
                                             "internal error in trans_obj");
              }
            }
        }

        // Store box ptr in outer pair.
        let p = bcx.build.PointerCast(box.box, llbox_ty);
        bcx.build.Store(p, pair_box);
    }
    bcx.build.RetVoid();

    // Insert the mandatory first few basic blocks before lltop.
    finish_fn(fcx, lltop);
}

fn trans_res_ctor(cx: @local_ctxt, sp: &span, dtor: &ast::_fn,
                  ctor_id: ast::node_id, ty_params: &ast::ty_param[]) {
    // Create a function for the constructor
    let llctor_decl;
    alt cx.ccx.item_ids.find(ctor_id) {
      some(x) { llctor_decl = x; }
      _ { cx.ccx.sess.span_fatal(sp, "unbound ctor_id in trans_res_ctor"); }
    }
    let fcx = new_fn_ctxt(cx, sp, llctor_decl);
    let ret_t = ty::ret_ty_of_fn(cx.ccx.tcx, ctor_id);
    create_llargs_for_fn_args(fcx, ast::proto_fn, none[ty::t], ret_t,
                              dtor.decl.inputs, ty_params);
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;
    let arg_t = arg_tys_of_fn(cx.ccx, ctor_id).(0).ty;
    let tup_t = ty::mk_imm_tup(cx.ccx.tcx, ~[ty::mk_int(cx.ccx.tcx), arg_t]);
    let arg;
    alt fcx.llargs.find(dtor.decl.inputs.(0).id) {
      some(x) { arg = load_if_immediate(bcx, x, arg_t); }
      _ { cx.ccx.sess.span_fatal(sp, "unbound dtor decl in trans_res_ctor"); }
    }

    let llretptr = fcx.llretptr;
    if ty::type_has_dynamic_size(cx.ccx.tcx, ret_t) {
        let llret_t = T_ptr(T_struct(~[T_i32(), llvm::LLVMTypeOf(arg)]));
        llretptr = bcx.build.BitCast(llretptr, llret_t);
    }

    let dst = GEP_tup_like(bcx, tup_t, llretptr, ~[0, 1]);
    bcx = dst.bcx;
    bcx = copy_val(bcx, INIT, dst.val, arg, arg_t).bcx;
    let flag = GEP_tup_like(bcx, tup_t, llretptr, ~[0, 0]);
    bcx = flag.bcx;
    bcx.build.Store(C_int(1), flag.val);
    bcx.build.RetVoid();
    finish_fn(fcx, lltop);
}


fn trans_tag_variant(cx: @local_ctxt, tag_id: ast::node_id,
                     variant: &ast::variant, index: int, is_degen: bool,
                     ty_params: &ast::ty_param[]) {
    if std::ivec::len[ast::variant_arg](variant.node.args) == 0u {
        ret; // nullary constructors are just constants

    }
    // Translate variant arguments to function arguments.

    let fn_args: ast::arg[] = ~[];
    let i = 0u;
    for varg: ast::variant_arg  in variant.node.args {
        fn_args +=
            ~[{mode: ast::alias(false),
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
    create_llargs_for_fn_args(fcx, ast::proto_fn, none[ty::t],
                              ty::ret_ty_of_fn(cx.ccx.tcx, variant.node.id),
                              fn_args, ty_params);
    let ty_param_substs: ty::t[] = ~[];
    i = 0u;
    for tp: ast::ty_param  in ty_params {
        ty_param_substs += ~[ty::mk_param(cx.ccx.tcx, i, tp.kind)];
        i += 1u;
    }
    let arg_tys = arg_tys_of_fn(cx.ccx, variant.node.id);
    copy_args_to_allocas(fcx, fn_args, arg_tys);
    let bcx = new_top_block_ctxt(fcx);
    let lltop = bcx.llbb;

    // Cast the tag to a type we can GEP into.
    let llblobptr =
        if is_degen {
            fcx.llretptr
        } else {
            let lltagptr =
                bcx.build.PointerCast(fcx.llretptr,
                                      T_opaque_tag_ptr(fcx.lcx.ccx.tn));
            let lldiscrimptr = bcx.build.GEP(lltagptr, ~[C_int(0), C_int(0)]);
            bcx.build.Store(C_int(index), lldiscrimptr);
            bcx.build.GEP(lltagptr, ~[C_int(0), C_int(1)])
        };
    i = 0u;
    for va: ast::variant_arg  in variant.node.args {
        let rslt =
            GEP_tag(bcx, llblobptr, ast::local_def(tag_id),
                    ast::local_def(variant.node.id), ty_param_substs,
                    i as int);
        bcx = rslt.bcx;
        let lldestptr = rslt.val;
        // If this argument to this function is a tag, it'll have come in to
        // this function as an opaque blob due to the way that type_of()
        // works. So we have to cast to the destination's view of the type.

        let llargptr;
        alt fcx.llargs.find(va.id) {
          some(x) { llargptr = bcx.build.PointerCast(x, val_ty(lldestptr)); }
          none. {
            bcx_ccx(bcx).sess.bug("unbound argptr in \
                                      trans_tag_variant");
          }
        }
        let arg_ty = arg_tys.(i).ty;
        let llargval;
        if ty::type_is_structural(cx.ccx.tcx, arg_ty) ||
               ty::type_has_dynamic_size(cx.ccx.tcx, arg_ty) {
            llargval = llargptr;
        } else { llargval = bcx.build.Load(llargptr); }
        rslt = copy_val(bcx, INIT, lldestptr, llargval, arg_ty);
        bcx = rslt.bcx;
        i += 1u;
    }
    bcx = trans_block_cleanups(bcx, find_scope_cx(bcx));
    bcx.build.RetVoid();
    finish_fn(fcx, lltop);
}


// FIXME: this should do some structural hash-consing to avoid
// duplicate constants. I think. Maybe LLVM has a magical mode
// that does so later on?
fn trans_const_expr(cx: &@crate_ctxt, e: @ast::expr) -> ValueRef {
    alt e.node {
      ast::expr_lit(lit) { ret trans_crate_lit(cx, *lit); }
      _ { cx.sess.span_unimpl(e.span, "consts that's not a plain literal"); }
    }
}

fn trans_const(cx: &@crate_ctxt, e: @ast::expr, id: ast::node_id) {
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

fn trans_item(cx: @local_ctxt, item: &ast::item) {
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
            @{path: cx.path + ~[item.ident],
              module_path: cx.module_path + ~[item.ident] with *cx};
        trans_mod(sub_cx, m);
      }
      ast::item_tag(variants, tps) {
        let sub_cx = extend_path(cx, item.ident);
        let degen = std::ivec::len(variants) == 1u;
        let i = 0;
        for variant: ast::variant  in variants {
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
fn trans_mod(cx: @local_ctxt, m: &ast::_mod) {
    for item: @ast::item  in m.items { trans_item(cx, *item); }
}

fn get_pair_fn_ty(llpairty: TypeRef) -> TypeRef {
    // Bit of a kludge: pick the fn typeref out of the pair.

    ret struct_elt(llpairty, 0u);
}

fn decl_fn_and_pair(ccx: &@crate_ctxt, sp: &span, path: &str[], flav: str,
                    ty_params: &ast::ty_param[], node_id: ast::node_id) {
    decl_fn_and_pair_full(ccx, sp, path, flav, ty_params, node_id,
                          node_id_type(ccx, node_id));
}

fn decl_fn_and_pair_full(ccx: &@crate_ctxt, sp: &span, path: &str[],
                         flav: str, ty_params: &ast::ty_param[],
                         node_id: ast::node_id, node_type: ty::t) {
    let llfty = type_of_fn_from_ty(ccx, sp, node_type,
                                   std::ivec::len(ty_params));
    alt ty::struct(ccx.tcx, node_type) {
      ty::ty_fn(proto, inputs, output, _, _) {
        llfty =
            type_of_fn(ccx, sp, proto, inputs, output,
                       std::ivec::len[ast::ty_param](ty_params));
      }
      _ { ccx.sess.bug("decl_fn_and_pair(): fn item doesn't have fn type!"); }
    }
    let is_main: bool = is_main_name(path) && !ccx.sess.get_opts().library;
    // Declare the function itself.

    let s: str =
        if is_main {
            "_rust_main"
        } else { mangle_internal_name_by_path(ccx, path) };
    let llfn: ValueRef = decl_internal_fastcall_fn(ccx.llmod, s, llfty);
    // Declare the global constant pair that points to it.

    let ps: str = mangle_exported_name(ccx, path, node_type);
    register_fn_pair(ccx, ps, llfty, llfn, node_id);
    if is_main {
        if ccx.main_fn != none[ValueRef] {
            ccx.sess.span_fatal(sp, "multiple 'main' functions");
        }
        llvm::LLVMSetLinkage(llfn,
                             lib::llvm::LLVMExternalLinkage as llvm::Linkage);
        ccx.main_fn = some(llfn);
    }
}

// Create a closure: a pair containing (1) a ValueRef, pointing to where the
// fn's definition is in the executable we're creating, and (2) a pointer to
// space for the function's environment.
fn create_fn_pair(cx: &@crate_ctxt, ps: str, llfnty: TypeRef, llfn: ValueRef,
                  external: bool) -> ValueRef {
    let gvar =
        llvm::LLVMAddGlobal(cx.llmod, T_fn_pair(*cx, llfnty), str::buf(ps));
    let pair = C_struct(~[llfn, C_null(T_opaque_closure_ptr(*cx))]);
    llvm::LLVMSetInitializer(gvar, pair);
    llvm::LLVMSetGlobalConstant(gvar, True);
    if !external {
        llvm::LLVMSetLinkage(gvar,
                             lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    }
    ret gvar;
}

// Create a /real/ closure: this is like create_fn_pair, but creates a
// a fn value on the stack with a specified environment (which need not be
// on the stack).
fn create_real_fn_pair(cx: &@block_ctxt, llfnty: TypeRef, llfn: ValueRef,
                       llenvptr: ValueRef) -> ValueRef {
    let lcx = cx.fcx.lcx;

    let pair = alloca(cx, T_fn_pair(*lcx.ccx, llfnty));
    let code_cell =
        cx.build.GEP(pair, ~[C_int(0), C_int(abi::fn_field_code)]);
    cx.build.Store(llfn, code_cell);
    let env_cell = cx.build.GEP(pair, ~[C_int(0), C_int(abi::fn_field_box)]);
    let llenvblobptr =
        cx.build.PointerCast(llenvptr, T_opaque_closure_ptr(*lcx.ccx));
    cx.build.Store(llenvblobptr, env_cell);
    ret pair;
}

fn register_fn_pair(cx: &@crate_ctxt, ps: str, llfnty: TypeRef,
                    llfn: ValueRef, id: ast::node_id) {
    // FIXME: We should also hide the unexported pairs in crates.

    let gvar =
        create_fn_pair(cx, ps, llfnty, llfn, cx.sess.get_opts().library);
    cx.item_ids.insert(id, llfn);
    cx.item_symbols.insert(id, ps);
    cx.fn_pairs.insert(id, gvar);
}


// Returns the number of type parameters that the given native function has.
fn native_fn_ty_param_count(cx: &@crate_ctxt, id: ast::node_id) -> uint {
    let count;
    let native_item =
        alt cx.ast_map.find(id) { some(ast_map::node_native_item(i)) { i } };
    alt native_item.node {
      ast::native_item_ty. {
        cx.sess.bug("decl_native_fn_and_pair(): native fn isn't \
                        actually a fn");
      }
      ast::native_item_fn(_, _, tps) {
        count = std::ivec::len[ast::ty_param](tps);
      }
    }
    ret count;
}

fn native_fn_wrapper_type(cx: &@crate_ctxt, sp: &span, ty_param_count: uint,
                          x: ty::t) -> TypeRef {
    alt ty::struct(cx.tcx, x) {
      ty::ty_native_fn(abi, args, out) {
        ret type_of_fn(cx, sp, ast::proto_fn, args, out, ty_param_count);
      }
    }
}

fn decl_native_fn_and_pair(ccx: &@crate_ctxt, sp: &span, path: &str[],
                           name: str, id: ast::node_id) {
    let num_ty_param = native_fn_ty_param_count(ccx, id);
    // Declare the wrapper.

    let t = node_id_type(ccx, id);
    let wrapper_type = native_fn_wrapper_type(ccx, sp, num_ty_param, t);
    let s: str = mangle_internal_name_by_path(ccx, path);
    let wrapper_fn: ValueRef =
        decl_internal_fastcall_fn(ccx.llmod, s, wrapper_type);
    // Declare the global constant pair that points to it.

    let ps: str = mangle_exported_name(ccx, path, node_id_type(ccx, id));
    register_fn_pair(ccx, ps, wrapper_type, wrapper_fn, id);
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

    let call_args: ValueRef[] = ~[];
    if pass_task { call_args += ~[lltaskptr]; }
    if uses_retptr { call_args += ~[bcx.fcx.llretptr]; }

    let arg_n = 3u;
    for each i: uint  in uint::range(0u, num_ty_param) {
        let llarg = llvm::LLVMGetParam(fcx.llfn, arg_n);
        fcx.lltydescs += ~[llarg];
        assert (llarg as int != 0);
        if cast_to_i32 {
            call_args += ~[vp2i(bcx, llarg)];
        } else { call_args += ~[llarg]; }
        arg_n += 1u;
    }
    fn convert_arg_to_i32(cx: &@block_ctxt, v: ValueRef, t: ty::t,
                          mode: ty::mode) -> ValueRef {
        if mode == ty::mo_val {
            if ty::type_is_integral(bcx_tcx(cx), t) {
                let lldsttype = T_int();
                let llsrctype = type_of(bcx_ccx(cx), cx.sp, t);
                if llvm::LLVMGetIntTypeWidth(lldsttype) >
                       llvm::LLVMGetIntTypeWidth(llsrctype) {
                    ret cx.build.ZExtOrBitCast(v, T_int());
                }
                ret cx.build.TruncOrBitCast(v, T_int());
            }
            if ty::type_is_fp(bcx_tcx(cx), t) {
                ret cx.build.FPToSI(v, T_int());
            }
        }
        ret vp2i(cx, v);
    }

    fn trans_simple_native_abi(bcx: &@block_ctxt, name: str,
                               call_args: &mutable ValueRef[], fn_type: ty::t,
                               first_arg_n: uint, uses_retptr: bool, cc: uint)
       -> {val: ValueRef, rptr: ValueRef} {
        let call_arg_tys: TypeRef[] = ~[];
        for arg: ValueRef  in call_args { call_arg_tys += ~[val_ty(arg)]; }

        let llnativefnty;
        if uses_retptr {
            llnativefnty = T_fn(call_arg_tys, T_void());
        } else {
            llnativefnty =
                T_fn(call_arg_tys,
                     type_of(bcx_ccx(bcx), bcx.sp,
                             ty::ty_fn_ret(bcx_tcx(bcx), fn_type)));
        }

        let llnativefn =
            get_extern_fn(bcx_ccx(bcx).externs, bcx_ccx(bcx).llmod, name, cc,
                          llnativefnty);
        let r =
            if cc == lib::llvm::LLVMCCallConv {
                bcx.build.Call(llnativefn, call_args)
            } else { bcx.build.CallWithConv(llnativefn, call_args, cc) };
        let rptr = bcx.fcx.llretptr;
        ret {val: r, rptr: rptr};
    }

    let args = ty::ty_fn_args(ccx.tcx, fn_type);
    // Build up the list of arguments.

    let drop_args: {val: ValueRef, ty: ty::t}[] = ~[];
    let i = arg_n;
    for arg: ty::arg  in args {
        let llarg = llvm::LLVMGetParam(fcx.llfn, i);
        assert (llarg as int != 0);
        if cast_to_i32 {
            let llarg_i32 = convert_arg_to_i32(bcx, llarg, arg.ty, arg.mode);
            call_args += ~[llarg_i32];
        } else { call_args += ~[llarg]; }
        if arg.mode == ty::mo_val {
            drop_args += ~[{val: llarg, ty: arg.ty}];
        }
        i += 1u;
    }
    let r;
    let rptr;
    alt abi {
      ast::native_abi_llvm. {
        let result =
            trans_simple_native_abi(bcx, name, call_args, fn_type, arg_n,
                                    uses_retptr, lib::llvm::LLVMCCallConv);
        r = result.val;
        rptr = result.rptr;
      }
      ast::native_abi_rust_intrinsic. {
        let external_name = "rust_intrinsic_" + name;
        let result =
            trans_simple_native_abi(bcx, external_name, call_args, fn_type,
                                    arg_n, uses_retptr,
                                    lib::llvm::LLVMCCallConv);
        r = result.val;
        rptr = result.rptr;
      }
      ast::native_abi_x86stdcall. {
        let result =
            trans_simple_native_abi(bcx, name, call_args, fn_type, arg_n,
                                    uses_retptr,
                                    lib::llvm::LLVMX86StdcallCallConv);
        r = result.val;
        rptr = result.rptr;
      }
      _ {
        r =
            trans_native_call(bcx.build, ccx.glues, lltaskptr, ccx.externs,
                              ccx.tn, ccx.llmod, name, pass_task, call_args);
        rptr = bcx.build.BitCast(fcx.llretptr, T_ptr(T_i32()));
      }
    }
    // We don't store the return value if it's nil, to avoid stomping on a nil
    // pointer. This is the only concession made to non-i32 return values. See
    // the FIXME above.

    if !rty_is_nil && !uses_retptr { bcx.build.Store(r, rptr); }

    for d: {val: ValueRef, ty: ty::t}  in drop_args {
        bcx = drop_ty(bcx, d.val, d.ty).bcx;
    }
    bcx.build.RetVoid();
    finish_fn(fcx, lltop);
}

fn item_path(item: &@ast::item) -> str[] { ret ~[item.ident]; }

fn collect_native_item(ccx: @crate_ctxt, i: &@ast::native_item, pt: &str[],
                       v: &vt[str[]]) {
    alt i.node {
      ast::native_item_fn(_, _, _) {
        if !ccx.obj_methods.contains_key(i.id) {
            decl_native_fn_and_pair(ccx, i.span, pt, i.ident, i.id);
        }
      }
      _ { }
    }
}

fn collect_item_1(ccx: @crate_ctxt, i: &@ast::item, pt: &str[],
                  v: &vt[str[]]) {
    visit::visit_item(i, pt + item_path(i), v);
    alt i.node {
      ast::item_const(_, _) {
        let typ = node_id_type(ccx, i.id);
        let s =
            mangle_exported_name(ccx, pt + ~[i.ident],
                                 node_id_type(ccx, i.id));
        let g =
            llvm::LLVMAddGlobal(ccx.llmod, type_of(ccx, i.span, typ),
                                str::buf(s));
        ccx.item_symbols.insert(i.id, s);
        ccx.consts.insert(i.id, g);
      }
      _ { }
    }
}

fn collect_item_2(ccx: &@crate_ctxt, i: &@ast::item, pt: &str[],
                  v: &vt[str[]]) {
    let new_pt = pt + item_path(i);
    visit::visit_item(i, new_pt, v);
    alt i.node {
      ast::item_fn(f, tps) {
        if !ccx.obj_methods.contains_key(i.id) {
            decl_fn_and_pair(ccx, i.span, new_pt, "fn", tps, i.id);
        }
      }
      ast::item_obj(ob, tps, ctor_id) {
        decl_fn_and_pair(ccx, i.span, new_pt, "obj_ctor", tps, ctor_id);
        for m: @ast::method  in ob.methods {
            ccx.obj_methods.insert(m.node.id, ());
        }
      }
      ast::item_res(_, dtor_id, tps, ctor_id) {
        decl_fn_and_pair(ccx, i.span, new_pt, "res_ctor", tps, ctor_id);
        // Note that the destructor is associated with the item's id, not
        // the dtor_id. This is a bit counter-intuitive, but simplifies
        // ty_res, which would have to carry around two def_ids otherwise
        // -- one to identify the type, and one to find the dtor symbol.
        decl_fn_and_pair_full(ccx, i.span, new_pt, "res_dtor", tps, i.id,
                              node_id_type(ccx, dtor_id));
      }
      _ { }
    }
}

fn collect_items(ccx: &@crate_ctxt, crate: @ast::crate) {
    let visitor0 = visit::default_visitor();
    let visitor1 =
        @{visit_native_item: bind collect_native_item(ccx, _, _, _),
          visit_item: bind collect_item_1(ccx, _, _, _) with *visitor0};
    let visitor2 =
        @{visit_item: bind collect_item_2(ccx, _, _, _) with *visitor0};
    visit::visit_crate(*crate, ~[], visit::mk_vt(visitor1));
    visit::visit_crate(*crate, ~[], visit::mk_vt(visitor2));
}

fn collect_tag_ctor(ccx: @crate_ctxt, i: &@ast::item, pt: &str[],
                    v: &vt[str[]]) {
    let new_pt = pt + item_path(i);
    visit::visit_item(i, new_pt, v);
    alt i.node {
      ast::item_tag(variants, tps) {
        for variant: ast::variant  in variants {
            if std::ivec::len(variant.node.args) != 0u {
                decl_fn_and_pair(ccx, i.span, new_pt + ~[variant.node.name],
                                 "tag", tps, variant.node.id);
            }
        }
      }
      _ {/* fall through */ }
    }
}

fn collect_tag_ctors(ccx: &@crate_ctxt, crate: @ast::crate) {
    let visitor =
        @{visit_item: bind collect_tag_ctor(ccx, _, _, _)
             with *visit::default_visitor()};
    visit::visit_crate(*crate, ~[], visit::mk_vt(visitor));
}


// The constant translation pass.
fn trans_constant(ccx: @crate_ctxt, it: &@ast::item, pt: &str[],
                  v: &vt[str[]]) {
    let new_pt = pt + item_path(it);
    visit::visit_item(it, new_pt, v);
    alt it.node {
      ast::item_tag(variants, _) {
        let i = 0u;
        let n_variants = std::ivec::len[ast::variant](variants);
        while i < n_variants {
            let variant = variants.(i);
            let p = new_pt + ~[it.ident, variant.node.name, "discrim"];
            let s = mangle_exported_name(ccx, p, ty::mk_int(ccx.tcx));
            let discrim_gvar =
                llvm::LLVMAddGlobal(ccx.llmod, T_int(), str::buf(s));
            if n_variants != 1u {
                llvm::LLVMSetInitializer(discrim_gvar, C_int(i as int));
                llvm::LLVMSetGlobalConstant(discrim_gvar, True);
            }
            ccx.discrims.insert(variant.node.id, discrim_gvar);
            ccx.discrim_symbols.insert(variant.node.id, s);
            i += 1u;
        }
      }
      _ { }
    }
}

fn trans_constants(ccx: &@crate_ctxt, crate: @ast::crate) {
    let visitor =
        @{visit_item: bind trans_constant(ccx, _, _, _)
             with *visit::default_visitor()};
    visit::visit_crate(*crate, ~[], visit::mk_vt(visitor));
}

fn vp2i(cx: &@block_ctxt, v: ValueRef) -> ValueRef {
    ret cx.build.PtrToInt(v, T_int());
}

fn vi2p(cx: &@block_ctxt, v: ValueRef, t: TypeRef) -> ValueRef {
    ret cx.build.IntToPtr(v, t);
}

fn p2i(v: ValueRef) -> ValueRef { ret llvm::LLVMConstPtrToInt(v, T_int()); }

fn i2p(v: ValueRef, t: TypeRef) -> ValueRef {
    ret llvm::LLVMConstIntToPtr(v, t);
}

fn declare_intrinsics(llmod: ModuleRef) -> hashmap[str, ValueRef] {
    let T_memmove32_args: TypeRef[] =
        ~[T_ptr(T_i8()), T_ptr(T_i8()), T_i32(), T_i32(), T_i1()];
    let T_memmove64_args: TypeRef[] =
        ~[T_ptr(T_i8()), T_ptr(T_i8()), T_i64(), T_i32(), T_i1()];
    let T_memset32_args: TypeRef[] =
        ~[T_ptr(T_i8()), T_i8(), T_i32(), T_i32(), T_i1()];
    let T_memset64_args: TypeRef[] =
        ~[T_ptr(T_i8()), T_i8(), T_i64(), T_i32(), T_i1()];
    let T_trap_args: TypeRef[] = ~[];
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
    let intrinsics = new_str_hash[ValueRef]();
    intrinsics.insert("llvm.memmove.p0i8.p0i8.i32", memmove32);
    intrinsics.insert("llvm.memmove.p0i8.p0i8.i64", memmove64);
    intrinsics.insert("llvm.memset.p0i8.i32", memset32);
    intrinsics.insert("llvm.memset.p0i8.i64", memset64);
    intrinsics.insert("llvm.trap", trap);
    ret intrinsics;
}

fn trace_str(cx: &@block_ctxt, s: str) {
    cx.build.Call(bcx_ccx(cx).upcalls.trace_str,
                  ~[cx.fcx.lltaskptr, C_cstr(bcx_ccx(cx), s)]);
}

fn trace_word(cx: &@block_ctxt, v: ValueRef) {
    cx.build.Call(bcx_ccx(cx).upcalls.trace_word, ~[cx.fcx.lltaskptr, v]);
}

fn trace_ptr(cx: &@block_ctxt, v: ValueRef) {
    trace_word(cx, cx.build.PtrToInt(v, T_int()));
}

fn trap(bcx: &@block_ctxt) {
    let v: ValueRef[] = ~[];
    alt bcx_ccx(bcx).intrinsics.find("llvm.trap") {
      some(x) { bcx.build.Call(x, v); }
      _ { bcx_ccx(bcx).sess.bug("unbound llvm.trap in trap"); }
    }
}

fn decl_no_op_type_glue(llmod: ModuleRef, taskptr_type: TypeRef) -> ValueRef {
    let ty = T_fn(~[taskptr_type, T_ptr(T_i8())], T_void());
    ret decl_fastcall_fn(llmod, abi::no_op_type_glue_name(), ty);
}

fn make_no_op_type_glue(fun: ValueRef) {
    let bb_name = str::buf("_rust_no_op_type_glue_bb");
    let llbb = llvm::LLVMAppendBasicBlock(fun, bb_name);
    new_builder(llbb).RetVoid();
}

fn vec_fill(bcx: &@block_ctxt, v: ValueRef) -> ValueRef {
    ret bcx.build.Load(bcx.build.GEP(v,
                                     ~[C_int(0), C_int(abi::vec_elt_fill)]));
}

fn vec_p0(bcx: &@block_ctxt, v: ValueRef) -> ValueRef {
    let p = bcx.build.GEP(v, ~[C_int(0), C_int(abi::vec_elt_data)]);
    ret bcx.build.PointerCast(p, T_ptr(T_i8()));
}

fn make_glues(llmod: ModuleRef, taskptr_type: TypeRef) -> @glue_fns {
    ret @{no_op_type_glue: decl_no_op_type_glue(llmod, taskptr_type)};
}

fn make_common_glue(sess: &session::session, output: &str) {
    // FIXME: part of this is repetitive and is probably a good idea
    // to autogen it.

    let task_type = T_task();
    let taskptr_type = T_ptr(task_type);

    let llmod =
        llvm::LLVMModuleCreateWithNameInContext(str::buf("rust_out"),
                                                llvm::LLVMGetGlobalContext());
    let dat_layt = x86::get_data_layout(); //HACK (buf lifetime issue)
    llvm::LLVMSetDataLayout(llmod, str::buf(dat_layt));
    let targ_trip = x86::get_target_triple(); //HACK (buf lifetime issue)
    llvm::LLVMSetTarget(llmod, str::buf(targ_trip));
    mk_target_data(x86::get_data_layout());
    declare_intrinsics(llmod);
    let modl_asm = x86::get_module_asm(); //HACK (buf lifetime issue)
    llvm::LLVMSetModuleInlineAsm(llmod, str::buf(modl_asm));
    make_glues(llmod, taskptr_type);
    link::write::run_passes(sess, llmod, output);
}

fn create_module_map(ccx: &@crate_ctxt) -> ValueRef {
    let elttype = T_struct(~[T_int(), T_int()]);
    let maptype = T_array(elttype, ccx.module_data.size() + 1u);
    let map =
        llvm::LLVMAddGlobal(ccx.llmod, maptype, str::buf("_rust_mod_map"));
    llvm::LLVMSetLinkage(map,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    let elts: ValueRef[] = ~[];
    for each item: @{key: str, val: ValueRef}  in ccx.module_data.items() {
        let elt = C_struct(~[p2i(C_cstr(ccx, item.key)), p2i(item.val)]);
        elts += ~[elt];
    }
    let term = C_struct(~[C_int(0), C_int(0)]);
    elts += ~[term];
    llvm::LLVMSetInitializer(map, C_array(elttype, elts));
    ret map;
}


// FIXME use hashed metadata instead of crate names once we have that
fn create_crate_map(ccx: &@crate_ctxt) -> ValueRef {
    let subcrates: ValueRef[] = ~[];
    let i = 1;
    let cstore = ccx.sess.get_cstore();
    while cstore::have_crate_data(cstore, i) {
        let nm = "_rust_crate_map_" + cstore::get_crate_data(cstore, i).name;
        let cr =
            llvm::LLVMAddGlobal(ccx.llmod, T_int(), str::buf(nm));
        subcrates += ~[p2i(cr)];
        i += 1;
    }
    subcrates += ~[C_int(0)];
    let mapname;
    if ccx.sess.get_opts().library {
        mapname = ccx.link_meta.name;
    } else { mapname = "toplevel"; }
    let sym_name = "_rust_crate_map_" + mapname;
    let arrtype = T_array(T_int(), std::ivec::len[ValueRef](subcrates));
    let maptype = T_struct(~[T_int(), arrtype]);
    let map = llvm::LLVMAddGlobal(ccx.llmod, maptype, str::buf(sym_name));
    llvm::LLVMSetLinkage(map,
                         lib::llvm::LLVMExternalLinkage as llvm::Linkage);
    llvm::LLVMSetInitializer(map,
                             C_struct(~[p2i(create_module_map(ccx)),
                                        C_array(T_int(), subcrates)]));
    ret map;
}

fn write_metadata(cx: &@crate_ctxt, crate: &@ast::crate) {
    if !cx.sess.get_opts().library { ret; }
    let llmeta = C_postr(metadata::encoder::encode_metadata(cx, crate));
    let llconst = trans_common::C_struct(~[llmeta]);
    let llglobal =
        llvm::LLVMAddGlobal(cx.llmod, val_ty(llconst),
                            str::buf("rust_metadata"));
    llvm::LLVMSetInitializer(llglobal, llconst);
    let met_sct_nm = x86::get_meta_sect_name(); //HACK (buf lifetime issue)
    llvm::LLVMSetSection(llglobal, str::buf(met_sct_nm));
    llvm::LLVMSetLinkage(llglobal,
                         lib::llvm::LLVMInternalLinkage as llvm::Linkage);

    let t_ptr_i8 = T_ptr(T_i8());
    llglobal = llvm::LLVMConstBitCast(llglobal, t_ptr_i8);
    let llvm_used =
        llvm::LLVMAddGlobal(cx.llmod, T_array(t_ptr_i8, 1u),
                            str::buf("llvm.used"));
    llvm::LLVMSetLinkage(llvm_used,
                         lib::llvm::LLVMAppendingLinkage as llvm::Linkage);
    llvm::LLVMSetInitializer(llvm_used, C_array(t_ptr_i8, ~[llglobal]));
}

fn trans_crate(sess: &session::session, crate: &@ast::crate, tcx: &ty::ctxt,
               output: &str, amap: &ast_map::map) -> ModuleRef {
    let llmod =
        llvm::LLVMModuleCreateWithNameInContext(str::buf("rust_out"),
                                                llvm::LLVMGetGlobalContext());
    let dat_layt = x86::get_data_layout(); //HACK (buf lifetime issue)
    llvm::LLVMSetDataLayout(llmod, str::buf(dat_layt));
    let targ_trip = x86::get_target_triple(); //HACK (buf lifetime issue)
    llvm::LLVMSetTarget(llmod, str::buf(targ_trip));
    let td = mk_target_data(dat_layt);
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
    let tag_sizes = map::mk_hashmap[ty::t, uint](hasher, eqer);
    let tydescs = map::mk_hashmap[ty::t, @tydesc_info](hasher, eqer);
    let lltypes = map::mk_hashmap[ty::t, TypeRef](hasher, eqer);
    let sha1s = map::mk_hashmap[ty::t, str](hasher, eqer);
    let short_names = map::mk_hashmap[ty::t, str](hasher, eqer);
    let sha = std::sha1::mk_sha1();
    let ccx =
        @{sess: sess,
          llmod: llmod,
          td: td,
          tn: tn,
          externs: new_str_hash[ValueRef](),
          intrinsics: intrinsics,
          item_ids: new_int_hash[ValueRef](),
          ast_map: amap,
          item_symbols: new_int_hash[str](),
          mutable main_fn: none[ValueRef],
          link_meta: link::build_link_meta(sess, *crate, output, sha),
          tag_sizes: tag_sizes,
          discrims: new_int_hash[ValueRef](),
          discrim_symbols: new_int_hash[str](),
          fn_pairs: new_int_hash[ValueRef](),
          consts: new_int_hash[ValueRef](),
          obj_methods: new_int_hash[()](),
          tydescs: tydescs,
          module_data: new_str_hash[ValueRef](),
          lltypes: lltypes,
          glues: glues,
          names: namegen(0),
          sha: sha,
          type_sha1s: sha1s,
          type_short_names: short_names,
          tcx: tcx,
          stats:
              {mutable n_static_tydescs: 0u,
               mutable n_derived_tydescs: 0u,
               mutable n_glues_created: 0u,
               mutable n_null_glues: 0u,
               mutable n_real_glues: 0u,
               fn_times: @mutable ~[]},
          upcalls:
              upcall::declare_upcalls(tn, tydesc_type, taskptr_type, llmod),
          rust_object_type: T_rust_object(),
          tydesc_type: tydesc_type,
          task_type: task_type};
    let cx = new_local_ctxt(ccx);
    collect_items(ccx, crate);
    collect_tag_ctors(ccx, crate);
    trans_constants(ccx, crate);
    trans_mod(cx, crate.node.module);
    create_crate_map(ccx);
    emit_tydescs(ccx);
    // Translate the metadata:

    write_metadata(cx.ccx, crate);
    if ccx.sess.get_opts().stats {
        log_err "--- trans stats ---";
        log_err #fmt("n_static_tydescs: %u", ccx.stats.n_static_tydescs);
        log_err #fmt("n_derived_tydescs: %u", ccx.stats.n_derived_tydescs);
        log_err #fmt("n_glues_created: %u", ccx.stats.n_glues_created);
        log_err #fmt("n_null_glues: %u", ccx.stats.n_null_glues);
        log_err #fmt("n_real_glues: %u", ccx.stats.n_real_glues);


        for timing: {ident: str, time: int}  in *ccx.stats.fn_times {
            log_err #fmt("time: %s took %d ms", timing.ident, timing.time);
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
