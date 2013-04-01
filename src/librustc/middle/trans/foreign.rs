// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use back::{link, abi};
use lib::llvm::{SequentiallyConsistent, Acquire, Release, Xchg};
use lib::llvm::{TypeRef, ValueRef};
use lib;
use middle::trans::base::*;
use middle::trans::cabi;
use middle::trans::cabi_x86_64::*;
use middle::trans::cabi_arm;
use middle::trans::cabi_mips::*;
use middle::trans::build::*;
use middle::trans::callee::*;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::expr::Ignore;
use middle::trans::machine::llsize_of;
use middle::trans::glue;
use middle::trans::machine;
use middle::trans::type_of::*;
use middle::trans::type_of;
use middle::ty;
use middle::ty::{FnSig, arg};
use util::ppaux::ty_to_str;

use syntax::codemap::span;
use syntax::{ast, ast_util};
use syntax::{attr, ast_map};
use syntax::opt_vec;
use syntax::parse::token::special_idents;
use syntax::abi::{Architecture, X86, X86_64, Arm, Mips};
use syntax::abi::{RustIntrinsic, Rust, Stdcall, Fastcall,
                  Cdecl, Aapcs, C};

fn abi_info(arch: Architecture) -> @cabi::ABIInfo {
    return match arch {
        X86_64 => x86_64_abi_info(),
        Arm => cabi_arm::abi_info(),
        Mips => mips_abi_info(),
        X86 => cabi::llvm_abi_info()
    }
}

pub fn link_name(ccx: @CrateContext, i: @ast::foreign_item) -> @~str {
     match attr::first_attr_value_str_by_name(i.attrs, ~"link_name") {
        None => ccx.sess.str_of(i.ident),
        Some(ln) => ln,
    }
}

struct ShimTypes {
    fn_sig: ty::FnSig,

    /// LLVM types that will appear on the foreign function
    llsig: LlvmSignature,

    /// True if there is a return value (not bottom, not unit)
    ret_def: bool,

    /// Type of the struct we will use to shuttle values back and forth.
    /// This is always derived from the llsig.
    bundle_ty: TypeRef,

    /// Type of the shim function itself.
    shim_fn_ty: TypeRef,

    /// Adapter object for handling native ABI rules (trust me, you
    /// don't want to know).
    fn_ty: cabi::FnType
}

struct LlvmSignature {
    llarg_tys: ~[TypeRef],
    llret_ty: TypeRef,
}

fn foreign_signature(ccx: @CrateContext,
                     fn_sig: &ty::FnSig) -> LlvmSignature {
    /*!
     * The ForeignSignature is the LLVM types of the arguments/return type
     * of a function.  Note that these LLVM types are not quite the same
     * as the LLVM types would be for a native Rust function because foreign
     * functions just plain ignore modes.  They also don't pass aggregate
     * values by pointer like we do.
     */

    let llarg_tys = fn_sig.inputs.map(|arg| type_of(ccx, arg.ty));
    let llret_ty = type_of::type_of(ccx, fn_sig.output);
    LlvmSignature {llarg_tys: llarg_tys, llret_ty: llret_ty}
}

fn shim_types(ccx: @CrateContext, id: ast::node_id) -> ShimTypes {
    let fn_sig = match ty::get(ty::node_id_to_type(ccx.tcx, id)).sty {
        ty::ty_bare_fn(ref fn_ty) => copy fn_ty.sig,
        _ => ccx.sess.bug(~"c_arg_and_ret_lltys called on non-function type")
    };
    let llsig = foreign_signature(ccx, &fn_sig);
    let bundle_ty = T_struct(vec::append_one(copy llsig.llarg_tys,
                                             T_ptr(llsig.llret_ty)));
    let ret_def =
        !ty::type_is_bot(fn_sig.output) &&
        !ty::type_is_nil(fn_sig.output);
    let fn_ty =
        abi_info(ccx.sess.targ_cfg.arch).compute_info(
            llsig.llarg_tys,
            llsig.llret_ty,
            ret_def);
    ShimTypes {
        fn_sig: fn_sig,
        llsig: llsig,
        ret_def: ret_def,
        bundle_ty: bundle_ty,
        shim_fn_ty: T_fn(~[T_ptr(bundle_ty)], T_void()),
        fn_ty: fn_ty
    }
}

type shim_arg_builder<'self> =
    &'self fn(bcx: block, tys: &ShimTypes,
              llargbundle: ValueRef) -> ~[ValueRef];

type shim_ret_builder<'self> =
    &'self fn(bcx: block, tys: &ShimTypes,
              llargbundle: ValueRef,
              llretval: ValueRef);

fn build_shim_fn_(ccx: @CrateContext,
                  +shim_name: ~str,
                  llbasefn: ValueRef,
                  tys: &ShimTypes,
                  cc: lib::llvm::CallConv,
                  arg_builder: shim_arg_builder,
                  ret_builder: shim_ret_builder) -> ValueRef
{
    let llshimfn = decl_internal_cdecl_fn(
        ccx.llmod, shim_name, tys.shim_fn_ty);

    // Declare the body of the shim function:
    let fcx = new_fn_ctxt(ccx, ~[], llshimfn, None);
    let bcx = top_scope_block(fcx, None);
    let lltop = bcx.llbb;
    let llargbundle = get_param(llshimfn, 0u);
    let llargvals = arg_builder(bcx, tys, llargbundle);

    // Create the call itself and store the return value:
    let llretval = CallWithConv(bcx, llbasefn, llargvals, cc);

    ret_builder(bcx, tys, llargbundle, llretval);

    build_return(bcx);
    finish_fn(fcx, lltop);

    return llshimfn;
}

type wrap_arg_builder<'self> =
    &'self fn(bcx: block, tys: &ShimTypes,
              llwrapfn: ValueRef, llargbundle: ValueRef);

type wrap_ret_builder<'self> =
    &'self fn(bcx: block, tys: &ShimTypes,
              llargbundle: ValueRef);

fn build_wrap_fn_(ccx: @CrateContext,
                  tys: &ShimTypes,
                  llshimfn: ValueRef,
                  llwrapfn: ValueRef,
                  shim_upcall: ValueRef,
                  arg_builder: wrap_arg_builder,
                  ret_builder: wrap_ret_builder)
{
    let _icx = ccx.insn_ctxt("foreign::build_wrap_fn_");
    let fcx = new_fn_ctxt(ccx, ~[], llwrapfn, None);
    let bcx = top_scope_block(fcx, None);
    let lltop = bcx.llbb;

    // Allocate the struct and write the arguments into it.
    let llargbundle = alloca(bcx, tys.bundle_ty);
    arg_builder(bcx, tys, llwrapfn, llargbundle);

    // Create call itself.
    let llshimfnptr = PointerCast(bcx, llshimfn, T_ptr(T_i8()));
    let llrawargbundle = PointerCast(bcx, llargbundle, T_ptr(T_i8()));
    Call(bcx, shim_upcall, ~[llrawargbundle, llshimfnptr]);
    ret_builder(bcx, tys, llargbundle);

    tie_up_header_blocks(fcx, lltop);

    // Make sure our standard return block (that we didn't use) is terminated
    let ret_cx = raw_block(fcx, false, fcx.llreturn);
    Unreachable(ret_cx);
}

// For each foreign function F, we generate a wrapper function W and a shim
// function S that all work together.  The wrapper function W is the function
// that other rust code actually invokes.  Its job is to marshall the
// arguments into a struct.  It then uses a small bit of assembly to switch
// over to the C stack and invoke the shim function.  The shim function S then
// unpacks the arguments from the struct and invokes the actual function F
// according to its specified calling convention.
//
// Example: Given a foreign c-stack function F(x: X, y: Y) -> Z,
// we generate a wrapper function W that looks like:
//
//    void W(Z* dest, void *env, X x, Y y) {
//        struct { X x; Y y; Z *z; } args = { x, y, z };
//        call_on_c_stack_shim(S, &args);
//    }
//
// The shim function S then looks something like:
//
//     void S(struct { X x; Y y; Z *z; } *args) {
//         *args->z = F(args->x, args->y);
//     }
//
// However, if the return type of F is dynamically sized or of aggregate type,
// the shim function looks like:
//
//     void S(struct { X x; Y y; Z *z; } *args) {
//         F(args->z, args->x, args->y);
//     }
//
// Note: on i386, the layout of the args struct is generally the same
// as the desired layout of the arguments on the C stack.  Therefore,
// we could use upcall_alloc_c_stack() to allocate the `args`
// structure and switch the stack pointer appropriately to avoid a
// round of copies.  (In fact, the shim function itself is
// unnecessary). We used to do this, in fact, and will perhaps do so
// in the future.
pub fn trans_foreign_mod(ccx: @CrateContext,
                         path: &ast_map::path,
                         foreign_mod: &ast::foreign_mod)
{
    let _icx = ccx.insn_ctxt("foreign::trans_foreign_mod");

    let arch = ccx.sess.targ_cfg.arch;
    let abi = match foreign_mod.abis.for_arch(arch) {
        None => {
            ccx.sess.fatal(
                fmt!("No suitable ABI for target architecture \
                      in module %s",
                     ast_map::path_to_str(*path,
                                          ccx.sess.intr())));
        }

        Some(abi) => abi,
    };

    for vec::each(foreign_mod.items) |&foreign_item| {
        match foreign_item.node {
            ast::foreign_item_fn(*) => {
                let id = foreign_item.id;
                match abi {
                    RustIntrinsic => {
                        // Intrinsics are emitted by monomorphic fn
                    }

                    Rust => {
                        // FIXME(#3678) Implement linking to foreign fns with Rust ABI
                        ccx.sess.unimpl(
                            fmt!("Foreign functions with Rust ABI"));
                    }

                    Stdcall => {
                        build_foreign_fn(ccx, id, foreign_item,
                                         lib::llvm::X86StdcallCallConv);
                    }

                    Fastcall => {
                        build_foreign_fn(ccx, id, foreign_item,
                                         lib::llvm::X86FastcallCallConv);
                    }

                    Cdecl => {
                        // FIXME(#3678) should really be more specific
                        build_foreign_fn(ccx, id, foreign_item,
                                         lib::llvm::CCallConv);
                    }

                    Aapcs => {
                        // FIXME(#3678) should really be more specific
                        build_foreign_fn(ccx, id, foreign_item,
                                         lib::llvm::CCallConv);
                    }

                    C => {
                        build_foreign_fn(ccx, id, foreign_item,
                                         lib::llvm::CCallConv);
                    }
                }
            }
            ast::foreign_item_const(*) => {
                let ident = ccx.sess.parse_sess.interner.get(
                    foreign_item.ident);
                ccx.item_symbols.insert(foreign_item.id, copy *ident);
            }
        }
    }

    fn build_foreign_fn(ccx: @CrateContext,
                        id: ast::node_id,
                        foreign_item: @ast::foreign_item,
                        cc: lib::llvm::CallConv)
    {
        let llwrapfn = get_item_val(ccx, id);
        let tys = shim_types(ccx, id);
        if attr::attrs_contains_name(
            foreign_item.attrs, "rust_stack")
        {
            build_direct_fn(ccx, llwrapfn, foreign_item,
                            &tys, cc);
        } else {
            let llshimfn = build_shim_fn(ccx, foreign_item,
                                         &tys, cc);
            build_wrap_fn(ccx, &tys, llshimfn, llwrapfn);
        }
    }

    fn build_shim_fn(ccx: @CrateContext,
                     foreign_item: @ast::foreign_item,
                     tys: &ShimTypes,
                     cc: lib::llvm::CallConv) -> ValueRef
    {
        /*!
         *
         * Build S, from comment above:
         *
         *     void S(struct { X x; Y y; Z *z; } *args) {
         *         F(args->z, args->x, args->y);
         *     }
         */

        let _icx = ccx.insn_ctxt("foreign::build_shim_fn");

        fn build_args(bcx: block, tys: &ShimTypes,
                      llargbundle: ValueRef) -> ~[ValueRef] {
            let _icx = bcx.insn_ctxt("foreign::shim::build_args");
            tys.fn_ty.build_shim_args(
                bcx, tys.llsig.llarg_tys, llargbundle)
        }

        fn build_ret(bcx: block, tys: &ShimTypes,
                     llargbundle: ValueRef, llretval: ValueRef)  {
            let _icx = bcx.insn_ctxt("foreign::shim::build_ret");
            tys.fn_ty.build_shim_ret(
                bcx, tys.llsig.llarg_tys,
                tys.ret_def, llargbundle, llretval);
        }

        let lname = link_name(ccx, foreign_item);
        let llbasefn = base_fn(ccx, *lname, tys, cc);
        // Name the shim function
        let shim_name = *lname + ~"__c_stack_shim";
        return build_shim_fn_(ccx, shim_name, llbasefn, tys, cc,
                           build_args, build_ret);
    }

    fn base_fn(ccx: @CrateContext, lname: &str, tys: &ShimTypes,
               cc: lib::llvm::CallConv) -> ValueRef {
        // Declare the "prototype" for the base function F:
        do tys.fn_ty.decl_fn |fnty| {
            decl_fn(ccx.llmod, lname, cc, fnty)
        }
    }

    // FIXME (#2535): this is very shaky and probably gets ABIs wrong all
    // over the place
    fn build_direct_fn(ccx: @CrateContext, decl: ValueRef,
                       item: @ast::foreign_item, tys: &ShimTypes,
                       cc: lib::llvm::CallConv) {
        let fcx = new_fn_ctxt(ccx, ~[], decl, None);
        let bcx = top_scope_block(fcx, None), lltop = bcx.llbb;
        let llbasefn = base_fn(ccx, *link_name(ccx, item), tys, cc);
        let ty = ty::lookup_item_type(ccx.tcx,
                                      ast_util::local_def(item.id)).ty;
        let args = vec::from_fn(ty::ty_fn_args(ty).len(), |i| {
            get_param(decl, i + first_real_arg)
        });
        let retval = Call(bcx, llbasefn, args);
        if !ty::type_is_nil(ty::ty_fn_ret(ty)) {
            Store(bcx, retval, fcx.llretptr);
        }
        build_return(bcx);
        finish_fn(fcx, lltop);
    }

    fn build_wrap_fn(ccx: @CrateContext,
                     tys: &ShimTypes,
                     llshimfn: ValueRef,
                     llwrapfn: ValueRef) {
        /*!
         *
         * Build W, from comment above:
         *
         *     void W(Z* dest, void *env, X x, Y y) {
         *         struct { X x; Y y; Z *z; } args = { x, y, z };
         *         call_on_c_stack_shim(S, &args);
         *     }
         *
         * One thing we have to be very careful of is to
         * account for the Rust modes.
         */

        let _icx = ccx.insn_ctxt("foreign::build_wrap_fn");

        build_wrap_fn_(ccx, tys, llshimfn, llwrapfn,
                       ccx.upcalls.call_shim_on_c_stack,
                       build_args, build_ret);

        fn build_args(bcx: block, tys: &ShimTypes,
                      llwrapfn: ValueRef, llargbundle: ValueRef) {
            let _icx = bcx.insn_ctxt("foreign::wrap::build_args");
            let ccx = bcx.ccx();
            let n = vec::len(tys.llsig.llarg_tys);
            let implicit_args = first_real_arg; // return + env
            for uint::range(0, n) |i| {
                let mut llargval = get_param(llwrapfn, i + implicit_args);

                // In some cases, Rust will pass a pointer which the
                // native C type doesn't have.  In that case, just
                // load the value from the pointer.
                if type_of::arg_is_indirect(ccx, &tys.fn_sig.inputs[i]) {
                    llargval = Load(bcx, llargval);
                }

                store_inbounds(bcx, llargval, llargbundle, ~[0u, i]);
            }
            let llretptr = get_param(llwrapfn, 0u);
            store_inbounds(bcx, llretptr, llargbundle, ~[0u, n]);
        }

        fn build_ret(bcx: block, _tys: &ShimTypes,
                     _llargbundle: ValueRef) {
            let _icx = bcx.insn_ctxt("foreign::wrap::build_ret");
            RetVoid(bcx);
        }
    }
}

pub fn trans_intrinsic(ccx: @CrateContext,
                       decl: ValueRef,
                       item: @ast::foreign_item,
                       +path: ast_map::path,
                       substs: @param_substs,
                       ref_id: Option<ast::node_id>) {
    debug!("trans_intrinsic(item.ident=%s)", *ccx.sess.str_of(item.ident));

    // XXX: Bad copy.
    let fcx = new_fn_ctxt_w_id(ccx, path, decl, item.id, None,
                               Some(copy substs), Some(item.span));
    let mut bcx = top_scope_block(fcx, None), lltop = bcx.llbb;
    match *ccx.sess.str_of(item.ident) {
        ~"atomic_cxchg" => {
            let old = AtomicCmpXchg(bcx,
                                    get_param(decl, first_real_arg),
                                    get_param(decl, first_real_arg + 1u),
                                    get_param(decl, first_real_arg + 2u),
                                    SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_cxchg_acq" => {
            let old = AtomicCmpXchg(bcx,
                                    get_param(decl, first_real_arg),
                                    get_param(decl, first_real_arg + 1u),
                                    get_param(decl, first_real_arg + 2u),
                                    Acquire);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_cxchg_rel" => {
            let old = AtomicCmpXchg(bcx,
                                    get_param(decl, first_real_arg),
                                    get_param(decl, first_real_arg + 1u),
                                    get_param(decl, first_real_arg + 2u),
                                    Release);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xchg" => {
            let old = AtomicRMW(bcx, Xchg,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xchg_acq" => {
            let old = AtomicRMW(bcx, Xchg,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Acquire);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xchg_rel" => {
            let old = AtomicRMW(bcx, Xchg,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Release);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xadd" => {
            let old = AtomicRMW(bcx, lib::llvm::Add,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xadd_acq" => {
            let old = AtomicRMW(bcx, lib::llvm::Add,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Acquire);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xadd_rel" => {
            let old = AtomicRMW(bcx, lib::llvm::Add,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Release);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xsub" => {
            let old = AtomicRMW(bcx, lib::llvm::Sub,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                SequentiallyConsistent);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xsub_acq" => {
            let old = AtomicRMW(bcx, lib::llvm::Sub,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Acquire);
            Store(bcx, old, fcx.llretptr);
        }
        ~"atomic_xsub_rel" => {
            let old = AtomicRMW(bcx, lib::llvm::Sub,
                                get_param(decl, first_real_arg),
                                get_param(decl, first_real_arg + 1u),
                                Release);
            Store(bcx, old, fcx.llretptr);
        }
        ~"size_of" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Store(bcx, C_uint(ccx, machine::llsize_of_real(ccx, lltp_ty)),
                  fcx.llretptr);
        }
        ~"move_val" => {
            // Create a datum reflecting the value being moved:
            //
            // - the datum will be by ref if the value is non-immediate;
            //
            // - the datum has a RevokeClean source because, that way,
            //   the `move_to()` method does not feel compelled to
            //   zero out the memory where the datum resides.  Zeroing
            //   is not necessary since, for intrinsics, there is no
            //   cleanup to concern ourselves with.
            let tp_ty = substs.tys[0];
            let mode = appropriate_mode(tp_ty);
            let src = Datum {val: get_param(decl, first_real_arg + 1u),
                             ty: tp_ty, mode: mode, source: RevokeClean};
            bcx = src.move_to(bcx, DROP_EXISTING,
                              get_param(decl, first_real_arg));
        }
        ~"move_val_init" => {
            // See comments for `"move_val"`.
            let tp_ty = substs.tys[0];
            let mode = appropriate_mode(tp_ty);
            let src = Datum {val: get_param(decl, first_real_arg + 1u),
                             ty: tp_ty, mode: mode, source: RevokeClean};
            bcx = src.move_to(bcx, INIT, get_param(decl, first_real_arg));
        }
        ~"min_align_of" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Store(bcx, C_uint(ccx, machine::llalign_of_min(ccx, lltp_ty)),
                  fcx.llretptr);
        }
        ~"pref_align_of"=> {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            Store(bcx, C_uint(ccx, machine::llalign_of_pref(ccx, lltp_ty)),
                  fcx.llretptr);
        }
        ~"get_tydesc" => {
            let tp_ty = substs.tys[0];
            let static_ti = get_tydesc(ccx, tp_ty);
            glue::lazily_emit_all_tydesc_glue(ccx, static_ti);

            // FIXME (#3727): change this to T_ptr(ccx.tydesc_ty) when the
            // core::sys copy of the get_tydesc interface dies off.
            let td = PointerCast(bcx, static_ti.tydesc, T_ptr(T_nil()));
            Store(bcx, td, fcx.llretptr);
        }
        ~"init" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            if !ty::type_is_nil(tp_ty) {
                Store(bcx, C_null(lltp_ty), fcx.llretptr);
            }
        }
        ~"forget" => {}
        ~"reinterpret_cast" => {
            let tp_ty = substs.tys[0];
            let lltp_ty = type_of::type_of(ccx, tp_ty);
            let llout_ty = type_of::type_of(ccx, substs.tys[1]);
            let tp_sz = machine::llbitsize_of_real(ccx, lltp_ty),
            out_sz = machine::llbitsize_of_real(ccx, llout_ty);
          if tp_sz != out_sz {
              let sp = match *ccx.tcx.items.get(&ref_id.get()) {
                  ast_map::node_expr(e) => e.span,
                  _ => fail!(~"reinterpret_cast or forget has non-expr arg")
              };
              ccx.sess.span_fatal(
                  sp, fmt!("reinterpret_cast called on types \
                            with different size: %s (%u bit(s)) to %s \
                            (%u bit(s))",
                           ty_to_str(ccx.tcx, tp_ty), tp_sz,
                           ty_to_str(ccx.tcx, substs.tys[1]), out_sz));
          }
          if !ty::type_is_nil(substs.tys[1]) {
              // NB: Do not use a Load and Store here. This causes
              // massive code bloat when reinterpret_cast is used on
              // large structural types.
              let llretptr = PointerCast(bcx, fcx.llretptr, T_ptr(T_i8()));
              let llcast = get_param(decl, first_real_arg);
              let llcast = PointerCast(bcx, llcast, T_ptr(T_i8()));
              call_memcpy(bcx, llretptr, llcast, llsize_of(ccx, lltp_ty));
          }
        }
        ~"addr_of" => {
            Store(bcx, get_param(decl, first_real_arg), fcx.llretptr);
        }
        ~"needs_drop" => {
            let tp_ty = substs.tys[0];
            Store(bcx,
                  C_bool(ty::type_needs_drop(ccx.tcx, tp_ty)),
                  fcx.llretptr);
        }
        ~"visit_tydesc" => {
            let td = get_param(decl, first_real_arg);
            let visitor = get_param(decl, first_real_arg + 1u);
            let td = PointerCast(bcx, td, T_ptr(ccx.tydesc_type));
            glue::call_tydesc_glue_full(bcx, visitor, td,
                                        abi::tydesc_field_visit_glue, None);
        }
        ~"frame_address" => {
            let frameaddress = *ccx.intrinsics.get(&~"llvm.frameaddress");
            let frameaddress_val = Call(bcx, frameaddress, ~[C_i32(0i32)]);
            let star_u8 = ty::mk_imm_ptr(
                bcx.tcx(),
                ty::mk_mach_uint(bcx.tcx(), ast::ty_u8));
            let fty = ty::mk_closure(bcx.tcx(), ty::ClosureTy {
                purity: ast::impure_fn,
                sigil: ast::BorrowedSigil,
                onceness: ast::Many,
                region: ty::re_bound(ty::br_anon(0)),
                sig: FnSig {bound_lifetime_names: opt_vec::Empty,
                            inputs: ~[arg {mode: ast::expl(ast::by_copy),
                                           ty: star_u8}],
                            output: ty::mk_nil(bcx.tcx())}
            });
            let datum = Datum {val: get_param(decl, first_real_arg),
                               mode: ByRef, ty: fty, source: ZeroMem};
            let arg_vals = ~[frameaddress_val];
            bcx = trans_call_inner(
                bcx, None, fty, ty::mk_nil(bcx.tcx()),
                |bcx| Callee {bcx: bcx, data: Closure(datum)},
                ArgVals(arg_vals), Ignore, DontAutorefArg);
        }
        ~"morestack_addr" => {
            // XXX This is a hack to grab the address of this particular
            // native function. There should be a general in-language
            // way to do this
            let llfty = type_of_fn(bcx.ccx(), ~[], ty::mk_nil(bcx.tcx()));
            let morestack_addr = decl_cdecl_fn(
                bcx.ccx().llmod, ~"__morestack", llfty);
            let morestack_addr = PointerCast(bcx, morestack_addr,
                                             T_ptr(T_nil()));
            Store(bcx, morestack_addr, fcx.llretptr);
        }
        ~"memmove32" => {
            let dst_ptr = get_param(decl, first_real_arg);
            let src_ptr = get_param(decl, first_real_arg + 1);
            let size = get_param(decl, first_real_arg + 2);
            let align = C_i32(1);
            let volatile = C_i1(false);
            let llfn = *bcx.ccx().intrinsics.get(
                &~"llvm.memmove.p0i8.p0i8.i32");
            Call(bcx, llfn, ~[dst_ptr, src_ptr, size, align, volatile]);
        }
        ~"memmove64" => {
            let dst_ptr = get_param(decl, first_real_arg);
            let src_ptr = get_param(decl, first_real_arg + 1);
            let size = get_param(decl, first_real_arg + 2);
            let align = C_i32(1);
            let volatile = C_i1(false);
            let llfn = *bcx.ccx().intrinsics.get(
                &~"llvm.memmove.p0i8.p0i8.i64");
            Call(bcx, llfn, ~[dst_ptr, src_ptr, size, align, volatile]);
        }
        ~"sqrtf32" => {
            let x = get_param(decl, first_real_arg);
            let sqrtf = *ccx.intrinsics.get(&~"llvm.sqrt.f32");
            Store(bcx, Call(bcx, sqrtf, ~[x]), fcx.llretptr);
        }
        ~"sqrtf64" => {
            let x = get_param(decl, first_real_arg);
            let sqrtf = *ccx.intrinsics.get(&~"llvm.sqrt.f64");
            Store(bcx, Call(bcx, sqrtf, ~[x]), fcx.llretptr);
        }
        ~"powif32" => {
            let a = get_param(decl, first_real_arg);
            let x = get_param(decl, first_real_arg + 1u);
            let powif = *ccx.intrinsics.get(&~"llvm.powi.f32");
            Store(bcx, Call(bcx, powif, ~[a, x]), fcx.llretptr);
        }
        ~"powif64" => {
            let a = get_param(decl, first_real_arg);
            let x = get_param(decl, first_real_arg + 1u);
            let powif = *ccx.intrinsics.get(&~"llvm.powi.f64");
            Store(bcx, Call(bcx, powif, ~[a, x]), fcx.llretptr);
        }
        ~"sinf32" => {
            let x = get_param(decl, first_real_arg);
            let sinf = *ccx.intrinsics.get(&~"llvm.sin.f32");
            Store(bcx, Call(bcx, sinf, ~[x]), fcx.llretptr);
        }
        ~"sinf64" => {
            let x = get_param(decl, first_real_arg);
            let sinf = *ccx.intrinsics.get(&~"llvm.sin.f64");
            Store(bcx, Call(bcx, sinf, ~[x]), fcx.llretptr);
        }
        ~"cosf32" => {
            let x = get_param(decl, first_real_arg);
            let cosf = *ccx.intrinsics.get(&~"llvm.cos.f32");
            Store(bcx, Call(bcx, cosf, ~[x]), fcx.llretptr);
        }
        ~"cosf64" => {
            let x = get_param(decl, first_real_arg);
            let cosf = *ccx.intrinsics.get(&~"llvm.cos.f64");
            Store(bcx, Call(bcx, cosf, ~[x]), fcx.llretptr);
        }
        ~"powf32" => {
            let a = get_param(decl, first_real_arg);
            let x = get_param(decl, first_real_arg + 1u);
            let powf = *ccx.intrinsics.get(&~"llvm.pow.f32");
            Store(bcx, Call(bcx, powf, ~[a, x]), fcx.llretptr);
        }
        ~"powf64" => {
            let a = get_param(decl, first_real_arg);
            let x = get_param(decl, first_real_arg + 1u);
            let powf = *ccx.intrinsics.get(&~"llvm.pow.f64");
            Store(bcx, Call(bcx, powf, ~[a, x]), fcx.llretptr);
        }
        ~"expf32" => {
            let x = get_param(decl, first_real_arg);
            let expf = *ccx.intrinsics.get(&~"llvm.exp.f32");
            Store(bcx, Call(bcx, expf, ~[x]), fcx.llretptr);
        }
        ~"expf64" => {
            let x = get_param(decl, first_real_arg);
            let expf = *ccx.intrinsics.get(&~"llvm.exp.f64");
            Store(bcx, Call(bcx, expf, ~[x]), fcx.llretptr);
        }
        ~"exp2f32" => {
            let x = get_param(decl, first_real_arg);
            let exp2f = *ccx.intrinsics.get(&~"llvm.exp2.f32");
            Store(bcx, Call(bcx, exp2f, ~[x]), fcx.llretptr);
        }
        ~"exp2f64" => {
            let x = get_param(decl, first_real_arg);
            let exp2f = *ccx.intrinsics.get(&~"llvm.exp2.f64");
            Store(bcx, Call(bcx, exp2f, ~[x]), fcx.llretptr);
        }
        ~"logf32" => {
            let x = get_param(decl, first_real_arg);
            let logf = *ccx.intrinsics.get(&~"llvm.log.f32");
            Store(bcx, Call(bcx, logf, ~[x]), fcx.llretptr);
        }
        ~"logf64" => {
            let x = get_param(decl, first_real_arg);
            let logf = *ccx.intrinsics.get(&~"llvm.log.f64");
            Store(bcx, Call(bcx, logf, ~[x]), fcx.llretptr);
        }
        ~"log10f32" => {
            let x = get_param(decl, first_real_arg);
            let log10f = *ccx.intrinsics.get(&~"llvm.log10.f32");
            Store(bcx, Call(bcx, log10f, ~[x]), fcx.llretptr);
        }
        ~"log10f64" => {
            let x = get_param(decl, first_real_arg);
            let log10f = *ccx.intrinsics.get(&~"llvm.log10.f64");
            Store(bcx, Call(bcx, log10f, ~[x]), fcx.llretptr);
        }
        ~"log2f32" => {
            let x = get_param(decl, first_real_arg);
            let log2f = *ccx.intrinsics.get(&~"llvm.log2.f32");
            Store(bcx, Call(bcx, log2f, ~[x]), fcx.llretptr);
        }
        ~"log2f64" => {
            let x = get_param(decl, first_real_arg);
            let log2f = *ccx.intrinsics.get(&~"llvm.log2.f64");
            Store(bcx, Call(bcx, log2f, ~[x]), fcx.llretptr);
        }
        ~"fmaf32" => {
            let a = get_param(decl, first_real_arg);
            let b = get_param(decl, first_real_arg + 1u);
            let c = get_param(decl, first_real_arg + 2u);
            let fmaf = *ccx.intrinsics.get(&~"llvm.fma.f32");
            Store(bcx, Call(bcx, fmaf, ~[a, b, c]), fcx.llretptr);
        }
        ~"fmaf64" => {
            let a = get_param(decl, first_real_arg);
            let b = get_param(decl, first_real_arg + 1u);
            let c = get_param(decl, first_real_arg + 2u);
            let fmaf = *ccx.intrinsics.get(&~"llvm.fma.f64");
            Store(bcx, Call(bcx, fmaf, ~[a, b, c]), fcx.llretptr);
        }
        ~"fabsf32" => {
            let x = get_param(decl, first_real_arg);
            let fabsf = *ccx.intrinsics.get(&~"llvm.fabs.f32");
            Store(bcx, Call(bcx, fabsf, ~[x]), fcx.llretptr);
        }
        ~"fabsf64" => {
            let x = get_param(decl, first_real_arg);
            let fabsf = *ccx.intrinsics.get(&~"llvm.fabs.f64");
            Store(bcx, Call(bcx, fabsf, ~[x]), fcx.llretptr);
        }
        ~"floorf32" => {
            let x = get_param(decl, first_real_arg);
            let floorf = *ccx.intrinsics.get(&~"llvm.floor.f32");
            Store(bcx, Call(bcx, floorf, ~[x]), fcx.llretptr);
        }
        ~"floorf64" => {
            let x = get_param(decl, first_real_arg);
            let floorf = *ccx.intrinsics.get(&~"llvm.floor.f64");
            Store(bcx, Call(bcx, floorf, ~[x]), fcx.llretptr);
        }
        ~"ceilf32" => {
            let x = get_param(decl, first_real_arg);
            let ceilf = *ccx.intrinsics.get(&~"llvm.ceil.f32");
            Store(bcx, Call(bcx, ceilf, ~[x]), fcx.llretptr);
        }
        ~"ceilf64" => {
            let x = get_param(decl, first_real_arg);
            let ceilf = *ccx.intrinsics.get(&~"llvm.ceil.f64");
            Store(bcx, Call(bcx, ceilf, ~[x]), fcx.llretptr);
        }
        ~"truncf32" => {
            let x = get_param(decl, first_real_arg);
            let truncf = *ccx.intrinsics.get(&~"llvm.trunc.f32");
            Store(bcx, Call(bcx, truncf, ~[x]), fcx.llretptr);
        }
        ~"truncf64" => {
            let x = get_param(decl, first_real_arg);
            let truncf = *ccx.intrinsics.get(&~"llvm.trunc.f64");
            Store(bcx, Call(bcx, truncf, ~[x]), fcx.llretptr);
        }
        ~"ctpop8" => {
            let x = get_param(decl, first_real_arg);
            let ctpop = *ccx.intrinsics.get(&~"llvm.ctpop.i8");
            Store(bcx, Call(bcx, ctpop, ~[x]), fcx.llretptr)
        }
        ~"ctpop16" => {
            let x = get_param(decl, first_real_arg);
            let ctpop = *ccx.intrinsics.get(&~"llvm.ctpop.i16");
            Store(bcx, Call(bcx, ctpop, ~[x]), fcx.llretptr)
        }
        ~"ctpop32" => {
            let x = get_param(decl, first_real_arg);
            let ctpop = *ccx.intrinsics.get(&~"llvm.ctpop.i32");
            Store(bcx, Call(bcx, ctpop, ~[x]), fcx.llretptr)
        }
        ~"ctpop64" => {
            let x = get_param(decl, first_real_arg);
            let ctpop = *ccx.intrinsics.get(&~"llvm.ctpop.i64");
            Store(bcx, Call(bcx, ctpop, ~[x]), fcx.llretptr)
        }
        ~"ctlz8" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let ctlz = *ccx.intrinsics.get(&~"llvm.ctlz.i8");
            Store(bcx, Call(bcx, ctlz, ~[x, y]), fcx.llretptr)
        }
        ~"ctlz16" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let ctlz = *ccx.intrinsics.get(&~"llvm.ctlz.i16");
            Store(bcx, Call(bcx, ctlz, ~[x, y]), fcx.llretptr)
        }
        ~"ctlz32" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let ctlz = *ccx.intrinsics.get(&~"llvm.ctlz.i32");
            Store(bcx, Call(bcx, ctlz, ~[x, y]), fcx.llretptr)
        }
        ~"ctlz64" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let ctlz = *ccx.intrinsics.get(&~"llvm.ctlz.i64");
            Store(bcx, Call(bcx, ctlz, ~[x, y]), fcx.llretptr)
        }
        ~"cttz8" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let cttz = *ccx.intrinsics.get(&~"llvm.cttz.i8");
            Store(bcx, Call(bcx, cttz, ~[x, y]), fcx.llretptr)
        }
        ~"cttz16" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let cttz = *ccx.intrinsics.get(&~"llvm.cttz.i16");
            Store(bcx, Call(bcx, cttz, ~[x, y]), fcx.llretptr)
        }
        ~"cttz32" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let cttz = *ccx.intrinsics.get(&~"llvm.cttz.i32");
            Store(bcx, Call(bcx, cttz, ~[x, y]), fcx.llretptr)
        }
        ~"cttz64" => {
            let x = get_param(decl, first_real_arg);
            let y = C_i1(false);
            let cttz = *ccx.intrinsics.get(&~"llvm.cttz.i64");
            Store(bcx, Call(bcx, cttz, ~[x, y]), fcx.llretptr)
        }
        ~"bswap16" => {
            let x = get_param(decl, first_real_arg);
            let cttz = *ccx.intrinsics.get(&~"llvm.bswap.i16");
            Store(bcx, Call(bcx, cttz, ~[x]), fcx.llretptr)
        }
        ~"bswap32" => {
            let x = get_param(decl, first_real_arg);
            let cttz = *ccx.intrinsics.get(&~"llvm.bswap.i32");
            Store(bcx, Call(bcx, cttz, ~[x]), fcx.llretptr)
        }
        ~"bswap64" => {
            let x = get_param(decl, first_real_arg);
            let cttz = *ccx.intrinsics.get(&~"llvm.bswap.i64");
            Store(bcx, Call(bcx, cttz, ~[x]), fcx.llretptr)
        }
        _ => {
            // Could we make this an enum rather than a string? does it get
            // checked earlier?
            ccx.sess.span_bug(item.span, ~"unknown intrinsic");
        }
    }
    build_return(bcx);
    finish_fn(fcx, lltop);
}

/**
 * Translates a "crust" fn, meaning a Rust fn that can be called
 * from C code.  In this case, we have to perform some adaptation
 * to (1) switch back to the Rust stack and (2) adapt the C calling
 * convention to our own.
 *
 * Example: Given a crust fn F(x: X, y: Y) -> Z, we generate a
 * Rust function R as normal:
 *
 *    void R(Z* dest, void *env, X x, Y y) {...}
 *
 * and then we generate a wrapper function W that looks like:
 *
 *    Z W(X x, Y y) {
 *        struct { X x; Y y; Z *z; } args = { x, y, z };
 *        call_on_c_stack_shim(S, &args);
 *    }
 *
 * Note that the wrapper follows the foreign (typically "C") ABI.
 * The wrapper is the actual "value" of the foreign fn.  Finally,
 * we generate a shim function S that looks like:
 *
 *     void S(struct { X x; Y y; Z *z; } *args) {
 *         R(args->z, NULL, args->x, args->y);
 *     }
 */
pub fn trans_foreign_fn(ccx: @CrateContext,
                        +path: ast_map::path,
                        decl: &ast::fn_decl,
                        body: &ast::blk,
                        llwrapfn: ValueRef,
                        id: ast::node_id) {
    let _icx = ccx.insn_ctxt("foreign::build_foreign_fn");

    fn build_rust_fn(ccx: @CrateContext, +path: ast_map::path,
                     decl: &ast::fn_decl, body: &ast::blk,
                     id: ast::node_id) -> ValueRef {
        let _icx = ccx.insn_ctxt("foreign::foreign::build_rust_fn");
        let t = ty::node_id_to_type(ccx.tcx, id);
        // XXX: Bad copy.
        let ps = link::mangle_internal_name_by_path(
            ccx, vec::append_one(copy path, ast_map::path_name(
                special_idents::clownshoe_abi
            )));
        let llty = type_of_fn_from_ty(ccx, t);
        let llfndecl = decl_internal_cdecl_fn(ccx.llmod, ps, llty);
        trans_fn(ccx, path, decl, body, llfndecl, no_self, None, id, None);
        return llfndecl;
    }

    fn build_shim_fn(ccx: @CrateContext, +path: ast_map::path,
                     llrustfn: ValueRef, tys: &ShimTypes) -> ValueRef {
        /*!
         *
         * Generate the shim S:
         *
         *     void S(struct { X x; Y y; Z *z; } *args) {
         *         R(args->z, NULL, &args->x, args->y);
         *     }
         *
         * One complication is that we must adapt to the Rust
         * calling convention, which introduces indirection
         * in some cases.  To demonstrate this, I wrote one of the
         * entries above as `&args->x`, because presumably `X` is
         * one of those types that is passed by pointer in Rust.
         */

        let _icx = ccx.insn_ctxt("foreign::foreign::build_shim_fn");

        fn build_args(bcx: block, tys: &ShimTypes,
                      llargbundle: ValueRef) -> ~[ValueRef] {
            let _icx = bcx.insn_ctxt("foreign::extern::shim::build_args");
            let ccx = bcx.ccx();
            let mut llargvals = ~[];
            let mut i = 0u;
            let n = tys.fn_sig.inputs.len();
            let llretptr = load_inbounds(bcx, llargbundle, ~[0u, n]);
            llargvals.push(llretptr);
            let llenvptr = C_null(T_opaque_box_ptr(bcx.ccx()));
            llargvals.push(llenvptr);
            while i < n {
                // Get a pointer to the argument:
                let mut llargval = GEPi(bcx, llargbundle, [0u, i]);

                if !type_of::arg_is_indirect(ccx, &tys.fn_sig.inputs[i]) {
                    // If Rust would pass this by value, load the value.
                    llargval = Load(bcx, llargval);
                }

                llargvals.push(llargval);
                i += 1u;
            }
            return llargvals;
        }

        fn build_ret(_bcx: block, _tys: &ShimTypes,
                     _llargbundle: ValueRef, _llretval: ValueRef)  {
            // Nop. The return pointer in the Rust ABI function
            // is wired directly into the return slot in the shim struct
        }

        let shim_name = link::mangle_internal_name_by_path(
            ccx, vec::append_one(path, ast_map::path_name(
                special_idents::clownshoe_stack_shim
            )));
        return build_shim_fn_(ccx, shim_name, llrustfn, tys,
                           lib::llvm::CCallConv,
                           build_args, build_ret);
    }

    fn build_wrap_fn(ccx: @CrateContext, llshimfn: ValueRef,
                     llwrapfn: ValueRef, tys: &ShimTypes)
    {
        /*!
         *
         * Generate the wrapper W:
         *
         *    Z W(X x, Y y) {
         *        struct { X x; Y y; Z *z; } args = { x, y, z };
         *        call_on_c_stack_shim(S, &args);
         *    }
         */

        let _icx = ccx.insn_ctxt("foreign::foreign::build_wrap_fn");

        build_wrap_fn_(ccx, tys, llshimfn, llwrapfn,
                       ccx.upcalls.call_shim_on_rust_stack,
                       build_args, build_ret);

        fn build_args(bcx: block, tys: &ShimTypes,
                      llwrapfn: ValueRef, llargbundle: ValueRef) {
            let _icx = bcx.insn_ctxt("foreign::foreign::wrap::build_args");
            tys.fn_ty.build_wrap_args(
                bcx, tys.llsig.llret_ty,
                llwrapfn, llargbundle);
        }

        fn build_ret(bcx: block, tys: &ShimTypes,
                     llargbundle: ValueRef) {
            let _icx = bcx.insn_ctxt("foreign::foreign::wrap::build_ret");
            tys.fn_ty.build_wrap_ret(
                bcx, tys.llsig.llarg_tys, llargbundle);
        }
    }

    let tys = shim_types(ccx, id);
    // The internal Rust ABI function - runs on the Rust stack
    // XXX: Bad copy.
    let llrustfn = build_rust_fn(ccx, copy path, decl, body, id);
    // The internal shim function - runs on the Rust stack
    let llshimfn = build_shim_fn(ccx, path, llrustfn, &tys);
    // The foreign C function - runs on the C stack
    build_wrap_fn(ccx, llshimfn, llwrapfn, &tys)
}

pub fn register_foreign_fn(ccx: @CrateContext,
                           sp: span,
                           +path: ast_map::path,
                           node_id: ast::node_id,
                           attrs: &[ast::attribute])
                        -> ValueRef {
    let _icx = ccx.insn_ctxt("foreign::register_foreign_fn");
    let t = ty::node_id_to_type(ccx.tcx, node_id);
    let tys = shim_types(ccx, node_id);
    do tys.fn_ty.decl_fn |fnty| {
        register_fn_fuller(ccx, sp, /*bad*/copy path, node_id, attrs,
                           t, lib::llvm::CCallConv, fnty)
    }
}
