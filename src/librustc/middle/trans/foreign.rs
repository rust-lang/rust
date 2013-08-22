// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::{link};
use std::libc::c_uint;
use lib::llvm::{ValueRef, Attribute, CallConv};
use lib::llvm::llvm;
use lib;
use middle::trans::machine;
use middle::trans::base;
use middle::trans::base::push_ctxt;
use middle::trans::cabi;
use middle::trans::build::*;
use middle::trans::builder::noname;
use middle::trans::common::*;
use middle::trans::llrepr::LlvmRepr;
use middle::trans::type_of::*;
use middle::trans::type_of;
use middle::ty;
use middle::ty::FnSig;

use std::uint;
use std::vec;
use syntax::codemap::span;
use syntax::{ast};
use syntax::{attr, ast_map};
use syntax::parse::token::special_idents;
use syntax::abi::{RustIntrinsic, Rust, Stdcall, Fastcall,
                  Cdecl, Aapcs, C, AbiSet};
use util::ppaux::{Repr, UserString};
use middle::trans::type_::Type;

///////////////////////////////////////////////////////////////////////////
// Type definitions

struct ForeignTypes {
    /// Rust signature of the function
    fn_sig: ty::FnSig,

    /// Adapter object for handling native ABI rules (trust me, you
    /// don't want to know)
    fn_ty: cabi::FnType,

    /// LLVM types that will appear on the foreign function
    llsig: LlvmSignature,

    /// True if there is a return value (not bottom, not unit)
    ret_def: bool,
}

struct LlvmSignature {
    // LLVM versions of the types of this function's arguments.
    llarg_tys: ~[Type],

    // LLVM version of the type that this function returns.  Note that
    // this *may not be* the declared return type of the foreign
    // function, because the foreign function may opt to return via an
    // out pointer.
    llret_ty: Type,

    // True if *Rust* would use an outpointer for this function.
    sret: bool,
}


///////////////////////////////////////////////////////////////////////////
// Calls to external functions

fn llvm_calling_convention(ccx: @mut CrateContext,
                           abis: AbiSet)
                           -> Option<CallConv> {
    let arch = ccx.sess.targ_cfg.arch;
    abis.for_arch(arch).map(|abi| {
        match *abi {
            RustIntrinsic => {
                // Intrinsics are emitted by monomorphic fn
                ccx.sess.bug(fmt!("Asked to register intrinsic fn"));
            }

            Rust => {
                // FIXME(#3678) Implement linking to foreign fns with Rust ABI
                ccx.sess.unimpl(
                    fmt!("Foreign functions with Rust ABI"));
            }

            Stdcall => lib::llvm::X86StdcallCallConv,
            Fastcall => lib::llvm::X86FastcallCallConv,
            C => lib::llvm::CCallConv,

            // NOTE These API constants ought to be more specific
            Cdecl => lib::llvm::CCallConv,
            Aapcs => lib::llvm::CCallConv,
        }
    })
}


pub fn register_foreign_item_fn(ccx: @mut CrateContext,
                                abis: AbiSet,
                                path: &ast_map::path,
                                foreign_item: @ast::foreign_item) -> ValueRef {
    /*!
     * Registers a foreign function found in a library.
     * Just adds a LLVM global.
     */

    debug!("register_foreign_item_fn(abis=%s, \
            path=%s, \
            foreign_item.id=%?)",
           abis.repr(ccx.tcx),
           path.repr(ccx.tcx),
           foreign_item.id);

    let cc = match llvm_calling_convention(ccx, abis) {
        Some(cc) => cc,
        None => {
            // FIXME(#8357) We really ought to report a span here
            ccx.sess.fatal(
                fmt!("ABI `%s` has no suitable ABI \
                      for target architecture \
                      in module %s",
                     abis.user_string(ccx.tcx),
                     ast_map::path_to_str(*path,
                                          ccx.sess.intr())));
        }
    };

    // Register the function as a C extern fn
    let lname = link_name(ccx, foreign_item);
    let tys = foreign_types_for_id(ccx, foreign_item.id);

    // Create the LLVM value for the C extern fn
    let llfn_ty = lltype_for_fn_from_foreign_types(&tys);
    let llfn = base::get_extern_fn(&mut ccx.externs, ccx.llmod,
                                   lname, cc, llfn_ty);
    add_argument_attributes(&tys, llfn);

    return llfn;
}

pub fn trans_native_call(bcx: @mut Block,
                         callee_ty: ty::t,
                         llfn: ValueRef,
                         llretptr: ValueRef,
                         llargs_rust: &[ValueRef]) -> @mut Block {
    /*!
     * Prepares a call to a native function. This requires adapting
     * from the Rust argument passing rules to the native rules.
     *
     * # Parameters
     *
     * - `callee_ty`: Rust type for the function we are calling
     * - `llfn`: the function pointer we are calling
     * - `llretptr`: where to store the return value of the function
     * - `llargs_rust`: a list of the argument values, prepared
     *   as they would be if calling a Rust function
     */

    let ccx = bcx.ccx();
    let tcx = bcx.tcx();

    debug!("trans_native_call(callee_ty=%s, \
            llfn=%s, \
            llretptr=%s)",
           callee_ty.repr(tcx),
           ccx.tn.val_to_str(llfn),
           ccx.tn.val_to_str(llretptr));

    let (fn_abis, fn_sig) = match ty::get(callee_ty).sty {
        ty::ty_bare_fn(ref fn_ty) => (fn_ty.abis, fn_ty.sig.clone()),
        _ => ccx.sess.bug("trans_native_call called on non-function type")
    };
    let llsig = foreign_signature(ccx, &fn_sig);
    let ret_def = !ty::type_is_voidish(fn_sig.output);
    let fn_type = cabi::compute_abi_info(ccx,
                                         llsig.llarg_tys,
                                         llsig.llret_ty,
                                         ret_def);

    let all_arg_tys: &[cabi::LLVMType] = fn_type.arg_tys;
    let all_attributes: &[Option<Attribute>] = fn_type.attrs;

    let mut llargs_foreign = ~[];

    // If the foreign ABI expects return value by pointer, supply the
    // pointer that Rust gave us. Sometimes we have to bitcast
    // because foreign fns return slightly different (but equivalent)
    // views on the same type (e.g., i64 in place of {i32,i32}).
    let (arg_tys, attributes) = {
        if fn_type.sret {
            if all_arg_tys[0].cast {
                let llcastedretptr =
                    BitCast(bcx, llretptr, all_arg_tys[0].ty.ptr_to());
                llargs_foreign.push(llcastedretptr);
            } else {
                llargs_foreign.push(llretptr);
            }
            (all_arg_tys.tail(), all_attributes.tail())
        } else {
            (all_arg_tys, all_attributes)
        }
    };

    for (i, &llarg_rust) in llargs_rust.iter().enumerate() {
        let mut llarg_rust = llarg_rust;

        // Does Rust pass this argument by pointer?
        let rust_indirect = type_of::arg_is_indirect(ccx, fn_sig.inputs[i]);

        debug!("argument %u, llarg_rust=%s, rust_indirect=%b, arg_ty=%s",
               i,
               ccx.tn.val_to_str(llarg_rust),
               rust_indirect,
               ccx.tn.type_to_str(arg_tys[i].ty));

        // Ensure that we always have the Rust value indirectly,
        // because it makes bitcasting easier.
        if !rust_indirect {
            let scratch = base::alloca(bcx, arg_tys[i].ty, "__arg");
            Store(bcx, llarg_rust, scratch);
            llarg_rust = scratch;
        }

        debug!("llarg_rust=%s (after indirection)",
               ccx.tn.val_to_str(llarg_rust));

        // Check whether we need to do any casting
        let foreignarg_ty = arg_tys[i].ty;
        if arg_tys[i].cast {
            llarg_rust = BitCast(bcx, llarg_rust, foreignarg_ty.ptr_to());
        }

        debug!("llarg_rust=%s (after casting)",
               ccx.tn.val_to_str(llarg_rust));

        // Finally, load the value if needed for the foreign ABI
        let foreign_indirect = attributes[i].is_some();
        let llarg_foreign = if foreign_indirect {
            llarg_rust
        } else {
            Load(bcx, llarg_rust)
        };

        debug!("argument %u, llarg_foreign=%s",
               i, ccx.tn.val_to_str(llarg_foreign));

        llargs_foreign.push(llarg_foreign);
    }

    let cc = match llvm_calling_convention(ccx, fn_abis) {
        Some(cc) => cc,
        None => {
            // FIXME(#8357) We really ought to report a span here
            ccx.sess.fatal(
                fmt!("ABI string `%s` has no suitable ABI \
                      for target architecture",
                     fn_abis.user_string(ccx.tcx)));
        }
    };

    let llforeign_retval = CallWithConv(bcx, llfn, llargs_foreign, cc);

    // If the function we just called does not use an outpointer,
    // store the result into the rust outpointer. Cast the outpointer
    // type to match because some ABIs will use a different type than
    // the Rust type. e.g., a {u32,u32} struct could be returned as
    // u64.
    if ret_def && !fn_type.sret {
        let llrust_ret_ty = llsig.llret_ty;
        let llforeign_ret_ty = fn_type.ret_ty.ty;

        debug!("llretptr=%s", ccx.tn.val_to_str(llretptr));
        debug!("llforeign_retval=%s", ccx.tn.val_to_str(llforeign_retval));
        debug!("llrust_ret_ty=%s", ccx.tn.type_to_str(llrust_ret_ty));
        debug!("llforeign_ret_ty=%s", ccx.tn.type_to_str(llforeign_ret_ty));

        if llrust_ret_ty == llforeign_ret_ty {
            Store(bcx, llforeign_retval, llretptr);
        } else {
            // The actual return type is a struct, but the ABI
            // adaptation code has cast it into some scalar type.  The
            // code that follows is the only reliable way I have
            // found to do a transform like i64 -> {i32,i32}.
            // Basically we dump the data onto the stack then memcpy it.
            //
            // Other approaches I tried:
            // - Casting rust ret pointer to the foreign type and using Store
            //   is (a) unsafe if size of foreign type > size of rust type and
            //   (b) runs afoul of strict aliasing rules, yielding invalid
            //   assembly under -O (specifically, the store gets removed).
            // - Truncating foreign type to correct integral type and then
            //   bitcasting to the struct type yields invalid cast errors.
            let llscratch = base::alloca(bcx, llforeign_ret_ty, "__cast");
            Store(bcx, llforeign_retval, llscratch);
            let llscratch_i8 = BitCast(bcx, llscratch, Type::i8().ptr_to());
            let llretptr_i8 = BitCast(bcx, llretptr, Type::i8().ptr_to());
            let llrust_size = machine::llsize_of_store(ccx, llrust_ret_ty);
            let llforeign_align = machine::llalign_of_min(ccx, llforeign_ret_ty);
            let llrust_align = machine::llalign_of_min(ccx, llrust_ret_ty);
            let llalign = uint::min(llforeign_align, llrust_align);
            debug!("llrust_size=%?", llrust_size);
            base::call_memcpy(bcx, llretptr_i8, llscratch_i8,
                              C_uint(ccx, llrust_size), llalign as u32);
        }
    }

    return bcx;
}

pub fn trans_foreign_mod(ccx: @mut CrateContext,
                         foreign_mod: &ast::foreign_mod) {
    let _icx = push_ctxt("foreign::trans_foreign_mod");
    for &foreign_item in foreign_mod.items.iter() {
        let lname = link_name(ccx, foreign_item);
        ccx.item_symbols.insert(foreign_item.id, lname.to_owned());
    }
}

///////////////////////////////////////////////////////////////////////////
// Rust functions with foreign ABIs
//
// These are normal Rust functions defined with foreign ABIs.  For
// now, and perhaps forever, we translate these using a "layer of
// indirection". That is, given a Rust declaration like:
//
//     extern "C" fn foo(i: u32) -> u32 { ... }
//
// we will generate a function like:
//
//     S foo(T i) {
//         S r;
//         foo0(&r, NULL, i);
//         return r;
//     }
//
//     #[inline_always]
//     void foo0(uint32_t *r, void *env, uint32_t i) { ... }
//
// Here the (internal) `foo0` function follows the Rust ABI as normal,
// where the `foo` function follows the C ABI. We rely on LLVM to
// inline the one into the other. Of course we could just generate the
// correct code in the first place, but this is much simpler.

pub fn register_rust_fn_with_foreign_abi(ccx: @mut CrateContext,
                                         sp: span,
                                         sym: ~str,
                                         node_id: ast::NodeId)
                                         -> ValueRef {
    let _icx = push_ctxt("foreign::register_foreign_fn");

    let tys = foreign_types_for_id(ccx, node_id);
    let llfn_ty = lltype_for_fn_from_foreign_types(&tys);
    let llfn = base::register_fn_llvmty(ccx,
                                        sp,
                                        sym,
                                        node_id,
                                        lib::llvm::CCallConv,
                                        llfn_ty);
    add_argument_attributes(&tys, llfn);
    debug!("register_rust_fn_with_foreign_abi(node_id=%?, llfn_ty=%s, llfn=%s)",
           node_id, ccx.tn.type_to_str(llfn_ty), ccx.tn.val_to_str(llfn));
    llfn
}

pub fn trans_rust_fn_with_foreign_abi(ccx: @mut CrateContext,
                                      path: &ast_map::path,
                                      decl: &ast::fn_decl,
                                      body: &ast::Block,
                                      llwrapfn: ValueRef,
                                      id: ast::NodeId) {
    let _icx = push_ctxt("foreign::build_foreign_fn");
    let tys = foreign_types_for_id(ccx, id);

    unsafe { // unsafe because we call LLVM operations
        // Build up the Rust function (`foo0` above).
        let llrustfn = build_rust_fn(ccx, path, decl, body, id);

        // Build up the foreign wrapper (`foo` above).
        return build_wrap_fn(ccx, llrustfn, llwrapfn, &tys);
    }

    fn build_rust_fn(ccx: @mut CrateContext,
                     path: &ast_map::path,
                     decl: &ast::fn_decl,
                     body: &ast::Block,
                     id: ast::NodeId)
                     -> ValueRef {
        let _icx = push_ctxt("foreign::foreign::build_rust_fn");
        let tcx = ccx.tcx;
        let t = ty::node_id_to_type(tcx, id);
        let ps = link::mangle_internal_name_by_path(
            ccx, vec::append_one((*path).clone(), ast_map::path_name(
                special_idents::clownshoe_abi
            )));

        // Compute the LLVM type that the function would have if it
        // were just a normal Rust function. This will be the type of
        // the wrappee fn.
        let llty = match ty::get(t).sty {
            ty::ty_bare_fn(ref f) => {
                assert!(!f.abis.is_rust() && !f.abis.is_intrinsic());
                type_of_rust_fn(ccx, f.sig.inputs, f.sig.output)
            }
            _ => {
                ccx.sess.bug(fmt!("build_rust_fn: extern fn %s has ty %s, \
                                  expected a bare fn ty",
                                  path.repr(tcx),
                                  t.repr(tcx)));
            }
        };

        debug!("build_rust_fn: path=%s id=%? t=%s llty=%s",
               path.repr(tcx),
               id,
               t.repr(tcx),
               llty.llrepr(ccx));

        let llfndecl = base::decl_internal_cdecl_fn(ccx.llmod, ps, llty);
        base::trans_fn(ccx,
                       (*path).clone(),
                       decl,
                       body,
                       llfndecl,
                       base::no_self,
                       None,
                       id,
                       []);
        return llfndecl;
    }

    unsafe fn build_wrap_fn(ccx: @mut CrateContext,
                            llrustfn: ValueRef,
                            llwrapfn: ValueRef,
                            tys: &ForeignTypes) {
        let _icx = push_ctxt(
            "foreign::trans_rust_fn_with_foreign_abi::build_wrap_fn");
        let tcx = ccx.tcx;

        debug!("build_wrap_fn(llrustfn=%s, llwrapfn=%s)",
               ccx.tn.val_to_str(llrustfn),
               ccx.tn.val_to_str(llwrapfn));

        // Avoid all the Rust generation stuff and just generate raw
        // LLVM here.
        //
        // We want to generate code like this:
        //
        //     S foo(T i) {
        //         S r;
        //         foo0(&r, NULL, i);
        //         return r;
        //     }

        let the_block =
            "the block".to_c_str().with_ref(
                |s| llvm::LLVMAppendBasicBlockInContext(ccx.llcx, llwrapfn, s));

        let builder = ccx.builder.B;
        llvm::LLVMPositionBuilderAtEnd(builder, the_block);

        // Array for the arguments we will pass to the rust function.
        let mut llrust_args = ~[];
        let mut next_foreign_arg_counter: c_uint = 0;
        let next_foreign_arg: &fn() -> c_uint = {
            || {
                next_foreign_arg_counter += 1;
                next_foreign_arg_counter - 1
            }
        };

        // If there is an out pointer on the foreign function
        let foreign_outptr = {
            if tys.fn_ty.sret {
                Some(llvm::LLVMGetParam(llwrapfn, next_foreign_arg()))
            } else {
                None
            }
        };

        // Push Rust return pointer, using null if it will be unused.
        let rust_uses_outptr =
            type_of::return_uses_outptr(tcx, tys.fn_sig.output);
        let return_alloca: Option<ValueRef>;
        let llrust_ret_ty = tys.llsig.llret_ty;
        let llrust_retptr_ty = llrust_ret_ty.ptr_to();
        if rust_uses_outptr {
            // Rust expects to use an outpointer. If the foreign fn
            // also uses an outpointer, we can reuse it, but the types
            // may vary, so cast first to the Rust type. If the
            // foriegn fn does NOT use an outpointer, we will have to
            // alloca some scratch space on the stack.
            match foreign_outptr {
                Some(llforeign_outptr) => {
                    debug!("out pointer, foreign=%s",
                           ccx.tn.val_to_str(llforeign_outptr));
                    let llrust_retptr =
                        llvm::LLVMBuildBitCast(builder,
                                               llforeign_outptr,
                                               llrust_ret_ty.ptr_to().to_ref(),
                                               noname());
                    debug!("out pointer, foreign=%s (casted)",
                           ccx.tn.val_to_str(llrust_retptr));
                    llrust_args.push(llrust_retptr);
                    return_alloca = None;
                }

                None => {
                    let slot = {
                        "return_alloca".to_c_str().with_ref(
                            |s| llvm::LLVMBuildAlloca(builder,
                                                      llrust_ret_ty.to_ref(),
                                                      s))
                    };
                    debug!("out pointer, \
                            allocad=%s, \
                            llrust_ret_ty=%s, \
                            return_ty=%s",
                           ccx.tn.val_to_str(slot),
                           ccx.tn.type_to_str(llrust_ret_ty),
                           tys.fn_sig.output.repr(tcx));
                    llrust_args.push(slot);
                    return_alloca = Some(slot);
                }
            }
        } else {
            // Rust does not expect an outpointer. If the foreign fn
            // does use an outpointer, then we will do a store of the
            // value that the Rust fn returns.
            return_alloca = None;
        };

        // Push an (null) env pointer
        let env_pointer = base::null_env_ptr(ccx);
        debug!("env pointer=%s", ccx.tn.val_to_str(env_pointer));
        llrust_args.push(env_pointer);

        // Build up the arguments to the call to the rust function.
        // Careful to adapt for cases where the native convention uses
        // a pointer and Rust does not or vice versa.
        for i in range(0, tys.fn_sig.inputs.len()) {
            let rust_ty = tys.fn_sig.inputs[i];
            let llrust_ty = tys.llsig.llarg_tys[i];
            let foreign_index = next_foreign_arg();
            let rust_indirect = type_of::arg_is_indirect(ccx, rust_ty);
            let foreign_indirect = tys.fn_ty.attrs[foreign_index].is_some();
            let mut llforeign_arg = llvm::LLVMGetParam(llwrapfn, foreign_index);

            debug!("llforeign_arg #%u: %s",
                   i, ccx.tn.val_to_str(llforeign_arg));
            debug!("rust_indirect = %b, foreign_indirect = %b",
                   rust_indirect, foreign_indirect);

            // Ensure that the foreign argument is indirect (by
            // pointer).  It makes adapting types easier, since we can
            // always just bitcast pointers.
            if !foreign_indirect {
                let lltemp =
                    llvm::LLVMBuildAlloca(
                        builder, val_ty(llforeign_arg).to_ref(), noname());
                llvm::LLVMBuildStore(
                    builder, llforeign_arg, lltemp);
                llforeign_arg = lltemp;
            }

            // If the types in the ABI and the Rust types don't match,
            // bitcast the llforeign_arg pointer so it matches the types
            // Rust expects.
            if tys.fn_ty.arg_tys[foreign_index].cast {
                assert!(!foreign_indirect);
                llforeign_arg = llvm::LLVMBuildBitCast(
                    builder, llforeign_arg,
                    llrust_ty.ptr_to().to_ref(), noname());
            }

            let llrust_arg = if rust_indirect {
                llforeign_arg
            } else {
                llvm::LLVMBuildLoad(builder, llforeign_arg, noname())
            };

            debug!("llrust_arg #%u: %s",
                   i, ccx.tn.val_to_str(llrust_arg));
            llrust_args.push(llrust_arg);
        }

        // Perform the call itself
        let llrust_ret_val = do llrust_args.as_imm_buf |ptr, len| {
            debug!("calling llrustfn = %s", ccx.tn.val_to_str(llrustfn));
            llvm::LLVMBuildCall(builder, llrustfn, ptr,
                                len as c_uint, noname())
        };

        // Get the return value where the foreign fn expects it.
        let llforeign_ret_ty = tys.fn_ty.ret_ty.ty;
        match foreign_outptr {
            None if !tys.ret_def => {
                // Function returns `()` or `bot`, which in Rust is the LLVM
                // type "{}" but in foreign ABIs is "Void".
                llvm::LLVMBuildRetVoid(builder);
            }

            None if rust_uses_outptr => {
                // Rust uses an outpointer, but the foreign ABI does not. Load.
                let llrust_outptr = return_alloca.unwrap();
                let llforeign_outptr_casted =
                    llvm::LLVMBuildBitCast(builder,
                                           llrust_outptr,
                                           llforeign_ret_ty.ptr_to().to_ref(),
                                           noname());
                let llforeign_retval =
                    llvm::LLVMBuildLoad(builder, llforeign_outptr_casted, noname());
                llvm::LLVMBuildRet(builder, llforeign_retval);
            }

            None if llforeign_ret_ty != llrust_ret_ty => {
                // Neither ABI uses an outpointer, but the types don't
                // quite match. Must cast. Probably we should try and
                // examine the types and use a concrete llvm cast, but
                // right now we just use a temp memory location and
                // bitcast the pointer, which is the same thing the
                // old wrappers used to do.
                let lltemp =
                    llvm::LLVMBuildAlloca(
                        builder, llforeign_ret_ty.to_ref(), noname());
                let lltemp_casted =
                    llvm::LLVMBuildBitCast(builder,
                                           lltemp,
                                           llrust_ret_ty.ptr_to().to_ref(),
                                           noname());
                llvm::LLVMBuildStore(
                    builder, llrust_ret_val, lltemp_casted);
                let llforeign_retval =
                    llvm::LLVMBuildLoad(builder, lltemp, noname());
                llvm::LLVMBuildRet(builder, llforeign_retval);
            }

            None => {
                // Neither ABI uses an outpointer, and the types
                // match. Easy peasy.
                llvm::LLVMBuildRet(builder, llrust_ret_val);
            }

            Some(llforeign_outptr) if !rust_uses_outptr => {
                // Foreign ABI requires an out pointer, but Rust doesn't.
                // Store Rust return value.
                let llforeign_outptr_casted =
                    llvm::LLVMBuildBitCast(builder,
                                           llforeign_outptr,
                                           llrust_retptr_ty.to_ref(),
                                           noname());
                llvm::LLVMBuildStore(
                    builder, llrust_ret_val, llforeign_outptr_casted);
                llvm::LLVMBuildRetVoid(builder);
            }

            Some(_) => {
                // Both ABIs use outpointers. Easy peasy.
                llvm::LLVMBuildRetVoid(builder);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// General ABI Support
//
// This code is kind of a confused mess and needs to be reworked given
// the massive simplifications that have occurred.

pub fn link_name(ccx: &CrateContext, i: @ast::foreign_item) -> @str {
     match attr::first_attr_value_str_by_name(i.attrs, "link_name") {
        None => ccx.sess.str_of(i.ident),
        Some(ln) => ln,
    }
}

fn foreign_signature(ccx: &mut CrateContext, fn_sig: &ty::FnSig)
                     -> LlvmSignature {
    /*!
     * The ForeignSignature is the LLVM types of the arguments/return type
     * of a function.  Note that these LLVM types are not quite the same
     * as the LLVM types would be for a native Rust function because foreign
     * functions just plain ignore modes.  They also don't pass aggregate
     * values by pointer like we do.
     */

    let llarg_tys = fn_sig.inputs.map(|&arg| type_of(ccx, arg));
    let llret_ty = type_of::type_of(ccx, fn_sig.output);
    LlvmSignature {
        llarg_tys: llarg_tys,
        llret_ty: llret_ty,
        sret: type_of::return_uses_outptr(ccx.tcx, fn_sig.output),
    }
}

fn foreign_types_for_id(ccx: &mut CrateContext,
                        id: ast::NodeId) -> ForeignTypes {
    foreign_types_for_fn_ty(ccx, ty::node_id_to_type(ccx.tcx, id))
}

fn foreign_types_for_fn_ty(ccx: &mut CrateContext,
                           ty: ty::t) -> ForeignTypes {
    let fn_sig = match ty::get(ty).sty {
        ty::ty_bare_fn(ref fn_ty) => fn_ty.sig.clone(),
        _ => ccx.sess.bug("foreign_types_for_fn_ty called on non-function type")
    };
    let llsig = foreign_signature(ccx, &fn_sig);
    let ret_def = !ty::type_is_voidish(fn_sig.output);
    let fn_ty = cabi::compute_abi_info(ccx,
                                       llsig.llarg_tys,
                                       llsig.llret_ty,
                                       ret_def);
    debug!("foreign_types_for_fn_ty(\
           ty=%s, \
           llsig=%s -> %s, \
           fn_ty=%s -> %s, \
           ret_def=%b",
           ty.repr(ccx.tcx),
           ccx.tn.types_to_str(llsig.llarg_tys),
           ccx.tn.type_to_str(llsig.llret_ty),
           ccx.tn.types_to_str(fn_ty.arg_tys.map(|t| t.ty)),
           ccx.tn.type_to_str(fn_ty.ret_ty.ty),
           ret_def);

    ForeignTypes {
        fn_sig: fn_sig,
        llsig: llsig,
        ret_def: ret_def,
        fn_ty: fn_ty
    }
}

fn lltype_for_fn_from_foreign_types(tys: &ForeignTypes) -> Type {
    let llargument_tys: ~[Type] =
        tys.fn_ty.arg_tys.iter().map(|t| t.ty).collect();
    let llreturn_ty = tys.fn_ty.ret_ty.ty;
    Type::func(llargument_tys, &llreturn_ty)
}

pub fn lltype_for_foreign_fn(ccx: &mut CrateContext, ty: ty::t) -> Type {
    let fn_types = foreign_types_for_fn_ty(ccx, ty);
    lltype_for_fn_from_foreign_types(&fn_types)
}

fn add_argument_attributes(tys: &ForeignTypes,
                           llfn: ValueRef) {
    for (i, a) in tys.fn_ty.attrs.iter().enumerate() {
        match *a {
            Some(attr) => {
                let llarg = get_param(llfn, i);
                unsafe {
                    llvm::LLVMAddAttribute(llarg, attr as c_uint);
                }
            }
            None => ()
        }
    }
}
