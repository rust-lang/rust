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
use llvm::{ValueRef, CallConv, Linkage, get_param};
use llvm;
use middle::weak_lang_items;
use middle::trans::base::push_ctxt;
use middle::trans::base;
use middle::trans::build::*;
use middle::trans::cabi;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_::Type;
use middle::trans::type_of::*;
use middle::trans::type_of;
use middle::ty::FnSig;
use middle::ty;
use std::cmp;
use libc::c_uint;
use syntax::abi::{Cdecl, Aapcs, C, Win64, Abi};
use syntax::abi::{RustIntrinsic, Rust, RustCall, Stdcall, Fastcall, System};
use syntax::codemap::Span;
use syntax::parse::token::{InternedString, special_idents};
use syntax::parse::token;
use syntax::{ast};
use syntax::{attr, ast_map};
use util::ppaux::{Repr, UserString};

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
    llarg_tys: Vec<Type> ,

    // LLVM version of the type that this function returns.  Note that
    // this *may not be* the declared return type of the foreign
    // function, because the foreign function may opt to return via an
    // out pointer.
    llret_ty: Type,
}


///////////////////////////////////////////////////////////////////////////
// Calls to external functions

pub fn llvm_calling_convention(ccx: &CrateContext,
                               abi: Abi) -> Option<CallConv> {
    let os = ccx.sess().targ_cfg.os;
    let arch = ccx.sess().targ_cfg.arch;
    abi.for_target(os, arch).map(|abi| {
        match abi {
            RustIntrinsic => {
                // Intrinsics are emitted at the call site
                ccx.sess().bug("asked to register intrinsic fn");
            }

            Rust => {
                // FIXME(#3678) Implement linking to foreign fns with Rust ABI
                ccx.sess().unimpl("foreign functions with Rust ABI");
            }

            RustCall => {
                // FIXME(#3678) Implement linking to foreign fns with Rust ABI
                ccx.sess().unimpl("foreign functions with RustCall ABI");
            }

            // It's the ABI's job to select this, not us.
            System => ccx.sess().bug("system abi should be selected elsewhere"),

            Stdcall => llvm::X86StdcallCallConv,
            Fastcall => llvm::X86FastcallCallConv,
            C => llvm::CCallConv,
            Win64 => llvm::X86_64_Win64,

            // These API constants ought to be more specific...
            Cdecl => llvm::CCallConv,
            Aapcs => llvm::CCallConv,
        }
    })
}

pub fn llvm_linkage_by_name(name: &str) -> Option<Linkage> {
    // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
    // applicable to variable declarations and may not really make sense for
    // Rust code in the first place but whitelist them anyway and trust that
    // the user knows what s/he's doing. Who knows, unanticipated use cases
    // may pop up in the future.
    //
    // ghost, dllimport, dllexport and linkonce_odr_autohide are not supported
    // and don't have to be, LLVM treats them as no-ops.
    match name {
        "appending" => Some(llvm::AppendingLinkage),
        "available_externally" => Some(llvm::AvailableExternallyLinkage),
        "common" => Some(llvm::CommonLinkage),
        "extern_weak" => Some(llvm::ExternalWeakLinkage),
        "external" => Some(llvm::ExternalLinkage),
        "internal" => Some(llvm::InternalLinkage),
        "linkonce" => Some(llvm::LinkOnceAnyLinkage),
        "linkonce_odr" => Some(llvm::LinkOnceODRLinkage),
        "private" => Some(llvm::PrivateLinkage),
        "weak" => Some(llvm::WeakAnyLinkage),
        "weak_odr" => Some(llvm::WeakODRLinkage),
        _ => None,
    }
}

pub fn register_static(ccx: &CrateContext,
                       foreign_item: &ast::ForeignItem) -> ValueRef {
    let ty = ty::node_id_to_type(ccx.tcx(), foreign_item.id);
    let llty = type_of::type_of(ccx, ty);

    let ident = link_name(foreign_item);
    match attr::first_attr_value_str_by_name(foreign_item.attrs.as_slice(),
                                             "linkage") {
        // If this is a static with a linkage specified, then we need to handle
        // it a little specially. The typesystem prevents things like &T and
        // extern "C" fn() from being non-null, so we can't just declare a
        // static and call it a day. Some linkages (like weak) will make it such
        // that the static actually has a null value.
        Some(name) => {
            let linkage = match llvm_linkage_by_name(name.get()) {
                Some(linkage) => linkage,
                None => {
                    ccx.sess().span_fatal(foreign_item.span,
                                          "invalid linkage specified");
                }
            };
            let llty2 = match ty::get(ty).sty {
                ty::ty_ptr(ref mt) => type_of::type_of(ccx, mt.ty),
                _ => {
                    ccx.sess().span_fatal(foreign_item.span,
                                          "must have type `*T` or `*mut T`");
                }
            };
            unsafe {
                let g1 = ident.get().with_c_str(|buf| {
                    llvm::LLVMAddGlobal(ccx.llmod, llty2.to_ref(), buf)
                });
                llvm::SetLinkage(g1, linkage);

                let mut real_name = "_rust_extern_with_linkage_".to_string();
                real_name.push_str(ident.get());
                let g2 = real_name.with_c_str(|buf| {
                    llvm::LLVMAddGlobal(ccx.llmod, llty.to_ref(), buf)
                });
                llvm::SetLinkage(g2, llvm::InternalLinkage);
                llvm::LLVMSetInitializer(g2, g1);
                g2
            }
        }
        None => unsafe {
            ident.get().with_c_str(|buf| {
                llvm::LLVMAddGlobal(ccx.llmod, llty.to_ref(), buf)
            })
        }
    }
}

pub fn register_foreign_item_fn(ccx: &CrateContext, abi: Abi, fty: ty::t,
                                name: &str, span: Option<Span>) -> ValueRef {
    /*!
     * Registers a foreign function found in a library.
     * Just adds a LLVM global.
     */

    debug!("register_foreign_item_fn(abi={}, \
            ty={}, \
            name={})",
           abi.repr(ccx.tcx()),
           fty.repr(ccx.tcx()),
           name);

    let cc = match llvm_calling_convention(ccx, abi) {
        Some(cc) => cc,
        None => {
            match span {
                Some(s) => {
                    ccx.sess().span_fatal(s,
                        format!("ABI `{}` has no suitable calling convention \
                                 for target architecture",
                                abi.user_string(ccx.tcx())).as_slice())
                }
                None => {
                    ccx.sess().fatal(
                        format!("ABI `{}` has no suitable calling convention \
                                 for target architecture",
                                abi.user_string(ccx.tcx())).as_slice())
                }
            }
        }
    };

    // Register the function as a C extern fn
    let tys = foreign_types_for_fn_ty(ccx, fty);

    // Make sure the calling convention is right for variadic functions
    // (should've been caught if not in typeck)
    if tys.fn_sig.variadic {
        assert!(cc == llvm::CCallConv);
    }

    // Create the LLVM value for the C extern fn
    let llfn_ty = lltype_for_fn_from_foreign_types(ccx, &tys);

    let llfn = base::get_extern_fn(ccx,
                                   &mut *ccx.externs.borrow_mut(),
                                   name,
                                   cc,
                                   llfn_ty,
                                   fty);
    add_argument_attributes(&tys, llfn);

    llfn
}

pub fn trans_native_call<'a>(
                         bcx: &'a Block<'a>,
                         callee_ty: ty::t,
                         llfn: ValueRef,
                         llretptr: ValueRef,
                         llargs_rust: &[ValueRef],
                         passed_arg_tys: Vec<ty::t> )
                         -> &'a Block<'a> {
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
     * - `passed_arg_tys`: Rust type for the arguments. Normally we
     *   can derive these from callee_ty but in the case of variadic
     *   functions passed_arg_tys will include the Rust type of all
     *   the arguments including the ones not specified in the fn's signature.
     */

    let ccx = bcx.ccx();
    let tcx = bcx.tcx();

    debug!("trans_native_call(callee_ty={}, \
            llfn={}, \
            llretptr={})",
           callee_ty.repr(tcx),
           ccx.tn.val_to_string(llfn),
           ccx.tn.val_to_string(llretptr));

    let (fn_abi, fn_sig) = match ty::get(callee_ty).sty {
        ty::ty_bare_fn(ref fn_ty) => (fn_ty.abi, fn_ty.sig.clone()),
        _ => ccx.sess().bug("trans_native_call called on non-function type")
    };
    let llsig = foreign_signature(ccx, &fn_sig, passed_arg_tys.as_slice());
    let ret_def = !return_type_is_void(bcx.ccx(), fn_sig.output);
    let fn_type = cabi::compute_abi_info(ccx,
                                         llsig.llarg_tys.as_slice(),
                                         llsig.llret_ty,
                                         ret_def);

    let arg_tys: &[cabi::ArgType] = fn_type.arg_tys.as_slice();

    let mut llargs_foreign = Vec::new();

    // If the foreign ABI expects return value by pointer, supply the
    // pointer that Rust gave us. Sometimes we have to bitcast
    // because foreign fns return slightly different (but equivalent)
    // views on the same type (e.g., i64 in place of {i32,i32}).
    if fn_type.ret_ty.is_indirect() {
        match fn_type.ret_ty.cast {
            Some(ty) => {
                let llcastedretptr =
                    BitCast(bcx, llretptr, ty.ptr_to());
                llargs_foreign.push(llcastedretptr);
            }
            None => {
                llargs_foreign.push(llretptr);
            }
        }
    }

    for (i, &llarg_rust) in llargs_rust.iter().enumerate() {
        let mut llarg_rust = llarg_rust;

        if arg_tys[i].is_ignore() {
            continue;
        }

        // Does Rust pass this argument by pointer?
        let rust_indirect = type_of::arg_is_indirect(ccx,
                                                     *passed_arg_tys.get(i));

        debug!("argument {}, llarg_rust={}, rust_indirect={}, arg_ty={}",
               i,
               ccx.tn.val_to_string(llarg_rust),
               rust_indirect,
               ccx.tn.type_to_string(arg_tys[i].ty));

        // Ensure that we always have the Rust value indirectly,
        // because it makes bitcasting easier.
        if !rust_indirect {
            let scratch =
                base::alloca(bcx,
                             type_of::type_of(ccx, *passed_arg_tys.get(i)),
                             "__arg");
            base::store_ty(bcx, llarg_rust, scratch, *passed_arg_tys.get(i));
            llarg_rust = scratch;
        }

        debug!("llarg_rust={} (after indirection)",
               ccx.tn.val_to_string(llarg_rust));

        // Check whether we need to do any casting
        match arg_tys[i].cast {
            Some(ty) => llarg_rust = BitCast(bcx, llarg_rust, ty.ptr_to()),
            None => ()
        }

        debug!("llarg_rust={} (after casting)",
               ccx.tn.val_to_string(llarg_rust));

        // Finally, load the value if needed for the foreign ABI
        let foreign_indirect = arg_tys[i].is_indirect();
        let llarg_foreign = if foreign_indirect {
            llarg_rust
        } else {
            if ty::type_is_bool(*passed_arg_tys.get(i)) {
                let val = LoadRangeAssert(bcx, llarg_rust, 0, 2, llvm::False);
                Trunc(bcx, val, Type::i1(bcx.ccx()))
            } else {
                Load(bcx, llarg_rust)
            }
        };

        debug!("argument {}, llarg_foreign={}",
               i, ccx.tn.val_to_string(llarg_foreign));

        // fill padding with undef value
        match arg_tys[i].pad {
            Some(ty) => llargs_foreign.push(C_undef(ty)),
            None => ()
        }
        llargs_foreign.push(llarg_foreign);
    }

    let cc = match llvm_calling_convention(ccx, fn_abi) {
        Some(cc) => cc,
        None => {
            // FIXME(#8357) We really ought to report a span here
            ccx.sess().fatal(
                format!("ABI string `{}` has no suitable ABI \
                         for target architecture",
                         fn_abi.user_string(ccx.tcx())).as_slice());
        }
    };

    // A function pointer is called without the declaration available, so we have to apply
    // any attributes with ABI implications directly to the call instruction.
    let mut attrs = Vec::new();

    // Add attributes that are always applicable, independent of the concrete foreign ABI
    if fn_type.ret_ty.is_indirect() {
        // The outptr can be noalias and nocapture because it's entirely
        // invisible to the program. We can also mark it as nonnull
        attrs.push((1, llvm::NoAliasAttribute as u64));
        attrs.push((1, llvm::NoCaptureAttribute as u64));
        attrs.push((1, llvm::NonNullAttribute as u64));
    };

    // Add attributes that depend on the concrete foreign ABI
    let mut arg_idx = if fn_type.ret_ty.is_indirect() { 1 } else { 0 };
    match fn_type.ret_ty.attr {
        Some(attr) => attrs.push((arg_idx, attr as u64)),
        _ => ()
    }

    arg_idx += 1;
    for arg_ty in fn_type.arg_tys.iter() {
        if arg_ty.is_ignore() {
            continue;
        }
        // skip padding
        if arg_ty.pad.is_some() { arg_idx += 1; }

        match arg_ty.attr {
            Some(attr) => attrs.push((arg_idx, attr as u64)),
            _ => {}
        }

        arg_idx += 1;
    }

    let llforeign_retval = CallWithConv(bcx,
                                        llfn,
                                        llargs_foreign.as_slice(),
                                        cc,
                                        attrs.as_slice());

    // If the function we just called does not use an outpointer,
    // store the result into the rust outpointer. Cast the outpointer
    // type to match because some ABIs will use a different type than
    // the Rust type. e.g., a {u32,u32} struct could be returned as
    // u64.
    if ret_def && !fn_type.ret_ty.is_indirect() {
        let llrust_ret_ty = llsig.llret_ty;
        let llforeign_ret_ty = match fn_type.ret_ty.cast {
            Some(ty) => ty,
            None => fn_type.ret_ty.ty
        };

        debug!("llretptr={}", ccx.tn.val_to_string(llretptr));
        debug!("llforeign_retval={}", ccx.tn.val_to_string(llforeign_retval));
        debug!("llrust_ret_ty={}", ccx.tn.type_to_string(llrust_ret_ty));
        debug!("llforeign_ret_ty={}", ccx.tn.type_to_string(llforeign_ret_ty));

        if llrust_ret_ty == llforeign_ret_ty {
            base::store_ty(bcx, llforeign_retval, llretptr, fn_sig.output)
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
            let llscratch_i8 = BitCast(bcx, llscratch, Type::i8(ccx).ptr_to());
            let llretptr_i8 = BitCast(bcx, llretptr, Type::i8(ccx).ptr_to());
            let llrust_size = machine::llsize_of_store(ccx, llrust_ret_ty);
            let llforeign_align = machine::llalign_of_min(ccx, llforeign_ret_ty);
            let llrust_align = machine::llalign_of_min(ccx, llrust_ret_ty);
            let llalign = cmp::min(llforeign_align, llrust_align);
            debug!("llrust_size={:?}", llrust_size);
            base::call_memcpy(bcx, llretptr_i8, llscratch_i8,
                              C_uint(ccx, llrust_size as uint), llalign as u32);
        }
    }

    return bcx;
}

pub fn trans_foreign_mod(ccx: &CrateContext, foreign_mod: &ast::ForeignMod) {
    let _icx = push_ctxt("foreign::trans_foreign_mod");
    for foreign_item in foreign_mod.items.iter() {
        let lname = link_name(&**foreign_item);

        match foreign_item.node {
            ast::ForeignItemFn(..) => {
                match foreign_mod.abi {
                    Rust | RustIntrinsic => {}
                    abi => {
                        let ty = ty::node_id_to_type(ccx.tcx(), foreign_item.id);
                        register_foreign_item_fn(ccx, abi, ty,
                                                 lname.get().as_slice(),
                                                 Some(foreign_item.span));
                    }
                }
            }
            _ => {}
        }

        ccx.item_symbols.borrow_mut().insert(foreign_item.id,
                                             lname.get().to_string());
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

pub fn register_rust_fn_with_foreign_abi(ccx: &CrateContext,
                                         sp: Span,
                                         sym: String,
                                         node_id: ast::NodeId)
                                         -> ValueRef {
    let _icx = push_ctxt("foreign::register_foreign_fn");

    let tys = foreign_types_for_id(ccx, node_id);
    let llfn_ty = lltype_for_fn_from_foreign_types(ccx, &tys);
    let t = ty::node_id_to_type(ccx.tcx(), node_id);
    let cconv = match ty::get(t).sty {
        ty::ty_bare_fn(ref fn_ty) => {
            let c = llvm_calling_convention(ccx, fn_ty.abi);
            c.unwrap_or(llvm::CCallConv)
        }
        _ => fail!("expected bare fn in register_rust_fn_with_foreign_abi")
    };
    let llfn = base::register_fn_llvmty(ccx, sp, sym, node_id, cconv, llfn_ty);
    add_argument_attributes(&tys, llfn);
    debug!("register_rust_fn_with_foreign_abi(node_id={:?}, llfn_ty={}, llfn={})",
           node_id, ccx.tn.type_to_string(llfn_ty), ccx.tn.val_to_string(llfn));
    llfn
}

pub fn trans_rust_fn_with_foreign_abi(ccx: &CrateContext,
                                      decl: &ast::FnDecl,
                                      body: &ast::Block,
                                      attrs: &[ast::Attribute],
                                      llwrapfn: ValueRef,
                                      id: ast::NodeId) {
    let _icx = push_ctxt("foreign::build_foreign_fn");
    let tys = foreign_types_for_id(ccx, id);

    unsafe { // unsafe because we call LLVM operations
        // Build up the Rust function (`foo0` above).
        let llrustfn = build_rust_fn(ccx, decl, body, attrs, id);

        // Build up the foreign wrapper (`foo` above).
        return build_wrap_fn(ccx, llrustfn, llwrapfn, &tys, id);
    }

    fn build_rust_fn(ccx: &CrateContext,
                     decl: &ast::FnDecl,
                     body: &ast::Block,
                     attrs: &[ast::Attribute],
                     id: ast::NodeId)
                     -> ValueRef {
        let _icx = push_ctxt("foreign::foreign::build_rust_fn");
        let tcx = ccx.tcx();
        let t = ty::node_id_to_type(tcx, id);

        let ps = ccx.tcx.map.with_path(id, |path| {
            let abi = Some(ast_map::PathName(special_idents::clownshoe_abi.name));
            link::mangle(path.chain(abi.move_iter()), None)
        });

        // Compute the type that the function would have if it were just a
        // normal Rust function. This will be the type of the wrappee fn.
        match ty::get(t).sty {
            ty::ty_bare_fn(ref f) => {
                assert!(f.abi != Rust && f.abi != RustIntrinsic);
            }
            _ => {
                ccx.sess().bug(format!("build_rust_fn: extern fn {} has ty {}, \
                                        expected a bare fn ty",
                                       ccx.tcx.map.path_to_string(id),
                                       t.repr(tcx)).as_slice());
            }
        };

        debug!("build_rust_fn: path={} id={} t={}",
               ccx.tcx.map.path_to_string(id),
               id, t.repr(tcx));

        let llfn = base::decl_internal_rust_fn(ccx, t, ps.as_slice());
        base::set_llvm_fn_attrs(attrs, llfn);
        base::trans_fn(ccx, decl, body, llfn, &param_substs::empty(), id, []);
        llfn
    }

    unsafe fn build_wrap_fn(ccx: &CrateContext,
                            llrustfn: ValueRef,
                            llwrapfn: ValueRef,
                            tys: &ForeignTypes,
                            id: ast::NodeId) {
        let _icx = push_ctxt(
            "foreign::trans_rust_fn_with_foreign_abi::build_wrap_fn");
        let tcx = ccx.tcx();

        let t = ty::node_id_to_type(tcx, id);

        debug!("build_wrap_fn(llrustfn={}, llwrapfn={}, t={})",
               ccx.tn.val_to_string(llrustfn),
               ccx.tn.val_to_string(llwrapfn),
               t.repr(ccx.tcx()));

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
            "the block".with_c_str(
                |s| llvm::LLVMAppendBasicBlockInContext(ccx.llcx, llwrapfn, s));

        let builder = ccx.builder();
        builder.position_at_end(the_block);

        // Array for the arguments we will pass to the rust function.
        let mut llrust_args = Vec::new();
        let mut next_foreign_arg_counter: c_uint = 0;
        let next_foreign_arg: |pad: bool| -> c_uint = |pad: bool| {
            next_foreign_arg_counter += if pad {
                2
            } else {
                1
            };
            next_foreign_arg_counter - 1
        };

        // If there is an out pointer on the foreign function
        let foreign_outptr = {
            if tys.fn_ty.ret_ty.is_indirect() {
                Some(get_param(llwrapfn, next_foreign_arg(false)))
            } else {
                None
            }
        };

        // Push Rust return pointer, using null if it will be unused.
        let rust_uses_outptr =
            type_of::return_uses_outptr(ccx, tys.fn_sig.output);
        let return_alloca: Option<ValueRef>;
        let llrust_ret_ty = tys.llsig.llret_ty;
        let llrust_retptr_ty = llrust_ret_ty.ptr_to();
        if rust_uses_outptr {
            // Rust expects to use an outpointer. If the foreign fn
            // also uses an outpointer, we can reuse it, but the types
            // may vary, so cast first to the Rust type. If the
            // foreign fn does NOT use an outpointer, we will have to
            // alloca some scratch space on the stack.
            match foreign_outptr {
                Some(llforeign_outptr) => {
                    debug!("out pointer, foreign={}",
                           ccx.tn.val_to_string(llforeign_outptr));
                    let llrust_retptr =
                        builder.bitcast(llforeign_outptr, llrust_ret_ty.ptr_to());
                    debug!("out pointer, foreign={} (casted)",
                           ccx.tn.val_to_string(llrust_retptr));
                    llrust_args.push(llrust_retptr);
                    return_alloca = None;
                }

                None => {
                    let slot = builder.alloca(llrust_ret_ty, "return_alloca");
                    debug!("out pointer, \
                            allocad={}, \
                            llrust_ret_ty={}, \
                            return_ty={}",
                           ccx.tn.val_to_string(slot),
                           ccx.tn.type_to_string(llrust_ret_ty),
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

        // Build up the arguments to the call to the rust function.
        // Careful to adapt for cases where the native convention uses
        // a pointer and Rust does not or vice versa.
        for i in range(0, tys.fn_sig.inputs.len()) {
            let rust_ty = *tys.fn_sig.inputs.get(i);
            let llrust_ty = *tys.llsig.llarg_tys.get(i);
            let rust_indirect = type_of::arg_is_indirect(ccx, rust_ty);
            let llforeign_arg_ty = *tys.fn_ty.arg_tys.get(i);
            let foreign_indirect = llforeign_arg_ty.is_indirect();

            // skip padding
            let foreign_index = next_foreign_arg(llforeign_arg_ty.pad.is_some());
            let mut llforeign_arg = get_param(llwrapfn, foreign_index);

            debug!("llforeign_arg {}{}: {}", "#",
                   i, ccx.tn.val_to_string(llforeign_arg));
            debug!("rust_indirect = {}, foreign_indirect = {}",
                   rust_indirect, foreign_indirect);

            // Ensure that the foreign argument is indirect (by
            // pointer).  It makes adapting types easier, since we can
            // always just bitcast pointers.
            if !foreign_indirect {
                llforeign_arg = if ty::type_is_bool(rust_ty) {
                    let lltemp = builder.alloca(Type::bool(ccx), "");
                    builder.store(builder.zext(llforeign_arg, Type::bool(ccx)), lltemp);
                    lltemp
                } else {
                    let lltemp = builder.alloca(val_ty(llforeign_arg), "");
                    builder.store(llforeign_arg, lltemp);
                    lltemp
                }
            }

            // If the types in the ABI and the Rust types don't match,
            // bitcast the llforeign_arg pointer so it matches the types
            // Rust expects.
            if llforeign_arg_ty.cast.is_some() {
                assert!(!foreign_indirect);
                llforeign_arg = builder.bitcast(llforeign_arg, llrust_ty.ptr_to());
            }

            let llrust_arg = if rust_indirect {
                llforeign_arg
            } else {
                if ty::type_is_bool(rust_ty) {
                    let tmp = builder.load_range_assert(llforeign_arg, 0, 2, llvm::False);
                    builder.trunc(tmp, Type::i1(ccx))
                } else {
                    builder.load(llforeign_arg)
                }
            };

            debug!("llrust_arg {}{}: {}", "#",
                   i, ccx.tn.val_to_string(llrust_arg));
            llrust_args.push(llrust_arg);
        }

        // Perform the call itself
        debug!("calling llrustfn = {}, t = {}", ccx.tn.val_to_string(llrustfn), t.repr(ccx.tcx()));
        let attributes = base::get_fn_llvm_attributes(ccx, t);
        let llrust_ret_val = builder.call(llrustfn, llrust_args.as_slice(), attributes.as_slice());

        // Get the return value where the foreign fn expects it.
        let llforeign_ret_ty = match tys.fn_ty.ret_ty.cast {
            Some(ty) => ty,
            None => tys.fn_ty.ret_ty.ty
        };
        match foreign_outptr {
            None if !tys.ret_def => {
                // Function returns `()` or `bot`, which in Rust is the LLVM
                // type "{}" but in foreign ABIs is "Void".
                builder.ret_void();
            }

            None if rust_uses_outptr => {
                // Rust uses an outpointer, but the foreign ABI does not. Load.
                let llrust_outptr = return_alloca.unwrap();
                let llforeign_outptr_casted =
                    builder.bitcast(llrust_outptr, llforeign_ret_ty.ptr_to());
                let llforeign_retval = builder.load(llforeign_outptr_casted);
                builder.ret(llforeign_retval);
            }

            None if llforeign_ret_ty != llrust_ret_ty => {
                // Neither ABI uses an outpointer, but the types don't
                // quite match. Must cast. Probably we should try and
                // examine the types and use a concrete llvm cast, but
                // right now we just use a temp memory location and
                // bitcast the pointer, which is the same thing the
                // old wrappers used to do.
                let lltemp = builder.alloca(llforeign_ret_ty, "");
                let lltemp_casted = builder.bitcast(lltemp, llrust_ret_ty.ptr_to());
                builder.store(llrust_ret_val, lltemp_casted);
                let llforeign_retval = builder.load(lltemp);
                builder.ret(llforeign_retval);
            }

            None => {
                // Neither ABI uses an outpointer, and the types
                // match. Easy peasy.
                builder.ret(llrust_ret_val);
            }

            Some(llforeign_outptr) if !rust_uses_outptr => {
                // Foreign ABI requires an out pointer, but Rust doesn't.
                // Store Rust return value.
                let llforeign_outptr_casted =
                    builder.bitcast(llforeign_outptr, llrust_retptr_ty);
                builder.store(llrust_ret_val, llforeign_outptr_casted);
                builder.ret_void();
            }

            Some(_) => {
                // Both ABIs use outpointers. Easy peasy.
                builder.ret_void();
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// General ABI Support
//
// This code is kind of a confused mess and needs to be reworked given
// the massive simplifications that have occurred.

pub fn link_name(i: &ast::ForeignItem) -> InternedString {
    match attr::first_attr_value_str_by_name(i.attrs.as_slice(), "link_name") {
        Some(ln) => ln.clone(),
        None => match weak_lang_items::link_name(i.attrs.as_slice()) {
            Some(name) => name,
            None => token::get_ident(i.ident),
        }
    }
}

fn foreign_signature(ccx: &CrateContext, fn_sig: &ty::FnSig, arg_tys: &[ty::t])
                     -> LlvmSignature {
    /*!
     * The ForeignSignature is the LLVM types of the arguments/return type
     * of a function.  Note that these LLVM types are not quite the same
     * as the LLVM types would be for a native Rust function because foreign
     * functions just plain ignore modes.  They also don't pass aggregate
     * values by pointer like we do.
     */

    let llarg_tys = arg_tys.iter().map(|&arg| arg_type_of(ccx, arg)).collect();
    let llret_ty = type_of::arg_type_of(ccx, fn_sig.output);
    LlvmSignature {
        llarg_tys: llarg_tys,
        llret_ty: llret_ty
    }
}

fn foreign_types_for_id(ccx: &CrateContext,
                        id: ast::NodeId) -> ForeignTypes {
    foreign_types_for_fn_ty(ccx, ty::node_id_to_type(ccx.tcx(), id))
}

fn foreign_types_for_fn_ty(ccx: &CrateContext,
                           ty: ty::t) -> ForeignTypes {
    let fn_sig = match ty::get(ty).sty {
        ty::ty_bare_fn(ref fn_ty) => fn_ty.sig.clone(),
        _ => ccx.sess().bug("foreign_types_for_fn_ty called on non-function type")
    };
    let llsig = foreign_signature(ccx, &fn_sig, fn_sig.inputs.as_slice());
    let ret_def = !return_type_is_void(ccx, fn_sig.output);
    let fn_ty = cabi::compute_abi_info(ccx,
                                       llsig.llarg_tys.as_slice(),
                                       llsig.llret_ty,
                                       ret_def);
    debug!("foreign_types_for_fn_ty(\
           ty={}, \
           llsig={} -> {}, \
           fn_ty={} -> {}, \
           ret_def={}",
           ty.repr(ccx.tcx()),
           ccx.tn.types_to_str(llsig.llarg_tys.as_slice()),
           ccx.tn.type_to_string(llsig.llret_ty),
           ccx.tn.types_to_str(fn_ty.arg_tys.iter().map(|t| t.ty).collect::<Vec<_>>().as_slice()),
           ccx.tn.type_to_string(fn_ty.ret_ty.ty),
           ret_def);

    ForeignTypes {
        fn_sig: fn_sig,
        llsig: llsig,
        ret_def: ret_def,
        fn_ty: fn_ty
    }
}

fn lltype_for_fn_from_foreign_types(ccx: &CrateContext, tys: &ForeignTypes) -> Type {
    let mut llargument_tys = Vec::new();

    let ret_ty = tys.fn_ty.ret_ty;
    let llreturn_ty = if ret_ty.is_indirect() {
        llargument_tys.push(ret_ty.ty.ptr_to());
        Type::void(ccx)
    } else {
        match ret_ty.cast {
            Some(ty) => ty,
            None => ret_ty.ty
        }
    };

    for &arg_ty in tys.fn_ty.arg_tys.iter() {
        if arg_ty.is_ignore() {
            continue;
        }
        // add padding
        match arg_ty.pad {
            Some(ty) => llargument_tys.push(ty),
            None => ()
        }

        let llarg_ty = if arg_ty.is_indirect() {
            arg_ty.ty.ptr_to()
        } else {
            match arg_ty.cast {
                Some(ty) => ty,
                None => arg_ty.ty
            }
        };

        llargument_tys.push(llarg_ty);
    }

    if tys.fn_sig.variadic {
        Type::variadic_func(llargument_tys.as_slice(), &llreturn_ty)
    } else {
        Type::func(llargument_tys.as_slice(), &llreturn_ty)
    }
}

pub fn lltype_for_foreign_fn(ccx: &CrateContext, ty: ty::t) -> Type {
    lltype_for_fn_from_foreign_types(ccx, &foreign_types_for_fn_ty(ccx, ty))
}

fn add_argument_attributes(tys: &ForeignTypes,
                           llfn: ValueRef) {
    let mut i = if tys.fn_ty.ret_ty.is_indirect() {
        1i
    } else {
        0i
    };

    match tys.fn_ty.ret_ty.attr {
        Some(attr) => unsafe {
            llvm::LLVMAddFunctionAttribute(llfn, i as c_uint, attr as u64);
        },
        None => {}
    }

    i += 1;

    for &arg_ty in tys.fn_ty.arg_tys.iter() {
        if arg_ty.is_ignore() {
            continue;
        }
        // skip padding
        if arg_ty.pad.is_some() { i += 1; }

        match arg_ty.attr {
            Some(attr) => unsafe {
                llvm::LLVMAddFunctionAttribute(llfn, i as c_uint, attr as u64);
            },
            None => ()
        }

        i += 1;
    }
}
