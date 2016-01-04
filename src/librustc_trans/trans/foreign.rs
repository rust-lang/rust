// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::{abi, link};
use llvm::{ValueRef, CallConv, get_param};
use llvm;
use middle::weak_lang_items;
use trans::attributes;
use trans::base::{llvm_linkage_by_name, push_ctxt};
use trans::base;
use trans::build::*;
use trans::cabi;
use trans::common::*;
use trans::debuginfo::DebugLoc;
use trans::declare;
use trans::expr;
use trans::machine;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of::*;
use trans::type_of;
use middle::infer;
use middle::ty::{self, Ty};
use middle::subst::Substs;

use std::cmp;
use std::iter::once;
use libc::c_uint;
use syntax::abi::{Cdecl, Aapcs, C, Win64, Abi};
use syntax::abi::{PlatformIntrinsic, RustIntrinsic, Rust, RustCall, Stdcall, Fastcall, System};
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token::{InternedString, special_idents};
use syntax::ast;

use rustc_front::print::pprust;
use rustc_front::hir;

///////////////////////////////////////////////////////////////////////////
// Type definitions

struct ForeignTypes<'tcx> {
    /// Rust signature of the function
    fn_sig: ty::FnSig<'tcx>,

    /// Adapter object for handling native ABI rules (trust me, you
    /// don't want to know)
    fn_ty: cabi::FnType,

    /// LLVM types that will appear on the foreign function
    llsig: LlvmSignature,
}

struct LlvmSignature {
    // LLVM versions of the types of this function's arguments.
    llarg_tys: Vec<Type> ,

    // LLVM version of the type that this function returns.  Note that
    // this *may not be* the declared return type of the foreign
    // function, because the foreign function may opt to return via an
    // out pointer.
    llret_ty: Type,

    /// True if there is a return value (not bottom, not unit)
    ret_def: bool,
}


///////////////////////////////////////////////////////////////////////////
// Calls to external functions

pub fn llvm_calling_convention(ccx: &CrateContext,
                               abi: Abi) -> CallConv {
    match ccx.sess().target.target.adjust_abi(abi) {
        RustIntrinsic => {
            // Intrinsics are emitted at the call site
            ccx.sess().bug("asked to register intrinsic fn");
        }
        PlatformIntrinsic => {
            // Intrinsics are emitted at the call site
            ccx.sess().bug("asked to register platform intrinsic fn");
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
}

pub fn register_static(ccx: &CrateContext,
                       foreign_item: &hir::ForeignItem) -> ValueRef {
    let ty = ccx.tcx().node_id_to_type(foreign_item.id);
    let llty = type_of::type_of(ccx, ty);

    let ident = link_name(foreign_item);
    match attr::first_attr_value_str_by_name(&foreign_item.attrs,
                                             "linkage") {
        // If this is a static with a linkage specified, then we need to handle
        // it a little specially. The typesystem prevents things like &T and
        // extern "C" fn() from being non-null, so we can't just declare a
        // static and call it a day. Some linkages (like weak) will make it such
        // that the static actually has a null value.
        Some(name) => {
            let linkage = match llvm_linkage_by_name(&name) {
                Some(linkage) => linkage,
                None => {
                    ccx.sess().span_fatal(foreign_item.span,
                                          "invalid linkage specified");
                }
            };
            let llty2 = match ty.sty {
                ty::TyRawPtr(ref mt) => type_of::type_of(ccx, mt.ty),
                _ => {
                    ccx.sess().span_fatal(foreign_item.span,
                                          "must have type `*T` or `*mut T`");
                }
            };
            unsafe {
                // Declare a symbol `foo` with the desired linkage.
                let g1 = declare::declare_global(ccx, &ident[..], llty2);
                llvm::SetLinkage(g1, linkage);

                // Declare an internal global `extern_with_linkage_foo` which
                // is initialized with the address of `foo`.  If `foo` is
                // discarded during linking (for example, if `foo` has weak
                // linkage and there are no definitions), then
                // `extern_with_linkage_foo` will instead be initialized to
                // zero.
                let mut real_name = "_rust_extern_with_linkage_".to_string();
                real_name.push_str(&ident);
                let g2 = declare::define_global(ccx, &real_name[..], llty).unwrap_or_else(||{
                    ccx.sess().span_fatal(foreign_item.span,
                                          &format!("symbol `{}` is already defined", ident))
                });
                llvm::SetLinkage(g2, llvm::InternalLinkage);
                llvm::LLVMSetInitializer(g2, g1);
                g2
            }
        }
        None => // Generate an external declaration.
            declare::declare_global(ccx, &ident[..], llty),
    }
}

// only use this for foreign function ABIs and glue, use `get_extern_rust_fn` for Rust functions
pub fn get_extern_fn(ccx: &CrateContext,
                     externs: &mut ExternMap,
                     name: &str,
                     cc: llvm::CallConv,
                     ty: Type,
                     output: Ty)
                     -> ValueRef {
    match externs.get(name) {
        Some(n) => return *n,
        None => {}
    }
    let f = declare::declare_fn(ccx, name, cc, ty, ty::FnConverging(output));
    externs.insert(name.to_string(), f);
    f
}

/// Registers a foreign function found in a library. Just adds a LLVM global.
pub fn register_foreign_item_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                          abi: Abi, fty: Ty<'tcx>,
                                          name: &str,
                                          attrs: &[ast::Attribute])-> ValueRef {
    debug!("register_foreign_item_fn(abi={:?}, \
            ty={:?}, \
            name={})",
           abi,
           fty,
           name);

    let cc = llvm_calling_convention(ccx, abi);

    // Register the function as a C extern fn
    let tys = foreign_types_for_fn_ty(ccx, fty);

    // Make sure the calling convention is right for variadic functions
    // (should've been caught if not in typeck)
    if tys.fn_sig.variadic {
        assert!(cc == llvm::CCallConv);
    }

    // Create the LLVM value for the C extern fn
    let llfn_ty = lltype_for_fn_from_foreign_types(ccx, &tys);

    let llfn = get_extern_fn(ccx, &mut *ccx.externs().borrow_mut(), name, cc, llfn_ty, fty);
    attributes::unwind(llfn, false);
    add_argument_attributes(&tys, llfn);
    attributes::from_fn_attrs(ccx, attrs, llfn);
    llfn
}

/// Prepares a call to a native function. This requires adapting
/// from the Rust argument passing rules to the native rules.
///
/// # Parameters
///
/// - `callee_ty`: Rust type for the function we are calling
/// - `llfn`: the function pointer we are calling
/// - `llretptr`: where to store the return value of the function
/// - `llargs_rust`: a list of the argument values, prepared
///   as they would be if calling a Rust function
/// - `passed_arg_tys`: Rust type for the arguments. Normally we
///   can derive these from callee_ty but in the case of variadic
///   functions passed_arg_tys will include the Rust type of all
///   the arguments including the ones not specified in the fn's signature.
pub fn trans_native_call<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     callee_ty: Ty<'tcx>,
                                     llfn: ValueRef,
                                     llretptr: ValueRef,
                                     llargs_rust: &[ValueRef],
                                     passed_arg_tys: Vec<Ty<'tcx>>,
                                     call_debug_loc: DebugLoc)
                                     -> Block<'blk, 'tcx>
{
    let ccx = bcx.ccx();

    debug!("trans_native_call(callee_ty={:?}, \
            llfn={}, \
            llretptr={})",
           callee_ty,
           ccx.tn().val_to_string(llfn),
           ccx.tn().val_to_string(llretptr));

    let (fn_abi, fn_sig) = match callee_ty.sty {
        ty::TyBareFn(_, ref fn_ty) => (fn_ty.abi, &fn_ty.sig),
        _ => ccx.sess().bug("trans_native_call called on non-function type")
    };
    let fn_sig = ccx.tcx().erase_late_bound_regions(fn_sig);
    let fn_sig = infer::normalize_associated_type(ccx.tcx(), &fn_sig);
    let llsig = foreign_signature(ccx, &fn_sig, &passed_arg_tys[..]);
    let fn_type = cabi::compute_abi_info(ccx,
                                         &llsig.llarg_tys,
                                         llsig.llret_ty,
                                         llsig.ret_def);

    let arg_tys: &[cabi::ArgType] = &fn_type.arg_tys;

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

    let mut offset = 0;
    for (i, arg_ty) in arg_tys.iter().enumerate() {
        let mut llarg_rust = llargs_rust[i + offset];

        if arg_ty.is_ignore() {
            continue;
        }

        // Does Rust pass this argument by pointer?
        let rust_indirect = type_of::arg_is_indirect(ccx, passed_arg_tys[i]);

        debug!("argument {}, llarg_rust={}, rust_indirect={}, arg_ty={}",
               i,
               ccx.tn().val_to_string(llarg_rust),
               rust_indirect,
               ccx.tn().type_to_string(arg_ty.ty));

        // Ensure that we always have the Rust value indirectly,
        // because it makes bitcasting easier.
        if !rust_indirect {
            let scratch = base::alloc_ty(bcx, passed_arg_tys[i], "__arg");
            if type_is_fat_ptr(ccx.tcx(), passed_arg_tys[i]) {
                Store(bcx, llargs_rust[i + offset], expr::get_dataptr(bcx, scratch));
                Store(bcx, llargs_rust[i + offset + 1], expr::get_meta(bcx, scratch));
                offset += 1;
            } else {
                base::store_ty(bcx, llarg_rust, scratch, passed_arg_tys[i]);
            }
            llarg_rust = scratch;
        }

        debug!("llarg_rust={} (after indirection)",
               ccx.tn().val_to_string(llarg_rust));

        // Check whether we need to do any casting
        match arg_ty.cast {
            Some(ty) => llarg_rust = BitCast(bcx, llarg_rust, ty.ptr_to()),
            None => ()
        }

        debug!("llarg_rust={} (after casting)",
               ccx.tn().val_to_string(llarg_rust));

        // Finally, load the value if needed for the foreign ABI
        let foreign_indirect = arg_ty.is_indirect();
        let llarg_foreign = if foreign_indirect {
            llarg_rust
        } else {
            if passed_arg_tys[i].is_bool() {
                let val = LoadRangeAssert(bcx, llarg_rust, 0, 2, llvm::False);
                Trunc(bcx, val, Type::i1(bcx.ccx()))
            } else {
                Load(bcx, llarg_rust)
            }
        };

        debug!("argument {}, llarg_foreign={}",
               i, ccx.tn().val_to_string(llarg_foreign));

        // fill padding with undef value
        match arg_ty.pad {
            Some(ty) => llargs_foreign.push(C_undef(ty)),
            None => ()
        }
        llargs_foreign.push(llarg_foreign);
    }

    let cc = llvm_calling_convention(ccx, fn_abi);

    // A function pointer is called without the declaration available, so we have to apply
    // any attributes with ABI implications directly to the call instruction.
    let mut attrs = llvm::AttrBuilder::new();

    // Add attributes that are always applicable, independent of the concrete foreign ABI
    if fn_type.ret_ty.is_indirect() {
        let llret_sz = machine::llsize_of_real(ccx, fn_type.ret_ty.ty);

        // The outptr can be noalias and nocapture because it's entirely
        // invisible to the program. We also know it's nonnull as well
        // as how many bytes we can dereference
        attrs.arg(1, llvm::Attribute::NoAlias)
             .arg(1, llvm::Attribute::NoCapture)
             .arg(1, llvm::DereferenceableAttribute(llret_sz));
    };

    // Add attributes that depend on the concrete foreign ABI
    let mut arg_idx = if fn_type.ret_ty.is_indirect() { 1 } else { 0 };
    match fn_type.ret_ty.attr {
        Some(attr) => { attrs.arg(arg_idx, attr); },
        _ => ()
    }

    arg_idx += 1;
    for arg_ty in &fn_type.arg_tys {
        if arg_ty.is_ignore() {
            continue;
        }
        // skip padding
        if arg_ty.pad.is_some() { arg_idx += 1; }

        if let Some(attr) = arg_ty.attr {
            attrs.arg(arg_idx, attr);
        }

        arg_idx += 1;
    }

    let llforeign_retval = CallWithConv(bcx,
                                        llfn,
                                        &llargs_foreign[..],
                                        cc,
                                        Some(attrs),
                                        call_debug_loc);

    // If the function we just called does not use an outpointer,
    // store the result into the rust outpointer. Cast the outpointer
    // type to match because some ABIs will use a different type than
    // the Rust type. e.g., a {u32,u32} struct could be returned as
    // u64.
    if llsig.ret_def && !fn_type.ret_ty.is_indirect() {
        let llrust_ret_ty = llsig.llret_ty;
        let llforeign_ret_ty = match fn_type.ret_ty.cast {
            Some(ty) => ty,
            None => fn_type.ret_ty.ty
        };

        debug!("llretptr={}", ccx.tn().val_to_string(llretptr));
        debug!("llforeign_retval={}", ccx.tn().val_to_string(llforeign_retval));
        debug!("llrust_ret_ty={}", ccx.tn().type_to_string(llrust_ret_ty));
        debug!("llforeign_ret_ty={}", ccx.tn().type_to_string(llforeign_ret_ty));

        if llrust_ret_ty == llforeign_ret_ty {
            match fn_sig.output {
                ty::FnConverging(result_ty) => {
                    base::store_ty(bcx, llforeign_retval, llretptr, result_ty)
                }
                ty::FnDiverging => {}
            }
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
            base::call_lifetime_start(bcx, llscratch);
            Store(bcx, llforeign_retval, llscratch);
            let llscratch_i8 = BitCast(bcx, llscratch, Type::i8(ccx).ptr_to());
            let llretptr_i8 = BitCast(bcx, llretptr, Type::i8(ccx).ptr_to());
            let llrust_size = machine::llsize_of_store(ccx, llrust_ret_ty);
            let llforeign_align = machine::llalign_of_min(ccx, llforeign_ret_ty);
            let llrust_align = machine::llalign_of_min(ccx, llrust_ret_ty);
            let llalign = cmp::min(llforeign_align, llrust_align);
            debug!("llrust_size={}", llrust_size);
            base::call_memcpy(bcx, llretptr_i8, llscratch_i8,
                              C_uint(ccx, llrust_size), llalign as u32);
            base::call_lifetime_end(bcx, llscratch);
        }
    }

    return bcx;
}

// feature gate SIMD types in FFI, since I (huonw) am not sure the
// ABIs are handled at all correctly.
fn gate_simd_ffi(tcx: &ty::ctxt, decl: &hir::FnDecl, ty: &ty::BareFnTy) {
    if !tcx.sess.features.borrow().simd_ffi {
        let check = |ast_ty: &hir::Ty, ty: ty::Ty| {
            if ty.is_simd() {
                tcx.sess.struct_span_err(ast_ty.span,
                              &format!("use of SIMD type `{}` in FFI is highly experimental and \
                                        may result in invalid code",
                                       pprust::ty_to_string(ast_ty)))
                    .fileline_help(ast_ty.span,
                                   "add #![feature(simd_ffi)] to the crate attributes to enable")
                    .emit();
            }
        };
        let sig = &ty.sig.0;
        for (input, ty) in decl.inputs.iter().zip(&sig.inputs) {
            check(&*input.ty, *ty)
        }
        if let hir::Return(ref ty) = decl.output {
            check(&**ty, sig.output.unwrap())
        }
    }
}

pub fn trans_foreign_mod(ccx: &CrateContext, foreign_mod: &hir::ForeignMod) {
    let _icx = push_ctxt("foreign::trans_foreign_mod");
    for foreign_item in &foreign_mod.items {
        let lname = link_name(foreign_item);

        if let hir::ForeignItemFn(ref decl, _) = foreign_item.node {
            match foreign_mod.abi {
                Rust | RustIntrinsic | PlatformIntrinsic => {}
                abi => {
                    let ty = ccx.tcx().node_id_to_type(foreign_item.id);
                    match ty.sty {
                        ty::TyBareFn(_, bft) => gate_simd_ffi(ccx.tcx(), &**decl, bft),
                        _ => ccx.tcx().sess.span_bug(foreign_item.span,
                                                     "foreign fn's sty isn't a bare_fn_ty?")
                    }

                    register_foreign_item_fn(ccx, abi, ty, &lname, &foreign_item.attrs);
                    // Unlike for other items, we shouldn't call
                    // `base::update_linkage` here.  Foreign items have
                    // special linkage requirements, which are handled
                    // inside `foreign::register_*`.
                }
            }
        }

        ccx.item_symbols().borrow_mut().insert(foreign_item.id,
                                             lname.to_string());
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

pub fn decl_rust_fn_with_foreign_abi<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                               t: Ty<'tcx>,
                                               name: &str)
                                               -> ValueRef {
    let tys = foreign_types_for_fn_ty(ccx, t);
    let llfn_ty = lltype_for_fn_from_foreign_types(ccx, &tys);
    let cconv = match t.sty {
        ty::TyBareFn(_, ref fn_ty) => {
            llvm_calling_convention(ccx, fn_ty.abi)
        }
        _ => panic!("expected bare fn in decl_rust_fn_with_foreign_abi")
    };
    let llfn = declare::declare_fn(ccx, name, cconv, llfn_ty,
                                   ty::FnConverging(ccx.tcx().mk_nil()));
    add_argument_attributes(&tys, llfn);
    debug!("decl_rust_fn_with_foreign_abi(llfn_ty={}, llfn={})",
           ccx.tn().type_to_string(llfn_ty), ccx.tn().val_to_string(llfn));
    llfn
}

pub fn register_rust_fn_with_foreign_abi(ccx: &CrateContext,
                                         sp: Span,
                                         sym: String,
                                         node_id: ast::NodeId)
                                         -> ValueRef {
    let _icx = push_ctxt("foreign::register_foreign_fn");

    let t = ccx.tcx().node_id_to_type(node_id);
    let cconv = match t.sty {
        ty::TyBareFn(_, ref fn_ty) => {
            llvm_calling_convention(ccx, fn_ty.abi)
        }
        _ => panic!("expected bare fn in register_rust_fn_with_foreign_abi")
    };
    let tys = foreign_types_for_fn_ty(ccx, t);
    let llfn_ty = lltype_for_fn_from_foreign_types(ccx, &tys);
    let llfn = base::register_fn_llvmty(ccx, sp, sym, node_id, cconv, llfn_ty);
    add_argument_attributes(&tys, llfn);
    debug!("register_rust_fn_with_foreign_abi(node_id={}, llfn_ty={}, llfn={})",
           node_id, ccx.tn().type_to_string(llfn_ty), ccx.tn().val_to_string(llfn));
    llfn
}

pub fn trans_rust_fn_with_foreign_abi<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                decl: &hir::FnDecl,
                                                body: &hir::Block,
                                                attrs: &[ast::Attribute],
                                                llwrapfn: ValueRef,
                                                param_substs: &'tcx Substs<'tcx>,
                                                id: ast::NodeId,
                                                hash: Option<&str>) {
    let _icx = push_ctxt("foreign::build_foreign_fn");

    let fnty = ccx.tcx().node_id_to_type(id);
    let mty = monomorphize::apply_param_substs(ccx.tcx(), param_substs, &fnty);
    let tys = foreign_types_for_fn_ty(ccx, mty);

    unsafe { // unsafe because we call LLVM operations
        // Build up the Rust function (`foo0` above).
        let llrustfn = build_rust_fn(ccx, decl, body, param_substs, attrs, id, hash);

        // Build up the foreign wrapper (`foo` above).
        return build_wrap_fn(ccx, llrustfn, llwrapfn, &tys, mty);
    }

    fn build_rust_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                               decl: &hir::FnDecl,
                               body: &hir::Block,
                               param_substs: &'tcx Substs<'tcx>,
                               attrs: &[ast::Attribute],
                               id: ast::NodeId,
                               hash: Option<&str>)
                               -> ValueRef
    {
        let _icx = push_ctxt("foreign::foreign::build_rust_fn");
        let tcx = ccx.tcx();
        let t = tcx.node_id_to_type(id);
        let t = monomorphize::apply_param_substs(tcx, param_substs, &t);

        let path =
            tcx.map.def_path_from_id(id)
                   .into_iter()
                   .map(|e| e.data.as_interned_str())
                   .chain(once(special_idents::clownshoe_abi.name.as_str()));
        let ps = link::mangle(path, hash);

        // Compute the type that the function would have if it were just a
        // normal Rust function. This will be the type of the wrappee fn.
        match t.sty {
            ty::TyBareFn(_, ref f) => {
                assert!(f.abi != Rust && f.abi != RustIntrinsic && f.abi != PlatformIntrinsic);
            }
            _ => {
                ccx.sess().bug(&format!("build_rust_fn: extern fn {} has ty {:?}, \
                                        expected a bare fn ty",
                                       ccx.tcx().map.path_to_string(id),
                                       t));
            }
        };

        debug!("build_rust_fn: path={} id={} t={:?}",
               ccx.tcx().map.path_to_string(id),
               id, t);

        let llfn = declare::define_internal_rust_fn(ccx, &ps, t);
        attributes::from_fn_attrs(ccx, attrs, llfn);
        base::trans_fn(ccx, decl, body, llfn, param_substs, id, attrs);
        llfn
    }

    unsafe fn build_wrap_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                      llrustfn: ValueRef,
                                      llwrapfn: ValueRef,
                                      tys: &ForeignTypes<'tcx>,
                                      t: Ty<'tcx>) {
        let _icx = push_ctxt(
            "foreign::trans_rust_fn_with_foreign_abi::build_wrap_fn");

        debug!("build_wrap_fn(llrustfn={}, llwrapfn={}, t={:?})",
               ccx.tn().val_to_string(llrustfn),
               ccx.tn().val_to_string(llwrapfn),
               t);

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

        if llvm::LLVMCountBasicBlocks(llwrapfn) != 0 {
            ccx.sess().bug("wrapping a function inside non-empty wrapper, most likely cause is \
                           multiple functions being wrapped");
        }

        let ptr = "the block\0".as_ptr();
        let the_block = llvm::LLVMAppendBasicBlockInContext(ccx.llcx(), llwrapfn,
                                                            ptr as *const _);

        let builder = ccx.builder();
        builder.position_at_end(the_block);

        // Array for the arguments we will pass to the rust function.
        let mut llrust_args = Vec::new();
        let mut next_foreign_arg_counter: c_uint = 0;
        let mut next_foreign_arg = |pad: bool| -> c_uint {
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

        let rustfn_ty = Type::from_ref(llvm::LLVMTypeOf(llrustfn)).element_type();
        let mut rust_param_tys = rustfn_ty.func_params().into_iter();
        // Push Rust return pointer, using null if it will be unused.
        let rust_uses_outptr = match tys.fn_sig.output {
            ty::FnConverging(ret_ty) => type_of::return_uses_outptr(ccx, ret_ty),
            ty::FnDiverging => false
        };
        let return_alloca: Option<ValueRef>;
        let llrust_ret_ty = if rust_uses_outptr {
            rust_param_tys.next().expect("Missing return type!").element_type()
        } else {
            rustfn_ty.return_type()
        };
        if rust_uses_outptr {
            // Rust expects to use an outpointer. If the foreign fn
            // also uses an outpointer, we can reuse it, but the types
            // may vary, so cast first to the Rust type. If the
            // foreign fn does NOT use an outpointer, we will have to
            // alloca some scratch space on the stack.
            match foreign_outptr {
                Some(llforeign_outptr) => {
                    debug!("out pointer, foreign={}",
                           ccx.tn().val_to_string(llforeign_outptr));
                    let llrust_retptr =
                        builder.bitcast(llforeign_outptr, llrust_ret_ty.ptr_to());
                    debug!("out pointer, foreign={} (casted)",
                           ccx.tn().val_to_string(llrust_retptr));
                    llrust_args.push(llrust_retptr);
                    return_alloca = None;
                }

                None => {
                    let slot = builder.alloca(llrust_ret_ty, "return_alloca");
                    debug!("out pointer, \
                            allocad={}, \
                            llrust_ret_ty={}, \
                            return_ty={:?}",
                           ccx.tn().val_to_string(slot),
                           ccx.tn().type_to_string(llrust_ret_ty),
                           tys.fn_sig.output);
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
        for i in 0..tys.fn_sig.inputs.len() {
            let rust_ty = tys.fn_sig.inputs[i];
            let rust_indirect = type_of::arg_is_indirect(ccx, rust_ty);
            let llty = rust_param_tys.next().expect("Not enough parameter types!");
            let llrust_ty = if rust_indirect {
                llty.element_type()
            } else {
                llty
            };
            let llforeign_arg_ty = tys.fn_ty.arg_tys[i];
            let foreign_indirect = llforeign_arg_ty.is_indirect();

            if llforeign_arg_ty.is_ignore() {
                debug!("skipping ignored arg #{}", i);
                llrust_args.push(C_undef(llrust_ty));
                continue;
            }

            // skip padding
            let foreign_index = next_foreign_arg(llforeign_arg_ty.pad.is_some());
            let mut llforeign_arg = get_param(llwrapfn, foreign_index);

            debug!("llforeign_arg {}{}: {}", "#",
                   i, ccx.tn().val_to_string(llforeign_arg));
            debug!("rust_indirect = {}, foreign_indirect = {}",
                   rust_indirect, foreign_indirect);

            // Ensure that the foreign argument is indirect (by
            // pointer).  It makes adapting types easier, since we can
            // always just bitcast pointers.
            if !foreign_indirect {
                llforeign_arg = if rust_ty.is_bool() {
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
            if llforeign_arg_ty.cast.is_some() && !type_is_fat_ptr(ccx.tcx(), rust_ty){
                assert!(!foreign_indirect);
                llforeign_arg = builder.bitcast(llforeign_arg, llrust_ty.ptr_to());
            }

            let llrust_arg = if rust_indirect || type_is_fat_ptr(ccx.tcx(), rust_ty) {
                llforeign_arg
            } else {
                if rust_ty.is_bool() {
                    let tmp = builder.load_range_assert(llforeign_arg, 0, 2, llvm::False);
                    builder.trunc(tmp, Type::i1(ccx))
                } else if type_of::type_of(ccx, rust_ty).is_aggregate() {
                    // We want to pass small aggregates as immediate values, but using an aggregate
                    // LLVM type for this leads to bad optimizations, so its arg type is an
                    // appropriately sized integer and we have to convert it
                    let tmp = builder.bitcast(llforeign_arg,
                                              type_of::arg_type_of(ccx, rust_ty).ptr_to());
                    let load = builder.load(tmp);
                    llvm::LLVMSetAlignment(load, type_of::align_of(ccx, rust_ty));
                    load
                } else {
                    builder.load(llforeign_arg)
                }
            };

            debug!("llrust_arg {}{}: {}", "#",
                   i, ccx.tn().val_to_string(llrust_arg));
            if type_is_fat_ptr(ccx.tcx(), rust_ty) {
                let next_llrust_ty = rust_param_tys.next().expect("Not enough parameter types!");
                llrust_args.push(builder.load(builder.bitcast(builder.struct_gep(
                                llrust_arg, abi::FAT_PTR_ADDR), llrust_ty.ptr_to())));
                llrust_args.push(builder.load(builder.bitcast(builder.struct_gep(
                                llrust_arg, abi::FAT_PTR_EXTRA), next_llrust_ty.ptr_to())));
            } else {
                llrust_args.push(llrust_arg);
            }
        }

        // Perform the call itself
        debug!("calling llrustfn = {}, t = {:?}",
               ccx.tn().val_to_string(llrustfn), t);
        let attributes = attributes::from_fn_type(ccx, t);
        let llrust_ret_val = builder.call(llrustfn, &llrust_args, Some(attributes));

        // Get the return value where the foreign fn expects it.
        let llforeign_ret_ty = match tys.fn_ty.ret_ty.cast {
            Some(ty) => ty,
            None => tys.fn_ty.ret_ty.ty
        };
        match foreign_outptr {
            None if !tys.llsig.ret_def => {
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
                    builder.bitcast(llforeign_outptr, llrust_ret_ty.ptr_to());
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

pub fn link_name(i: &hir::ForeignItem) -> InternedString {
    match attr::first_attr_value_str_by_name(&i.attrs, "link_name") {
        Some(ln) => ln.clone(),
        None => match weak_lang_items::link_name(&i.attrs) {
            Some(name) => name,
            None => i.name.as_str(),
        }
    }
}

/// The ForeignSignature is the LLVM types of the arguments/return type of a function. Note that
/// these LLVM types are not quite the same as the LLVM types would be for a native Rust function
/// because foreign functions just plain ignore modes. They also don't pass aggregate values by
/// pointer like we do.
fn foreign_signature<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                               fn_sig: &ty::FnSig<'tcx>,
                               arg_tys: &[Ty<'tcx>])
                               -> LlvmSignature {
    let llarg_tys = arg_tys.iter().map(|&arg| foreign_arg_type_of(ccx, arg)).collect();
    let (llret_ty, ret_def) = match fn_sig.output {
        ty::FnConverging(ret_ty) =>
            (type_of::foreign_arg_type_of(ccx, ret_ty), !return_type_is_void(ccx, ret_ty)),
        ty::FnDiverging =>
            (Type::nil(ccx), false)
    };
    LlvmSignature {
        llarg_tys: llarg_tys,
        llret_ty: llret_ty,
        ret_def: ret_def
    }
}

fn foreign_types_for_fn_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                     ty: Ty<'tcx>) -> ForeignTypes<'tcx> {
    let fn_sig = match ty.sty {
        ty::TyBareFn(_, ref fn_ty) => &fn_ty.sig,
        _ => ccx.sess().bug("foreign_types_for_fn_ty called on non-function type")
    };
    let fn_sig = ccx.tcx().erase_late_bound_regions(fn_sig);
    let fn_sig = infer::normalize_associated_type(ccx.tcx(), &fn_sig);
    let llsig = foreign_signature(ccx, &fn_sig, &fn_sig.inputs);
    let fn_ty = cabi::compute_abi_info(ccx,
                                       &llsig.llarg_tys,
                                       llsig.llret_ty,
                                       llsig.ret_def);
    debug!("foreign_types_for_fn_ty(\
           ty={:?}, \
           llsig={} -> {}, \
           fn_ty={} -> {}, \
           ret_def={}",
           ty,
           ccx.tn().types_to_str(&llsig.llarg_tys),
           ccx.tn().type_to_string(llsig.llret_ty),
           ccx.tn().types_to_str(&fn_ty.arg_tys.iter().map(|t| t.ty).collect::<Vec<_>>()),
           ccx.tn().type_to_string(fn_ty.ret_ty.ty),
           llsig.ret_def);

    ForeignTypes {
        fn_sig: fn_sig,
        llsig: llsig,
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

    for &arg_ty in &tys.fn_ty.arg_tys {
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
        Type::variadic_func(&llargument_tys, &llreturn_ty)
    } else {
        Type::func(&llargument_tys[..], &llreturn_ty)
    }
}

pub fn lltype_for_foreign_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                       ty: Ty<'tcx>) -> Type {
    lltype_for_fn_from_foreign_types(ccx, &foreign_types_for_fn_ty(ccx, ty))
}

fn add_argument_attributes(tys: &ForeignTypes,
                           llfn: ValueRef) {
    let mut i = if tys.fn_ty.ret_ty.is_indirect() {
        1
    } else {
        0
    };

    match tys.fn_ty.ret_ty.attr {
        Some(attr) => unsafe {
            llvm::LLVMAddFunctionAttribute(llfn, i as c_uint, attr.bits() as u64);
        },
        None => {}
    }

    i += 1;

    for &arg_ty in &tys.fn_ty.arg_tys {
        if arg_ty.is_ignore() {
            continue;
        }
        // skip padding
        if arg_ty.pad.is_some() { i += 1; }

        match arg_ty.attr {
            Some(attr) => unsafe {
                llvm::LLVMAddFunctionAttribute(llfn, i as c_uint, attr.bits() as u64);
            },
            None => ()
        }

        i += 1;
    }
}
