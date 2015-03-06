// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! Set and unset common attributes on LLVM values.

use libc::{c_uint, c_ulonglong};
use llvm::{self, ValueRef, AttrHelper};
use middle::ty::{self, ClosureTyper};
use syntax::abi;
use syntax::ast;
pub use syntax::attr::InlineAttr;
use trans::base;
use trans::common;
use trans::context::CrateContext;
use trans::machine;
use trans::type_of;

/// Mark LLVM function to use split stack.
#[inline]
pub fn split_stack(val: ValueRef, set: bool) {
    unsafe {
        let attr = "split-stack\0".as_ptr() as *const _;
        if set {
            llvm::LLVMAddFunctionAttrString(val, llvm::FunctionIndex as c_uint, attr);
        } else {
            llvm::LLVMRemoveFunctionAttrString(val, llvm::FunctionIndex as c_uint, attr);
        }
    }
}

/// Mark LLVM function to use provided inline heuristic.
#[inline]
pub fn inline(val: ValueRef, inline: InlineAttr) {
    use self::InlineAttr::*;
    match inline {
        Hint   => llvm::SetFunctionAttribute(val, llvm::InlineHintAttribute),
        Always => llvm::SetFunctionAttribute(val, llvm::AlwaysInlineAttribute),
        Never  => llvm::SetFunctionAttribute(val, llvm::NoInlineAttribute),
        None   => {
            let attr = llvm::InlineHintAttribute |
                       llvm::AlwaysInlineAttribute |
                       llvm::NoInlineAttribute;
            unsafe {
                llvm::LLVMRemoveFunctionAttr(val, attr.bits() as c_ulonglong)
            }
        },
    };
}

/// Tell LLVM to emit or not emit the information necessary to unwind the stack for the function.
#[inline]
pub fn emit_uwtable(val: ValueRef, emit: bool) {
    if emit {
        llvm::SetFunctionAttribute(val, llvm::UWTableAttribute);
    } else {
        unsafe {
            llvm::LLVMRemoveFunctionAttr(val, llvm::UWTableAttribute.bits() as c_ulonglong);
        }
    }
}

/// Tell LLVM whether the function can or cannot unwind.
#[inline]
#[allow(dead_code)] // possibly useful function
pub fn unwind(val: ValueRef, can_unwind: bool) {
    if can_unwind {
        unsafe {
            llvm::LLVMRemoveFunctionAttr(val, llvm::NoUnwindAttribute.bits() as c_ulonglong);
        }
    } else {
        llvm::SetFunctionAttribute(val, llvm::NoUnwindAttribute);
    }
}

/// Tell LLVM whether it should optimise function for size.
#[inline]
#[allow(dead_code)] // possibly useful function
pub fn set_optimize_for_size(val: ValueRef, optimize: bool) {
    if optimize {
        llvm::SetFunctionAttribute(val, llvm::OptimizeForSizeAttribute);
    } else {
        unsafe {
            llvm::LLVMRemoveFunctionAttr(val, llvm::OptimizeForSizeAttribute.bits() as c_ulonglong);
        }
    }
}

/// Composite function which sets LLVM attributes for function depending on its AST (#[attribute])
/// attributes.
pub fn from_fn_attrs(ccx: &CrateContext, attrs: &[ast::Attribute], llfn: ValueRef) {
    use syntax::attr::*;
    inline(llfn, find_inline_attr(Some(ccx.sess().diagnostic()), attrs));

    for attr in attrs {
        if attr.check_name("no_stack_check") {
            split_stack(llfn, false);
        } else if attr.check_name("cold") {
            unsafe {
                llvm::LLVMAddFunctionAttribute(llfn,
                                               llvm::FunctionIndex as c_uint,
                                               llvm::ColdAttribute as u64)
            }
        } else if attr.check_name("allocator") {
            llvm::NoAliasAttribute.apply_llfn(llvm::ReturnIndex as c_uint, llfn);
        }
    }
}

/// Composite function which converts function type into LLVM attributes for the function.
pub fn from_fn_type<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fn_type: ty::Ty<'tcx>)
                              -> llvm::AttrBuilder {
    use middle::ty::{BrAnon, ReLateBound};

    let function_type;
    let (fn_sig, abi, env_ty) = match fn_type.sty {
        ty::ty_bare_fn(_, ref f) => (&f.sig, f.abi, None),
        ty::ty_closure(closure_did, substs) => {
            let typer = common::NormalizingClosureTyper::new(ccx.tcx());
            function_type = typer.closure_type(closure_did, substs);
            let self_type = base::self_type_for_closure(ccx, closure_did, fn_type);
            (&function_type.sig, abi::RustCall, Some(self_type))
        }
        _ => ccx.sess().bug("expected closure or function.")
    };

    let fn_sig = ty::erase_late_bound_regions(ccx.tcx(), fn_sig);

    let mut attrs = llvm::AttrBuilder::new();
    let ret_ty = fn_sig.output;

    // These have an odd calling convention, so we need to manually
    // unpack the input ty's
    let input_tys = match fn_type.sty {
        ty::ty_closure(..) => {
            assert!(abi == abi::RustCall);

            match fn_sig.inputs[0].sty {
                ty::ty_tup(ref inputs) => {
                    let mut full_inputs = vec![env_ty.expect("Missing closure environment")];
                    full_inputs.push_all(inputs);
                    full_inputs
                }
                _ => ccx.sess().bug("expected tuple'd inputs")
            }
        },
        ty::ty_bare_fn(..) if abi == abi::RustCall => {
            let mut inputs = vec![fn_sig.inputs[0]];

            match fn_sig.inputs[1].sty {
                ty::ty_tup(ref t_in) => {
                    inputs.push_all(&t_in[..]);
                    inputs
                }
                _ => ccx.sess().bug("expected tuple'd inputs")
            }
        }
        _ => fn_sig.inputs.clone()
    };

    // Index 0 is the return value of the llvm func, so we start at 1
    let mut first_arg_offset = 1;
    if let ty::FnConverging(ret_ty) = ret_ty {
        // A function pointer is called without the declaration
        // available, so we have to apply any attributes with ABI
        // implications directly to the call instruction. Right now,
        // the only attribute we need to worry about is `sret`.
        if type_of::return_uses_outptr(ccx, ret_ty) {
            let llret_sz = machine::llsize_of_real(ccx, type_of::type_of(ccx, ret_ty));

            // The outptr can be noalias and nocapture because it's entirely
            // invisible to the program. We also know it's nonnull as well
            // as how many bytes we can dereference
            attrs.arg(1, llvm::StructRetAttribute)
                 .arg(1, llvm::NoAliasAttribute)
                 .arg(1, llvm::NoCaptureAttribute)
                 .arg(1, llvm::DereferenceableAttribute(llret_sz));

            // Add one more since there's an outptr
            first_arg_offset += 1;
        } else {
            // The `noalias` attribute on the return value is useful to a
            // function ptr caller.
            match ret_ty.sty {
                // `~` pointer return values never alias because ownership
                // is transferred
                ty::ty_uniq(it) if common::type_is_sized(ccx.tcx(), it) => {
                    attrs.ret(llvm::NoAliasAttribute);
                }
                _ => {}
            }

            // We can also mark the return value as `dereferenceable` in certain cases
            match ret_ty.sty {
                // These are not really pointers but pairs, (pointer, len)
                ty::ty_rptr(_, ty::mt { ty: inner, .. })
                | ty::ty_uniq(inner) if common::type_is_sized(ccx.tcx(), inner) => {
                    let llret_sz = machine::llsize_of_real(ccx, type_of::type_of(ccx, inner));
                    attrs.ret(llvm::DereferenceableAttribute(llret_sz));
                }
                _ => {}
            }

            if let ty::ty_bool = ret_ty.sty {
                attrs.ret(llvm::ZExtAttribute);
            }
        }
    }

    for (idx, &t) in input_tys.iter().enumerate().map(|(i, v)| (i + first_arg_offset, v)) {
        match t.sty {
            // this needs to be first to prevent fat pointers from falling through
            _ if !common::type_is_immediate(ccx, t) => {
                let llarg_sz = machine::llsize_of_real(ccx, type_of::type_of(ccx, t));

                // For non-immediate arguments the callee gets its own copy of
                // the value on the stack, so there are no aliases. It's also
                // program-invisible so can't possibly capture
                attrs.arg(idx, llvm::NoAliasAttribute)
                     .arg(idx, llvm::NoCaptureAttribute)
                     .arg(idx, llvm::DereferenceableAttribute(llarg_sz));
            }

            ty::ty_bool => {
                attrs.arg(idx, llvm::ZExtAttribute);
            }

            // `~` pointer parameters never alias because ownership is transferred
            ty::ty_uniq(inner) => {
                let llsz = machine::llsize_of_real(ccx, type_of::type_of(ccx, inner));

                attrs.arg(idx, llvm::NoAliasAttribute)
                     .arg(idx, llvm::DereferenceableAttribute(llsz));
            }

            // `&mut` pointer parameters never alias other parameters, or mutable global data
            //
            // `&T` where `T` contains no `UnsafeCell<U>` is immutable, and can be marked as both
            // `readonly` and `noalias`, as LLVM's definition of `noalias` is based solely on
            // memory dependencies rather than pointer equality
            ty::ty_rptr(b, mt) if mt.mutbl == ast::MutMutable ||
                                  !ty::type_contents(ccx.tcx(), mt.ty).interior_unsafe() => {

                let llsz = machine::llsize_of_real(ccx, type_of::type_of(ccx, mt.ty));
                attrs.arg(idx, llvm::NoAliasAttribute)
                     .arg(idx, llvm::DereferenceableAttribute(llsz));

                if mt.mutbl == ast::MutImmutable {
                    attrs.arg(idx, llvm::ReadOnlyAttribute);
                }

                if let ReLateBound(_, BrAnon(_)) = *b {
                    attrs.arg(idx, llvm::NoCaptureAttribute);
                }
            }

            // When a reference in an argument has no named lifetime, it's impossible for that
            // reference to escape this function (returned or stored beyond the call by a closure).
            ty::ty_rptr(&ReLateBound(_, BrAnon(_)), mt) => {
                let llsz = machine::llsize_of_real(ccx, type_of::type_of(ccx, mt.ty));
                attrs.arg(idx, llvm::NoCaptureAttribute)
                     .arg(idx, llvm::DereferenceableAttribute(llsz));
            }

            // & pointer parameters are also never null and we know exactly how
            // many bytes we can dereference
            ty::ty_rptr(_, mt) => {
                let llsz = machine::llsize_of_real(ccx, type_of::type_of(ccx, mt.ty));
                attrs.arg(idx, llvm::DereferenceableAttribute(llsz));
            }
            _ => ()
        }
    }

    attrs
}
