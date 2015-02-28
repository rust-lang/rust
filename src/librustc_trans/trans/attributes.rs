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

use llvm::{self, ValueRef, AttrHelper};
use syntax::ast;
use syntax::attr::InlineAttr;
pub use syntax::attr::InlineAttr::*;
use trans::context::CrateContext;

use libc::{c_uint, c_ulonglong};

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
    match inline {
        InlineHint   => llvm::SetFunctionAttribute(val, llvm::InlineHintAttribute),
        InlineAlways => llvm::SetFunctionAttribute(val, llvm::AlwaysInlineAttribute),
        InlineNever  => llvm::SetFunctionAttribute(val, llvm::NoInlineAttribute),
        InlineNone   => {
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
pub fn convert_fn_attrs_to_llvm(ccx: &CrateContext, attrs: &[ast::Attribute], llfn: ValueRef) {
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
