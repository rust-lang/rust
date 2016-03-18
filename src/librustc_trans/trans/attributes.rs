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
use llvm::{self, ValueRef};
use session::config::NoDebugInfo;
pub use syntax::attr::InlineAttr;
use syntax::ast;
use trans::context::CrateContext;

/// Mark LLVM function to use provided inline heuristic.
#[inline]
pub fn inline(val: ValueRef, inline: InlineAttr) {
    use self::InlineAttr::*;
    match inline {
        Hint   => llvm::SetFunctionAttribute(val, llvm::Attribute::InlineHint),
        Always => llvm::SetFunctionAttribute(val, llvm::Attribute::AlwaysInline),
        Never  => llvm::SetFunctionAttribute(val, llvm::Attribute::NoInline),
        None   => {
            let attr = llvm::Attribute::InlineHint |
                       llvm::Attribute::AlwaysInline |
                       llvm::Attribute::NoInline;
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
        llvm::SetFunctionAttribute(val, llvm::Attribute::UWTable);
    } else {
        unsafe {
            llvm::LLVMRemoveFunctionAttr(
                val,
                llvm::Attribute::UWTable.bits() as c_ulonglong,
            );
        }
    }
}

/// Tell LLVM whether the function can or cannot unwind.
#[inline]
pub fn unwind(val: ValueRef, can_unwind: bool) {
    if can_unwind {
        unsafe {
            llvm::LLVMRemoveFunctionAttr(
                val,
                llvm::Attribute::NoUnwind.bits() as c_ulonglong,
            );
        }
    } else {
        llvm::SetFunctionAttribute(val, llvm::Attribute::NoUnwind);
    }
}

/// Tell LLVM whether it should optimise function for size.
#[inline]
#[allow(dead_code)] // possibly useful function
pub fn set_optimize_for_size(val: ValueRef, optimize: bool) {
    if optimize {
        llvm::SetFunctionAttribute(val, llvm::Attribute::OptimizeForSize);
    } else {
        unsafe {
            llvm::LLVMRemoveFunctionAttr(
                val,
                llvm::Attribute::OptimizeForSize.bits() as c_ulonglong,
            );
        }
    }
}

/// Composite function which sets LLVM attributes for function depending on its AST (#[attribute])
/// attributes.
pub fn from_fn_attrs(ccx: &CrateContext, attrs: &[ast::Attribute], llfn: ValueRef) {
    use syntax::attr::*;
    inline(llfn, find_inline_attr(Some(ccx.sess().diagnostic()), attrs));

    // FIXME: #11906: Omitting frame pointers breaks retrieving the value of a
    // parameter.
    let no_fp_elim = (ccx.sess().opts.debuginfo != NoDebugInfo) ||
                     !ccx.sess().target.target.options.eliminate_frame_pointer;
    if no_fp_elim {
        unsafe {
            let attr = "no-frame-pointer-elim\0".as_ptr() as *const _;
            let val = "true\0".as_ptr() as *const _;
            llvm::LLVMAddFunctionAttrStringValue(llfn,
                                                 llvm::FunctionIndex as c_uint,
                                                 attr, val);
        }
    }

    for attr in attrs {
        if attr.check_name("cold") {
            llvm::Attributes::default().set(llvm::Attribute::Cold)
                .apply_llfn(llvm::FunctionIndex as usize, llfn)
        } else if attr.check_name("allocator") {
            llvm::Attributes::default().set(llvm::Attribute::NoAlias)
                .apply_llfn(llvm::ReturnIndex as usize, llfn)
        } else if attr.check_name("unwind") {
            unwind(llfn, true);
        }
    }
}
