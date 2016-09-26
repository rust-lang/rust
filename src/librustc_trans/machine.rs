// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Information concerning the machine representation of various types.

#![allow(non_camel_case_types)]

use llvm::{self, ValueRef};
use common::*;

use type_::Type;

pub type llbits = u64;
pub type llsize = u64;
pub type llalign = u32;

// ______________________________________________________________________
// compute sizeof / alignof

// Returns the number of bytes between successive elements of type T in an
// array of T. This is the "ABI" size. It includes any ABI-mandated padding.
pub fn llsize_of_alloc(cx: &CrateContext, ty: Type) -> llsize {
    unsafe {
        return llvm::LLVMABISizeOfType(cx.td(), ty.to_ref());
    }
}

/// Returns the "real" size of the type in bits.
pub fn llbitsize_of_real(cx: &CrateContext, ty: Type) -> llbits {
    unsafe {
        llvm::LLVMSizeOfTypeInBits(cx.td(), ty.to_ref())
    }
}

/// Returns the size of the type as an LLVM constant integer value.
pub fn llsize_of(cx: &CrateContext, ty: Type) -> ValueRef {
    // Once upon a time, this called LLVMSizeOf, which does a
    // getelementptr(1) on a null pointer and casts to an int, in
    // order to obtain the type size as a value without requiring the
    // target data layout.  But we have the target data layout, so
    // there's no need for that contrivance.  The instruction
    // selection DAG generator would flatten that GEP(1) node into a
    // constant of the type's alloc size, so let's save it some work.
    return C_uint(cx, llsize_of_alloc(cx, ty));
}

// Returns the preferred alignment of the given type for the current target.
// The preferred alignment may be larger than the alignment used when
// packing the type into structs. This will be used for things like
// allocations inside a stack frame, which LLVM has a free hand in.
pub fn llalign_of_pref(cx: &CrateContext, ty: Type) -> llalign {
    unsafe {
        return llvm::LLVMPreferredAlignmentOfType(cx.td(), ty.to_ref());
    }
}

// Returns the minimum alignment of a type required by the platform.
// This is the alignment that will be used for struct fields, arrays,
// and similar ABI-mandated things.
pub fn llalign_of_min(cx: &CrateContext, ty: Type) -> llalign {
    unsafe {
        return llvm::LLVMABIAlignmentOfType(cx.td(), ty.to_ref());
    }
}

pub fn llelement_offset(cx: &CrateContext, struct_ty: Type, element: usize) -> u64 {
    unsafe {
        return llvm::LLVMOffsetOfElement(cx.td(),
                                         struct_ty.to_ref(),
                                         element as u32);
    }
}
