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

use lib::llvm::{ValueRef};
use lib::llvm::False;
use lib::llvm::llvm;
use middle::trans::common::*;
use middle::trans::type_of;
use middle::ty;
use util::ppaux::ty_to_str;

use middle::trans::type_::Type;

// ______________________________________________________________________
// compute sizeof / alignof

// Returns the number of bytes clobbered by a Store to this type.
pub fn llsize_of_store(cx: &CrateContext, ty: Type) -> uint {
    unsafe {
        return llvm::LLVMStoreSizeOfType(cx.td.lltd, ty.to_ref()) as uint;
    }
}

// Returns the number of bytes between successive elements of type T in an
// array of T. This is the "ABI" size. It includes any ABI-mandated padding.
pub fn llsize_of_alloc(cx: &CrateContext, ty: Type) -> uint {
    unsafe {
        return llvm::LLVMABISizeOfType(cx.td.lltd, ty.to_ref()) as uint;
    }
}

// Returns, as near as we can figure, the "real" size of a type. As in, the
// bits in this number of bytes actually carry data related to the datum
// with the type. Not junk, padding, accidentally-damaged words, or
// whatever. Rounds up to the nearest byte though, so if you have a 1-bit
// value, we return 1 here, not 0. Most of rustc works in bytes. Be warned
// that LLVM *does* distinguish between e.g. a 1-bit value and an 8-bit value
// at the codegen level! In general you should prefer `llbitsize_of_real`
// below.
pub fn llsize_of_real(cx: &CrateContext, ty: Type) -> uint {
    unsafe {
        let nbits = llvm::LLVMSizeOfTypeInBits(cx.td.lltd, ty.to_ref()) as uint;
        if nbits & 7u != 0u {
            // Not an even number of bytes, spills into "next" byte.
            1u + (nbits >> 3)
        } else {
            nbits >> 3
        }
    }
}

/// Returns the "real" size of the type in bits.
pub fn llbitsize_of_real(cx: &CrateContext, ty: Type) -> uint {
    unsafe {
        llvm::LLVMSizeOfTypeInBits(cx.td.lltd, ty.to_ref()) as uint
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

// Returns the "default" size of t (see above), or 1 if the size would
// be zero.  This is important for things like vectors that expect
// space to be consumed.
pub fn nonzero_llsize_of(cx: &CrateContext, ty: Type) -> ValueRef {
    if llbitsize_of_real(cx, ty) == 0 {
        unsafe { llvm::LLVMConstInt(cx.int_type.to_ref(), 1, False) }
    } else {
        llsize_of(cx, ty)
    }
}

// Returns the preferred alignment of the given type for the current target.
// The preferred alignment may be larger than the alignment used when
// packing the type into structs. This will be used for things like
// allocations inside a stack frame, which LLVM has a free hand in.
pub fn llalign_of_pref(cx: &CrateContext, ty: Type) -> uint {
    unsafe {
        return llvm::LLVMPreferredAlignmentOfType(cx.td.lltd, ty.to_ref()) as uint;
    }
}

// Returns the minimum alignment of a type required by the platform.
// This is the alignment that will be used for struct fields, arrays,
// and similar ABI-mandated things.
pub fn llalign_of_min(cx: &CrateContext, ty: Type) -> uint {
    unsafe {
        return llvm::LLVMABIAlignmentOfType(cx.td.lltd, ty.to_ref()) as uint;
    }
}

// Returns the "default" alignment of t, which is calculated by casting
// null to a record containing a single-bit followed by a t value, then
// doing gep(0,1) to get at the trailing (and presumably padded) t cell.
pub fn llalign_of(cx: &CrateContext, ty: Type) -> ValueRef {
    unsafe {
        return llvm::LLVMConstIntCast(
            llvm::LLVMAlignOf(ty.to_ref()), cx.int_type.to_ref(), False);
    }
}

// Computes the size of the data part of an enum.
pub fn static_size_of_enum(cx: &mut CrateContext, t: ty::t) -> uint {
    if cx.enum_sizes.contains_key(&t) {
        return cx.enum_sizes.get_copy(&t);
    }

    debug!("static_size_of_enum %s", ty_to_str(cx.tcx, t));

    match ty::get(t).sty {
        ty::ty_enum(tid, ref substs) => {
            // Compute max(variant sizes).
            let mut max_size = 0;
            let variants = ty::enum_variants(cx.tcx, tid);
            for variants.iter().advance |variant| {
                if variant.args.len() == 0 {
                    loop;
                }

                let lltypes = variant.args.map(|&variant_arg| {
                    let substituted = ty::subst(cx.tcx, substs, variant_arg);
                    type_of::sizing_type_of(cx, substituted)
                });

                debug!("static_size_of_enum: variant %s type %s",
                       cx.tcx.sess.str_of(variant.name),
                       cx.tn.type_to_str(Type::struct_(lltypes, false)));

                let this_size = llsize_of_real(cx, Type::struct_(lltypes, false));
                if max_size < this_size {
                    max_size = this_size;
                }
            }
            cx.enum_sizes.insert(t, max_size);
            return max_size;
        }
        _ => cx.sess.bug("static_size_of_enum called on non-enum")
    }
}
