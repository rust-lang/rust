// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Information concerning the machine representation of various types.

use middle::trans::common::*;

// Creates a simpler, size-equivalent type. The resulting type is guaranteed
// to have (a) the same size as the type that was passed in; (b) to be non-
// recursive. This is done by replacing all boxes in a type with boxed unit
// types.
// This should reduce all pointers to some simple pointer type, to
// ensure that we don't recurse endlessly when computing the size of a
// nominal type that has pointers to itself in it.
pub fn simplify_type(tcx: ty::ctxt, typ: ty::t) -> ty::t {
    fn nilptr(tcx: ty::ctxt) -> ty::t {
        ty::mk_ptr(tcx, {ty: ty::mk_nil(tcx), mutbl: ast::m_imm})
    }
    fn simplifier(tcx: ty::ctxt, typ: ty::t) -> ty::t {
        match ty::get(typ).sty {
          ty::ty_box(_) | ty::ty_opaque_box | ty::ty_uniq(_) |
          ty::ty_evec(_, ty::vstore_uniq) | ty::ty_evec(_, ty::vstore_box) |
          ty::ty_estr(ty::vstore_uniq) | ty::ty_estr(ty::vstore_box) |
          ty::ty_ptr(_) | ty::ty_rptr(_,_) => nilptr(tcx),
          ty::ty_fn(_) => ty::mk_tup(tcx, ~[nilptr(tcx), nilptr(tcx)]),
          ty::ty_evec(_, ty::vstore_slice(_)) |
          ty::ty_estr(ty::vstore_slice(_)) => {
            ty::mk_tup(tcx, ~[nilptr(tcx), ty::mk_int(tcx)])
          }
          // Reduce a class type to a record type in which all the fields are
          // simplified
          ty::ty_class(did, ref substs) => {
            let simpl_fields = (if ty::ty_dtor(tcx, did).is_present() {
                // remember the drop flag
                  ~[{ident: syntax::parse::token::special_idents::dtor,
                     mt: {ty: ty::mk_u8(tcx),
                          mutbl: ast::m_mutbl}}] }
                else { ~[] }) +
                do ty::lookup_class_fields(tcx, did).map |f| {
                 let t = ty::lookup_field_type(tcx, did, f.id, substs);
                 {ident: f.ident,
                  mt: {ty: simplify_type(tcx, t), mutbl: ast::m_const}}
            };
            ty::mk_rec(tcx, simpl_fields)
          }
          _ => typ
        }
    }
    ty::fold_ty(tcx, typ, |t| simplifier(tcx, t))
}

// ______________________________________________________________________
// compute sizeof / alignof

pub type metrics = {
    bcx: block,
    sz: ValueRef,
    align: ValueRef
};

pub type tag_metrics = {
    bcx: block,
    sz: ValueRef,
    align: ValueRef,
    payload_align: ValueRef
};

// Returns the number of bytes clobbered by a Store to this type.
pub fn llsize_of_store(cx: @crate_ctxt, t: TypeRef) -> uint {
    return llvm::LLVMStoreSizeOfType(cx.td.lltd, t) as uint;
}

// Returns the number of bytes between successive elements of type T in an
// array of T. This is the "ABI" size. It includes any ABI-mandated padding.
pub fn llsize_of_alloc(cx: @crate_ctxt, t: TypeRef) -> uint {
    return llvm::LLVMABISizeOfType(cx.td.lltd, t) as uint;
}

// Returns, as near as we can figure, the "real" size of a type. As in, the
// bits in this number of bytes actually carry data related to the datum
// with the type. Not junk, padding, accidentally-damaged words, or
// whatever. Rounds up to the nearest byte though, so if you have a 1-bit
// value, we return 1 here, not 0. Most of rustc works in bytes. Be warned
// that LLVM *does* distinguish between e.g. a 1-bit value and an 8-bit value
// at the codegen level! In general you should prefer `llbitsize_of_real`
// below.
pub fn llsize_of_real(cx: @crate_ctxt, t: TypeRef) -> uint {
    let nbits = llvm::LLVMSizeOfTypeInBits(cx.td.lltd, t) as uint;
    if nbits & 7u != 0u {
        // Not an even number of bytes, spills into "next" byte.
        1u + (nbits >> 3)
    } else {
        nbits >> 3
    }
}

/// Returns the "real" size of the type in bits.
pub fn llbitsize_of_real(cx: @crate_ctxt, t: TypeRef) -> uint {
    llvm::LLVMSizeOfTypeInBits(cx.td.lltd, t) as uint
}

// Returns the "default" size of t, which is calculated by casting null to a
// *T and then doing gep(1) on it and measuring the result. Really, look in
// the LLVM sources. It does that. So this is likely similar to the ABI size
// (i.e. including alignment-padding), but goodness knows which alignment it
// winds up using. Probably the ABI one? Not recommended.
pub fn llsize_of(cx: @crate_ctxt, t: TypeRef) -> ValueRef {
    return llvm::LLVMConstIntCast(lib::llvm::llvm::LLVMSizeOf(t), cx.int_type,
                               False);
}

// Returns the preferred alignment of the given type for the current target.
// The preffered alignment may be larger than the alignment used when
// packing the type into structs. This will be used for things like
// allocations inside a stack frame, which LLVM has a free hand in.
pub fn llalign_of_pref(cx: @crate_ctxt, t: TypeRef) -> uint {
    return llvm::LLVMPreferredAlignmentOfType(cx.td.lltd, t) as uint;
}

// Returns the minimum alignment of a type required by the plattform.
// This is the alignment that will be used for struct fields, arrays,
// and similar ABI-mandated things.
pub fn llalign_of_min(cx: @crate_ctxt, t: TypeRef) -> uint {
    return llvm::LLVMABIAlignmentOfType(cx.td.lltd, t) as uint;
}

// Returns the "default" alignment of t, which is calculated by casting
// null to a record containing a single-bit followed by a t value, then
// doing gep(0,1) to get at the trailing (and presumably padded) t cell.
pub fn llalign_of(cx: @crate_ctxt, t: TypeRef) -> ValueRef {
    return llvm::LLVMConstIntCast(
        lib::llvm::llvm::LLVMAlignOf(t), cx.int_type, False);
}

// Computes the size of the data part of an enum.
pub fn static_size_of_enum(cx: @crate_ctxt, t: ty::t) -> uint {
    if cx.enum_sizes.contains_key(t) { return cx.enum_sizes.get(t); }
    match ty::get(t).sty {
      ty::ty_enum(tid, ref substs) => {
        // Compute max(variant sizes).
        let mut max_size = 0u;
        let variants = ty::enum_variants(cx.tcx, tid);
        for vec::each(*variants) |variant| {
            let tup_ty = simplify_type(cx.tcx,
                                       ty::mk_tup(cx.tcx, variant.args));
            // Perform any type parameter substitutions.
            let tup_ty = ty::subst(cx.tcx, substs, tup_ty);
            // Here we possibly do a recursive call.
            let this_size =
                llsize_of_real(cx, type_of::type_of(cx, tup_ty));
            if max_size < this_size { max_size = this_size; }
        }
        cx.enum_sizes.insert(t, max_size);
        return max_size;
      }
      _ => cx.sess.bug(~"static_size_of_enum called on non-enum")
    }
}

