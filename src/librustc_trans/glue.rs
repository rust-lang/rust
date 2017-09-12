// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//!
//
// Code relating to drop glue.

use std;

use builder::Builder;
use common::*;
use llvm::{ValueRef};
use llvm;
use meth;
use rustc::ty::layout::LayoutOf;
use rustc::ty::{self, Ty};
use value::Value;

pub fn size_and_align_of_dst<'a, 'tcx>(bcx: &Builder<'a, 'tcx>, t: Ty<'tcx>, info: ValueRef)
                                       -> (ValueRef, ValueRef) {
    debug!("calculate size of DST: {}; with lost info: {:?}",
           t, Value(info));
    if bcx.ccx.shared().type_is_sized(t) {
        let (size, align) = bcx.ccx.size_and_align_of(t);
        debug!("size_and_align_of_dst t={} info={:?} size: {:?} align: {:?}",
               t, Value(info), size, align);
        let size = C_usize(bcx.ccx, size.bytes());
        let align = C_usize(bcx.ccx, align.abi());
        return (size, align);
    }
    assert!(!info.is_null());
    match t.sty {
        ty::TyDynamic(..) => {
            // load size/align from vtable
            (meth::SIZE.get_usize(bcx, info), meth::ALIGN.get_usize(bcx, info))
        }
        ty::TySlice(_) | ty::TyStr => {
            let unit = t.sequence_element_type(bcx.tcx());
            // The info in this case is the length of the str, so the size is that
            // times the unit size.
            let (size, align) = bcx.ccx.size_and_align_of(unit);
            (bcx.mul(info, C_usize(bcx.ccx, size.bytes())),
             C_usize(bcx.ccx, align.abi()))
        }
        _ => {
            let ccx = bcx.ccx;
            // First get the size of all statically known fields.
            // Don't use size_of because it also rounds up to alignment, which we
            // want to avoid, as the unsized field's alignment could be smaller.
            assert!(!t.is_simd());
            let layout = ccx.layout_of(t);
            debug!("DST {} layout: {:?}", t, layout);

            let (sized_size, sized_align) = match *layout {
                ty::layout::Layout::Univariant(ref variant) => {
                    (variant.offsets.last().map_or(0, |o| o.bytes()), variant.align.abi())
                }
                _ => {
                    bug!("size_and_align_of_dst: expcted Univariant for `{}`, found {:#?}",
                         t, layout);
                }
            };
            debug!("DST {} statically sized prefix size: {} align: {}",
                   t, sized_size, sized_align);
            let sized_size = C_usize(ccx, sized_size);
            let sized_align = C_usize(ccx, sized_align);

            // Recurse to get the size of the dynamically sized field (must be
            // the last field).
            let field_ty = layout.field(ccx, layout.field_count() - 1).ty;
            let (unsized_size, unsized_align) = size_and_align_of_dst(bcx, field_ty, info);

            // FIXME (#26403, #27023): We should be adding padding
            // to `sized_size` (to accommodate the `unsized_align`
            // required of the unsized field that follows) before
            // summing it with `sized_size`. (Note that since #26403
            // is unfixed, we do not yet add the necessary padding
            // here. But this is where the add would go.)

            // Return the sum of sizes and max of aligns.
            let size = bcx.add(sized_size, unsized_size);

            // Choose max of two known alignments (combined value must
            // be aligned according to more restrictive of the two).
            let align = match (const_to_opt_u128(sized_align, false),
                               const_to_opt_u128(unsized_align, false)) {
                (Some(sized_align), Some(unsized_align)) => {
                    // If both alignments are constant, (the sized_align should always be), then
                    // pick the correct alignment statically.
                    C_usize(ccx, std::cmp::max(sized_align, unsized_align) as u64)
                }
                _ => bcx.select(bcx.icmp(llvm::IntUGT, sized_align, unsized_align),
                                sized_align,
                                unsized_align)
            };

            // Issue #27023: must add any necessary padding to `size`
            // (to make it a multiple of `align`) before returning it.
            //
            // Namely, the returned size should be, in C notation:
            //
            //   `size + ((size & (align-1)) ? align : 0)`
            //
            // emulated via the semi-standard fast bit trick:
            //
            //   `(size + (align-1)) & -align`

            let addend = bcx.sub(align, C_usize(bcx.ccx, 1));
            let size = bcx.and(bcx.add(size, addend), bcx.neg(align));

            (size, align)
        }
    }
}
