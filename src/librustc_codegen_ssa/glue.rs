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

use common::IntPredicate;
use meth;
use rustc::ty::layout::{LayoutOf, TyLayout, HasTyCtxt};
use rustc::ty::{self, Ty};
use interfaces::*;

pub fn size_and_align_of_dst<'a, 'll: 'a, 'tcx: 'll,
    Bx: BuilderMethods<'a, 'll, 'tcx>
    >(
    bx: &mut Bx,
    t: Ty<'tcx>,
    info: Option<<Bx::CodegenCx as Backend<'ll>>::Value>
) -> (<Bx::CodegenCx as Backend<'ll>>::Value, <Bx::CodegenCx as Backend<'ll>>::Value)  where
    &'a Bx::CodegenCx: LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>> + HasTyCtxt<'tcx>
{
    debug!("calculate size of DST: {}; with lost info: {:?}",
           t, info);
    if bx.cx().type_is_sized(t) {
        let (size, align) = bx.cx().layout_of(t).size_and_align();
        debug!("size_and_align_of_dst t={} info={:?} size: {:?} align: {:?}",
               t, info, size, align);
        let size = bx.cx().const_usize(size.bytes());
        let align = bx.cx().const_usize(align.abi());
        return (size, align);
    }
    match t.sty {
        ty::Dynamic(..) => {
            // load size/align from vtable
            let vtable = info.unwrap();
            (meth::SIZE.get_usize(bx, vtable), meth::ALIGN.get_usize(bx, vtable))
        }
        ty::Slice(_) | ty::Str => {
            let unit = t.sequence_element_type(bx.tcx());
            // The info in this case is the length of the str, so the size is that
            // times the unit size.
            let (size, align) = bx.cx().layout_of(unit).size_and_align();
            (bx.mul(info.unwrap(), bx.cx().const_usize(size.bytes())),
             bx.cx().const_usize(align.abi()))
        }
        _ => {
            let cx = bx.cx();
            // First get the size of all statically known fields.
            // Don't use size_of because it also rounds up to alignment, which we
            // want to avoid, as the unsized field's alignment could be smaller.
            assert!(!t.is_simd());
            let layout = cx.layout_of(t);
            debug!("DST {} layout: {:?}", t, layout);

            let i = layout.fields.count() - 1;
            let sized_size = layout.fields.offset(i).bytes();
            let sized_align = layout.align.abi();
            debug!("DST {} statically sized prefix size: {} align: {}",
                   t, sized_size, sized_align);
            let sized_size = cx.const_usize(sized_size);
            let sized_align = cx.const_usize(sized_align);

            // Recurse to get the size of the dynamically sized field (must be
            // the last field).
            let field_ty = layout.field(cx, i).ty;
            let (unsized_size, mut unsized_align) = size_and_align_of_dst(bx, field_ty, info);

            // FIXME (#26403, #27023): We should be adding padding
            // to `sized_size` (to accommodate the `unsized_align`
            // required of the unsized field that follows) before
            // summing it with `sized_size`. (Note that since #26403
            // is unfixed, we do not yet add the necessary padding
            // here. But this is where the add would go.)

            // Return the sum of sizes and max of aligns.
            let size = bx.add(sized_size, unsized_size);

            // Packed types ignore the alignment of their fields.
            if let ty::Adt(def, _) = t.sty {
                if def.repr.packed() {
                    unsized_align = sized_align;
                }
            }

            // Choose max of two known alignments (combined value must
            // be aligned according to more restrictive of the two).
            let align = match (bx.cx().const_to_opt_u128(sized_align, false),
                               bx.cx().const_to_opt_u128(unsized_align, false)) {
                (Some(sized_align), Some(unsized_align)) => {
                    // If both alignments are constant, (the sized_align should always be), then
                    // pick the correct alignment statically.
                    cx.const_usize(std::cmp::max(sized_align, unsized_align) as u64)
                }
                _ => {
                    let cmp = bx.icmp(IntPredicate::IntUGT, sized_align, unsized_align);
                    bx.select(cmp, sized_align, unsized_align)
                }
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
            let one = bx.cx().const_usize(1);
            let addend = bx.sub(align, one);
            let add = bx.add(size, addend);
            let neg =  bx.neg(align);
            let size = bx.and(add, neg);

            (size, align)
        }
    }
}
