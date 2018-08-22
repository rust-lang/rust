// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::BasicBlock;
use rustc::ty::{self, layout::LayoutOf};
use syntax::source_map::Span;

use rustc::mir::interpret::EvalResult;
use interpret::{Machine, EvalContext, PlaceTy, PlaceExtra, OpTy, Operand};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub(crate) fn drop_in_place(
        &mut self,
        place: PlaceTy<'tcx>,
        instance: ty::Instance<'tcx>,
        span: Span,
        target: BasicBlock,
    ) -> EvalResult<'tcx> {
        trace!("drop_in_place: {:?},\n  {:?}, {:?}", *place, place.layout.ty, instance);
        // We take the address of the object.  This may well be unaligned, which is fine for us
        // here. However, unaligned accesses will probably make the actual drop implementation fail
        // -- a problem shared by rustc.
        let place = self.force_allocation(place)?;

        let (instance, place) = match place.layout.ty.sty {
            ty::Dynamic(..) => {
                // Dropping a trait object.
                let vtable = match place.extra {
                    PlaceExtra::Vtable(vtable) => vtable,
                    _ => bug!("Expected vtable when dropping {:#?}", place),
                };
                let place = self.unpack_unsized_mplace(place)?;
                let instance = self.read_drop_type_from_vtable(vtable)?;
                (instance, place)
            }
            _ => (instance, place),
        };

        let fn_sig = instance.ty(*self.tcx).fn_sig(*self.tcx);
        let fn_sig = self.tcx.normalize_erasing_late_bound_regions(self.param_env, &fn_sig);

        let arg = OpTy {
            op: Operand::Immediate(place.to_ref(&self)),
            layout: self.layout_of(self.tcx.mk_mut_ptr(place.layout.ty))?,
        };

        // This should always be (), but getting it from the sig seems
        // easier than creating a layout of ().
        let dest = PlaceTy::null(&self, self.layout_of(fn_sig.output())?);

        self.eval_fn_call(
            instance,
            Some((dest, target)),
            &[arg],
            span,
            fn_sig,
        )
    }
}
