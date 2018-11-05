use std::mem;

use rustc::ty;
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};

use crate::*;

pub trait ScalarExt {
    /// HACK: this function just extracts all bits if `defined != 0`
    /// Mainly used for args of C-functions and we should totally correctly fetch the size
    /// of their arguments
    fn to_bytes(self) -> EvalResult<'static, u128>;
}

impl<Tag> ScalarExt for Scalar<Tag> {
    fn to_bytes(self) -> EvalResult<'static, u128> {
        match self {
            Scalar::Bits { bits, size } => {
                assert_ne!(size, 0);
                Ok(bits)
            },
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
        }
    }
}

impl<Tag> ScalarExt for ScalarMaybeUndef<Tag> {
    fn to_bytes(self) -> EvalResult<'static, u128> {
        self.not_undef()?.to_bytes()
    }
}

pub trait EvalContextExt<'tcx> {
    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>>;

    /// Visit the memory covered by `place` that is frozen -- i.e., NOT
    /// what is inside an `UnsafeCell`.
    fn visit_frozen(
        &self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        action: impl FnMut(Pointer<Borrow>, Size) -> EvalResult<'tcx>,
    ) -> EvalResult<'tcx>;
}


impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    /// Get an instance for a path.
    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        self.tcx
            .crates()
            .iter()
            .find(|&&krate| self.tcx.original_crate_name(krate) == path[0])
            .and_then(|krate| {
                let krate = DefId {
                    krate: *krate,
                    index: CRATE_DEF_INDEX,
                };
                let mut items = self.tcx.item_children(krate);
                let mut path_it = path.iter().skip(1).peekable();

                while let Some(segment) = path_it.next() {
                    for item in mem::replace(&mut items, Default::default()).iter() {
                        if item.ident.name == *segment {
                            if path_it.peek().is_none() {
                                return Some(ty::Instance::mono(self.tcx.tcx, item.def.def_id()));
                            }

                            items = self.tcx.item_children(item.def.def_id());
                            break;
                        }
                    }
                }
                None
            })
            .ok_or_else(|| {
                let path = path.iter().map(|&s| s.to_owned()).collect();
                EvalErrorKind::PathNotFound(path).into()
            })
    }

    /// Visit the memory covered by `place` that is frozen -- i.e., NOT
    /// what is inside an `UnsafeCell`.
    fn visit_frozen(
        &self,
        place: MPlaceTy<'tcx, Borrow>,
        size: Size,
        mut frozen_action: impl FnMut(Pointer<Borrow>, Size) -> EvalResult<'tcx>,
    ) -> EvalResult<'tcx> {
        trace!("visit_frozen(place={:?}, size={:?})", *place, size);
        debug_assert_eq!(size,
            self.size_and_align_of_mplace(place)?
            .map(|(size, _)| size)
            .unwrap_or_else(|| place.layout.size)
        );
        // Store how far we proceeded into the place so far.  Everything to the left of
        // this offset has already been handled, in the sense that the frozen parts
        // have had `action` called on them.
        let mut end_ptr = place.ptr;
        // Called when we detected an `UnsafeCell` at the given offset and size.
        // Calls `action` and advances `end_ptr`.
        let mut unsafe_cell_action = |unsafe_cell_offset, unsafe_cell_size| {
            // We assume that we are given the fields in increasing offset order,
            // and nothing else changes.
            let end_offset = end_ptr.get_ptr_offset(self);
            assert!(unsafe_cell_offset >= end_offset);
            let frozen_size = unsafe_cell_offset - end_offset;
            // Everything between the end_ptr and this `UnsafeCell` is frozen.
            if frozen_size != Size::ZERO {
                frozen_action(end_ptr.to_ptr()?, frozen_size)?;
            }
            // Update end end_ptr.
            end_ptr = end_ptr.ptr_wrapping_offset(frozen_size+unsafe_cell_size, self);
            // Done
            Ok(())
        };
        // Run a visitor
        {
            let mut visitor = UnsafeCellVisitor {
                ecx: self,
                unsafe_cell_action: |place| {
                    trace!("unsafe_cell_action on {:?}", place.ptr);
                    // We need a size to go on.
                    let (unsafe_cell_size, _) = self.size_and_align_of_mplace(place)?
                        // for extern types, just cover what we can
                        .unwrap_or_else(|| place.layout.size_and_align());
                    // Now handle this `UnsafeCell`.
                    unsafe_cell_action(place.ptr.get_ptr_offset(self), unsafe_cell_size)
                },
            };
            visitor.visit_value(place)?;
        }
        // The part between the end_ptr and the end of the place is also frozen.
        // So pretend there is a 0-sized `UnsafeCell` at the end.
        unsafe_cell_action(place.ptr.get_ptr_offset(self) + size, Size::ZERO)?;
        // Done!
        return Ok(());

        /// Visiting the memory covered by a `MemPlace`, being aware of
        /// whether we are inside an `UnsafeCell` or not.
        struct UnsafeCellVisitor<'ecx, 'a, 'mir, 'tcx, F>
            where F: FnMut(MPlaceTy<'tcx, Borrow>) -> EvalResult<'tcx>
        {
            ecx: &'ecx MiriEvalContext<'a, 'mir, 'tcx>,
            unsafe_cell_action: F,
        }

        impl<'ecx, 'a, 'mir, 'tcx, F> ValueVisitor<'a, 'mir, 'tcx, Evaluator<'tcx>>
        for UnsafeCellVisitor<'ecx, 'a, 'mir, 'tcx, F>
        where
            F: FnMut(MPlaceTy<'tcx, Borrow>) -> EvalResult<'tcx>
        {
            type V = MPlaceTy<'tcx, Borrow>;

            const WANT_FIELDS_SORTED: bool = true; // sorted? yes please!

            #[inline(always)]
            fn ecx(&self) -> &MiriEvalContext<'a, 'mir, 'tcx> {
                &self.ecx
            }

            // Hook to detect `UnsafeCell`
            fn visit_value(&mut self, v: MPlaceTy<'tcx, Borrow>) -> EvalResult<'tcx>
            {
                trace!("UnsafeCellVisitor: {:?} {:?}", *v, v.layout.ty);
                let is_unsafe_cell = match v.layout.ty.sty {
                    ty::Adt(adt, _) => Some(adt.did) == self.ecx.tcx.lang_items().unsafe_cell_type(),
                    _ => false,
                };
                if is_unsafe_cell {
                    // We do not have to recurse further, this is an `UnsafeCell`.
                    (self.unsafe_cell_action)(v)
                } else if self.ecx.type_is_freeze(v.layout.ty) {
                    // This is `Freeze`, there cannot be an `UnsafeCell`
                    Ok(())
                } else {
                    // Proceed further
                    self.walk_value(v)
                }
            }

            // We have to do *something* for unions
            fn visit_union(&mut self, v: MPlaceTy<'tcx, Borrow>) -> EvalResult<'tcx>
            {
                // With unions, we fall back to whatever the type says, to hopefully be consistent
                // with LLVM IR.
                // FIXME Are we consistent?  And is this really the behavior we want?
                let frozen = self.ecx.type_is_freeze(v.layout.ty);
                if frozen {
                    Ok(())
                } else {
                    (self.unsafe_cell_action)(v)
                }
            }

            // We should never get to a primitive, but always short-circuit somewhere above
            fn visit_primitive(&mut self, _val: ImmTy<'tcx, Borrow>) -> EvalResult<'tcx>
            {
                bug!("We should always short-circit before coming to a primitive")
            }
        }
    }
}
