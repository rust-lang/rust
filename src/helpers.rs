use std::mem;

use rustc::ty::{self, layout::{self, Size}};
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};

use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}

pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Gets an instance for a path.
    fn resolve_path(&self, path: &[&str]) -> InterpResult<'tcx, ty::Instance<'tcx>> {
        let this = self.eval_context_ref();
        this.tcx
            .crates()
            .iter()
            .find(|&&krate| this.tcx.original_crate_name(krate).as_str() == path[0])
            .and_then(|krate| {
                let krate = DefId {
                    krate: *krate,
                    index: CRATE_DEF_INDEX,
                };
                let mut items = this.tcx.item_children(krate);
                let mut path_it = path.iter().skip(1).peekable();

                while let Some(segment) = path_it.next() {
                    for item in mem::replace(&mut items, Default::default()).iter() {
                        if item.ident.name.as_str() == *segment {
                            if path_it.peek().is_none() {
                                return Some(ty::Instance::mono(this.tcx.tcx, item.res.def_id()));
                            }

                            items = this.tcx.item_children(item.res.def_id());
                            break;
                        }
                    }
                }
                None
            })
            .ok_or_else(|| {
                let path = path.iter().map(|&s| s.to_owned()).collect();
                InterpError::PathNotFound(path).into()
            })
    }

    /// Visits the memory covered by `place`, sensitive to freezing: the 3rd parameter
    /// will be true if this is frozen, false if this is in an `UnsafeCell`.
    fn visit_freeze_sensitive(
        &self,
        place: MPlaceTy<'tcx, Tag>,
        size: Size,
        mut action: impl FnMut(Pointer<Tag>, Size, bool) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        trace!("visit_frozen(place={:?}, size={:?})", *place, size);
        debug_assert_eq!(size,
            this.size_and_align_of_mplace(place)?
            .map(|(size, _)| size)
            .unwrap_or_else(|| place.layout.size)
        );
        // Store how far we proceeded into the place so far. Everything to the left of
        // this offset has already been handled, in the sense that the frozen parts
        // have had `action` called on them.
        let mut end_ptr = place.ptr;
        // Called when we detected an `UnsafeCell` at the given offset and size.
        // Calls `action` and advances `end_ptr`.
        let mut unsafe_cell_action = |unsafe_cell_ptr: Scalar<Tag>, unsafe_cell_size: Size| {
            if unsafe_cell_size != Size::ZERO {
                debug_assert_eq!(unsafe_cell_ptr.to_ptr().unwrap().alloc_id,
                    end_ptr.to_ptr().unwrap().alloc_id);
                debug_assert_eq!(unsafe_cell_ptr.to_ptr().unwrap().tag,
                    end_ptr.to_ptr().unwrap().tag);
            }
            // We assume that we are given the fields in increasing offset order,
            // and nothing else changes.
            let unsafe_cell_offset = unsafe_cell_ptr.get_ptr_offset(this);
            let end_offset = end_ptr.get_ptr_offset(this);
            assert!(unsafe_cell_offset >= end_offset);
            let frozen_size = unsafe_cell_offset - end_offset;
            // Everything between the end_ptr and this `UnsafeCell` is frozen.
            if frozen_size != Size::ZERO {
                action(end_ptr.to_ptr()?, frozen_size, /*frozen*/true)?;
            }
            // This `UnsafeCell` is NOT frozen.
            if unsafe_cell_size != Size::ZERO {
                action(unsafe_cell_ptr.to_ptr()?, unsafe_cell_size, /*frozen*/false)?;
            }
            // Update end end_ptr.
            end_ptr = unsafe_cell_ptr.ptr_wrapping_offset(unsafe_cell_size, this);
            // Done
            Ok(())
        };
        // Run a visitor
        {
            let mut visitor = UnsafeCellVisitor {
                ecx: this,
                unsafe_cell_action: |place| {
                    trace!("unsafe_cell_action on {:?}", place.ptr);
                    // We need a size to go on.
                    let unsafe_cell_size = this.size_and_align_of_mplace(place)?
                        .map(|(size, _)| size)
                        // for extern types, just cover what we can
                        .unwrap_or_else(|| place.layout.size);
                    // Now handle this `UnsafeCell`, unless it is empty.
                    if unsafe_cell_size != Size::ZERO {
                        unsafe_cell_action(place.ptr, unsafe_cell_size)
                    } else {
                        Ok(())
                    }
                },
            };
            visitor.visit_value(place)?;
        }
        // The part between the end_ptr and the end of the place is also frozen.
        // So pretend there is a 0-sized `UnsafeCell` at the end.
        unsafe_cell_action(place.ptr.ptr_wrapping_offset(size, this), Size::ZERO)?;
        // Done!
        return Ok(());

        /// Visiting the memory covered by a `MemPlace`, being aware of
        /// whether we are inside an `UnsafeCell` or not.
        struct UnsafeCellVisitor<'ecx, 'mir, 'tcx, F>
            where F: FnMut(MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx>
        {
            ecx: &'ecx MiriEvalContext<'mir, 'tcx>,
            unsafe_cell_action: F,
        }

        impl<'ecx, 'mir, 'tcx, F>
            ValueVisitor<'mir, 'tcx, Evaluator<'tcx>>
        for
            UnsafeCellVisitor<'ecx, 'mir, 'tcx, F>
        where
            F: FnMut(MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx>
        {
            type V = MPlaceTy<'tcx, Tag>;

            #[inline(always)]
            fn ecx(&self) -> &MiriEvalContext<'mir, 'tcx> {
                &self.ecx
            }

            // Hook to detect `UnsafeCell`.
            fn visit_value(&mut self, v: MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx>
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

            // Make sure we visit aggregrates in increasing offset order.
            fn visit_aggregate(
                &mut self,
                place: MPlaceTy<'tcx, Tag>,
                fields: impl Iterator<Item=InterpResult<'tcx, MPlaceTy<'tcx, Tag>>>,
            ) -> InterpResult<'tcx> {
                match place.layout.fields {
                    layout::FieldPlacement::Array { .. } => {
                        // For the array layout, we know the iterator will yield sorted elements so
                        // we can avoid the allocation.
                        self.walk_aggregate(place, fields)
                    }
                    layout::FieldPlacement::Arbitrary { .. } => {
                        // Gather the subplaces and sort them before visiting.
                        let mut places = fields.collect::<InterpResult<'tcx, Vec<MPlaceTy<'tcx, Tag>>>>()?;
                        places.sort_by_key(|place| place.ptr.get_ptr_offset(self.ecx()));
                        self.walk_aggregate(place, places.into_iter().map(Ok))
                    }
                    layout::FieldPlacement::Union { .. } => {
                        // Uh, what?
                        bug!("a union is not an aggregate we should ever visit")
                    }
                }
            }

            // We have to do *something* for unions.
            fn visit_union(&mut self, v: MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx>
            {
                // With unions, we fall back to whatever the type says, to hopefully be consistent
                // with LLVM IR.
                // FIXME: are we consistent, and is this really the behavior we want?
                let frozen = self.ecx.type_is_freeze(v.layout.ty);
                if frozen {
                    Ok(())
                } else {
                    (self.unsafe_cell_action)(v)
                }
            }

            // We should never get to a primitive, but always short-circuit somewhere above.
            fn visit_primitive(&mut self, _v: MPlaceTy<'tcx, Tag>) -> InterpResult<'tcx>
            {
                bug!("we should always short-circuit before coming to a primitive")
            }
        }
    }
}
