use std::any::Any;
use std::{fmt, iter};

use rustc_abi::Size;

use crate::*;

/// The byte that will be filled into the memory that backs an opaque meta object.
/// Can be basically anything, the only constraint is that we must be able to tell
/// it apart from static initializers to avoid allowing a destryed object to be
/// implicitly re-initialized. Therefore, this must *not* be 0x00.
pub const OPAQUE_OBJ_COOKIE: u8 = 0x42;

/// A trait for the synchronization metadata that can be attached to a memory location.
/// Must *not* contain any provenance!
pub trait MetaObj: Any {
    /// Determines whether reads/writes to this object's location are currently permitted.
    fn on_access<'tcx>(&self) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Determines whether this object is opaque. Opaque objects have their memory contents
    /// reset to "uninit" on creation, and are implicitly removed on writes.
    fn opaque(&self) -> bool;
}

impl dyn MetaObj {
    #[inline(always)]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        let x: &dyn Any = self;
        x.downcast_ref()
    }

    #[inline(always)]
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        let x: &mut dyn Any = self;
        x.downcast_mut()
    }

    #[inline(always)]
    pub fn downcast<T: Any>(self: Box<Self>) -> Option<Box<T>> {
        let x: Box<dyn Any> = self;
        x.downcast().ok()
    }
}

impl fmt::Debug for dyn MetaObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetaObj").finish_non_exhaustive()
    }
}

impl<'tcx> AllocExtra<'tcx> {
    /// Get the meta object that starts at the given offset.
    fn get_meta<T: MetaObj>(&self, offset: Size) -> Option<&T> {
        self.meta_objs.get_start(offset).and_then(|s| s.downcast_ref::<T>())
    }

    /// Get the meta object that starts at the given offset.
    fn get_meta_mut<T: MetaObj>(&mut self, offset: Size) -> Option<&mut T> {
        self.meta_objs.get_start_mut(offset).and_then(|s| s.downcast_mut::<T>())
    }

    /// Insert a new meta object covering the given range. Any other objects that exist
    /// in that range are removed.
    fn insert_meta(&mut self, range: AllocRange, obj: Box<dyn MetaObj>) {
        use data_structures::range_object_map::AccessType;
        let pos = match self.meta_objs.access_type(range) {
            AccessType::PerfectlyOverlapping(pos) => {
                self.meta_objs.remove_from_pos(pos);
                pos
            }
            AccessType::Empty(pos) => pos,
            AccessType::ImperfectlyOverlapping(pos_range) => {
                self.meta_objs.remove_pos_range(pos_range.clone());
                pos_range.start
            }
        };
        self.meta_objs.insert_at_pos(pos, range, obj);
    }

    fn remove_meta(&mut self, range: AllocRange) -> Option<Box<dyn MetaObj>> {
        use data_structures::range_object_map::AccessType;
        match self.meta_objs.access_type(range) {
            AccessType::PerfectlyOverlapping(pos) => Some(self.meta_objs.remove_from_pos(pos)),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum GetMetaError {
    /// There is no memory behind this pointer.
    NoMem,
    /// There is no object stored at this location.
    NoObj,
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Get the meta object that starts at the given offset.
    fn try_get_meta<'a, T: MetaObj>(&'a self, ptr: Pointer) -> Result<&'a T, GetMetaError>
    where
        'tcx: 'a,
    {
        let ecx = self.eval_context_ref();
        let Ok((alloc_id, offset, ..)) = ecx.ptr_try_get_alloc_id(ptr, 0) else {
            return Err(GetMetaError::NoMem);
        };
        let info = ecx.get_alloc_info(alloc_id);
        if info.kind != AllocKind::LiveData || offset >= info.size {
            return Err(GetMetaError::NoMem);
        }
        // This cannot fail now.
        let alloc_extra = ecx.get_alloc_extra(alloc_id).unwrap();
        alloc_extra.get_meta::<T>(offset).ok_or(GetMetaError::NoObj)
    }

    fn insert_meta(&mut self, place: &MPlaceTy<'tcx>, obj: Box<dyn MetaObj>) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();
        assert!(place.layout.size > Size::ZERO);

        // Ensure there is memory here, and reset it for opaque objects.
        // We reset to something initialized so that e.g. a typed copy of a `pthread_mutex_t`
        // does not raise UB due to an invalid value.
        if obj.opaque() {
            ecx.write_bytes_ptr(
                place.ptr(),
                iter::repeat_n(OPAQUE_OBJ_COOKIE, place.layout.size.bytes_usize()),
            )?;
        } else {
            ecx.check_ptr_access(place.ptr(), place.layout.size, CheckInAllocMsg::MemoryAccess)?;
        }

        // Now we can insert the object.
        let (alloc_id, offset, ..) = ecx.ptr_try_get_alloc_id(place.ptr(), 0).unwrap();
        let (alloc_extra, _machine) = ecx.get_alloc_extra_mut(alloc_id)?;
        alloc_extra.insert_meta(AllocRange { start: offset, size: place.layout.size }, obj);
        interp_ok(())
    }

    fn get_or_insert_meta<'a, T: MetaObj>(
        &'a mut self,
        place: &MPlaceTy<'tcx>,
        obj: impl FnOnce(&mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, T>,
    ) -> InterpResult<'tcx, &'a mut T>
    where
        'tcx: 'a,
    {
        let ecx = self.eval_context_mut();
        let (alloc_id, offset, ..) = ecx.ptr_get_alloc_id(place.ptr(), 0)?;
        // Sadly, without Polonius, we cannot reuse `get_meta`, and we do some double-lookups.
        let alloc_extra = ecx.get_alloc_extra(alloc_id)?;
        if alloc_extra.get_meta::<T>(offset).is_none() {
            // No object yet. Create one.
            let obj = obj(ecx)?;
            ecx.insert_meta(place, Box::new(obj))?;
        }
        // Now there is an object.
        let (alloc_extra, _machine) = ecx.get_alloc_extra_mut(alloc_id).unwrap();
        interp_ok(alloc_extra.get_meta_mut(offset).unwrap())
    }

    fn remove_meta<'a, T: MetaObj>(
        &'a mut self,
        place: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, Option<Box<T>>> {
        let ecx = self.eval_context_mut();
        let (alloc_id, offset, ..) = ecx.ptr_get_alloc_id(place.ptr(), 0)?;
        let (alloc_extra, _machine) = ecx.get_alloc_extra_mut(alloc_id).unwrap();

        interp_ok(
            alloc_extra
                .remove_meta(AllocRange { start: offset, size: place.layout.size })
                .and_then(|obj| obj.downcast()),
        )
    }
}
