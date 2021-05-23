//! The memory subsystem.
//!
//! Generally, we use `Pointer` to denote memory addresses. However, some operations
//! have a "size"-like parameter, and they take `Scalar` for the address because
//! if the size is 0, then the pointer can also be a (properly aligned, non-null)
//! integer. It is crucial that these operations call `check_align` *before*
//! short-circuiting the empty case!

use std::borrow::Cow;
use std::collections::VecDeque;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ptr;

use rustc_ast::Mutability;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::ty::{Instance, ParamEnv, TyCtxt};
use rustc_target::abi::{Align, HasDataLayout, Size, TargetDataLayout};

use super::{
    alloc_range, AllocId, AllocMap, AllocRange, Allocation, CheckInAllocMsg, GlobalAlloc,
    InterpResult, Machine, MayLeak, Pointer, PointerArithmetic, Scalar, ScalarMaybeUninit,
};
use crate::util::pretty;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum MemoryKind<T> {
    /// Stack memory. Error if deallocated except during a stack pop.
    Stack,
    /// Memory backing vtables. Error if ever deallocated.
    Vtable,
    /// Memory allocated by `caller_location` intrinsic. Error if ever deallocated.
    CallerLocation,
    /// Additional memory kinds a machine wishes to distinguish from the builtin ones.
    Machine(T),
}

impl<T: MayLeak> MayLeak for MemoryKind<T> {
    #[inline]
    fn may_leak(self) -> bool {
        match self {
            MemoryKind::Stack => false,
            MemoryKind::Vtable => true,
            MemoryKind::CallerLocation => true,
            MemoryKind::Machine(k) => k.may_leak(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for MemoryKind<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryKind::Stack => write!(f, "stack variable"),
            MemoryKind::Vtable => write!(f, "vtable"),
            MemoryKind::CallerLocation => write!(f, "caller location"),
            MemoryKind::Machine(m) => write!(f, "{}", m),
        }
    }
}

/// Used by `get_size_and_align` to indicate whether the allocation needs to be live.
#[derive(Debug, Copy, Clone)]
pub enum AllocCheck {
    /// Allocation must be live and not a function pointer.
    Dereferenceable,
    /// Allocations needs to be live, but may be a function pointer.
    Live,
    /// Allocation may be dead.
    MaybeDead,
}

/// The value of a function pointer.
#[derive(Debug, Copy, Clone)]
pub enum FnVal<'tcx, Other> {
    Instance(Instance<'tcx>),
    Other(Other),
}

impl<'tcx, Other> FnVal<'tcx, Other> {
    pub fn as_instance(self) -> InterpResult<'tcx, Instance<'tcx>> {
        match self {
            FnVal::Instance(instance) => Ok(instance),
            FnVal::Other(_) => {
                throw_unsup_format!("'foreign' function pointers are not supported in this context")
            }
        }
    }
}

// `Memory` has to depend on the `Machine` because some of its operations
// (e.g., `get`) call a `Machine` hook.
pub struct Memory<'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    /// Allocations local to this instance of the miri engine. The kind
    /// helps ensure that the same mechanism is used for allocation and
    /// deallocation. When an allocation is not found here, it is a
    /// global and looked up in the `tcx` for read access. Some machines may
    /// have to mutate this map even on a read-only access to a global (because
    /// they do pointer provenance tracking and the allocations in `tcx` have
    /// the wrong type), so we let the machine override this type.
    /// Either way, if the machine allows writing to a global, doing so will
    /// create a copy of the global allocation here.
    // FIXME: this should not be public, but interning currently needs access to it
    pub(super) alloc_map: M::MemoryMap,

    /// Map for "extra" function pointers.
    extra_fn_ptr_map: FxHashMap<AllocId, M::ExtraFnVal>,

    /// To be able to compare pointers with null, and to check alignment for accesses
    /// to ZSTs (where pointers may dangle), we keep track of the size even for allocations
    /// that do not exist any more.
    // FIXME: this should not be public, but interning currently needs access to it
    pub(super) dead_alloc_map: FxHashMap<AllocId, (Size, Align)>,

    /// Extra data added by the machine.
    pub extra: M::MemoryExtra,

    /// Lets us implement `HasDataLayout`, which is awfully convenient.
    pub tcx: TyCtxt<'tcx>,
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> HasDataLayout for Memory<'mir, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

/// A reference to some allocation that was already bounds-checked for the given region
/// and had the on-access machine hooks run.
#[derive(Copy, Clone)]
pub struct AllocRef<'a, 'tcx, Tag, Extra> {
    alloc: &'a Allocation<Tag, Extra>,
    range: AllocRange,
    tcx: TyCtxt<'tcx>,
    alloc_id: AllocId,
}
/// A reference to some allocation that was already bounds-checked for the given region
/// and had the on-access machine hooks run.
pub struct AllocRefMut<'a, 'tcx, Tag, Extra> {
    alloc: &'a mut Allocation<Tag, Extra>,
    range: AllocRange,
    tcx: TyCtxt<'tcx>,
    alloc_id: AllocId,
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    pub fn new(tcx: TyCtxt<'tcx>, extra: M::MemoryExtra) -> Self {
        Memory {
            alloc_map: M::MemoryMap::default(),
            extra_fn_ptr_map: FxHashMap::default(),
            dead_alloc_map: FxHashMap::default(),
            extra,
            tcx,
        }
    }

    /// Call this to turn untagged "global" pointers (obtained via `tcx`) into
    /// the machine pointer to the allocation.  Must never be used
    /// for any other pointers, nor for TLS statics.
    ///
    /// Using the resulting pointer represents a *direct* access to that memory
    /// (e.g. by directly using a `static`),
    /// as opposed to access through a pointer that was created by the program.
    ///
    /// This function can fail only if `ptr` points to an `extern static`.
    #[inline]
    pub fn global_base_pointer(
        &self,
        mut ptr: Pointer,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        // We need to handle `extern static`.
        let ptr = match self.tcx.get_global_alloc(ptr.alloc_id) {
            Some(GlobalAlloc::Static(def_id)) if self.tcx.is_thread_local_static(def_id) => {
                bug!("global memory cannot point to thread-local static")
            }
            Some(GlobalAlloc::Static(def_id)) if self.tcx.is_foreign_item(def_id) => {
                ptr.alloc_id = M::extern_static_alloc_id(self, def_id)?;
                ptr
            }
            _ => {
                // No need to change the `AllocId`.
                ptr
            }
        };
        // And we need to get the tag.
        let tag = M::tag_global_base_pointer(&self.extra, ptr.alloc_id);
        Ok(ptr.with_tag(tag))
    }

    pub fn create_fn_alloc(
        &mut self,
        fn_val: FnVal<'tcx, M::ExtraFnVal>,
    ) -> Pointer<M::PointerTag> {
        let id = match fn_val {
            FnVal::Instance(instance) => self.tcx.create_fn_alloc(instance),
            FnVal::Other(extra) => {
                // FIXME(RalfJung): Should we have a cache here?
                let id = self.tcx.reserve_alloc_id();
                let old = self.extra_fn_ptr_map.insert(id, extra);
                assert!(old.is_none());
                id
            }
        };
        // Functions are global allocations, so make sure we get the right base pointer.
        // We know this is not an `extern static` so this cannot fail.
        self.global_base_pointer(Pointer::from(id)).unwrap()
    }

    pub fn allocate(
        &mut self,
        size: Size,
        align: Align,
        kind: MemoryKind<M::MemoryKind>,
    ) -> Pointer<M::PointerTag> {
        let alloc = Allocation::uninit(size, align);
        self.allocate_with(alloc, kind)
    }

    pub fn allocate_bytes(
        &mut self,
        bytes: &[u8],
        align: Align,
        kind: MemoryKind<M::MemoryKind>,
        mutability: Mutability,
    ) -> Pointer<M::PointerTag> {
        let alloc = Allocation::from_bytes(bytes, align, mutability);
        self.allocate_with(alloc, kind)
    }

    pub fn allocate_with(
        &mut self,
        alloc: Allocation,
        kind: MemoryKind<M::MemoryKind>,
    ) -> Pointer<M::PointerTag> {
        let id = self.tcx.reserve_alloc_id();
        debug_assert_ne!(
            Some(kind),
            M::GLOBAL_KIND.map(MemoryKind::Machine),
            "dynamically allocating global memory"
        );
        // This is a new allocation, not a new global one, so no `global_base_ptr`.
        let (alloc, tag) = M::init_allocation_extra(&self.extra, id, Cow::Owned(alloc), Some(kind));
        self.alloc_map.insert(id, (kind, alloc.into_owned()));
        Pointer::from(id).with_tag(tag)
    }

    pub fn reallocate(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        old_size_and_align: Option<(Size, Align)>,
        new_size: Size,
        new_align: Align,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        if ptr.offset.bytes() != 0 {
            throw_ub_format!(
                "reallocating {:?} which does not point to the beginning of an object",
                ptr
            );
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc".
        // This happens so rarely, the perf advantage is outweighed by the maintenance cost.
        let new_ptr = self.allocate(new_size, new_align, kind);
        let old_size = match old_size_and_align {
            Some((size, _align)) => size,
            None => self.get_raw(ptr.alloc_id)?.size(),
        };
        // This will also call the access hooks.
        self.copy(
            ptr.into(),
            Align::ONE,
            new_ptr.into(),
            Align::ONE,
            old_size.min(new_size),
            /*nonoverlapping*/ true,
        )?;
        self.deallocate(ptr, old_size_and_align, kind)?;

        Ok(new_ptr)
    }

    /// Deallocate a local, or do nothing if that local has been made into a global.
    pub fn deallocate_local(&mut self, ptr: Pointer<M::PointerTag>) -> InterpResult<'tcx> {
        // The allocation might be already removed by global interning.
        // This can only really happen in the CTFE instance, not in miri.
        if self.alloc_map.contains_key(&ptr.alloc_id) {
            self.deallocate(ptr, None, MemoryKind::Stack)
        } else {
            Ok(())
        }
    }

    pub fn deallocate(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        old_size_and_align: Option<(Size, Align)>,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx> {
        trace!("deallocating: {}", ptr.alloc_id);

        if ptr.offset.bytes() != 0 {
            throw_ub_format!(
                "deallocating {:?} which does not point to the beginning of an object",
                ptr
            );
        }

        let (alloc_kind, mut alloc) = match self.alloc_map.remove(&ptr.alloc_id) {
            Some(alloc) => alloc,
            None => {
                // Deallocating global memory -- always an error
                return Err(match self.tcx.get_global_alloc(ptr.alloc_id) {
                    Some(GlobalAlloc::Function(..)) => {
                        err_ub_format!("deallocating {}, which is a function", ptr.alloc_id)
                    }
                    Some(GlobalAlloc::Static(..) | GlobalAlloc::Memory(..)) => {
                        err_ub_format!("deallocating {}, which is static memory", ptr.alloc_id)
                    }
                    None => err_ub!(PointerUseAfterFree(ptr.alloc_id)),
                }
                .into());
            }
        };

        if alloc.mutability == Mutability::Not {
            throw_ub_format!("deallocating immutable allocation {}", ptr.alloc_id);
        }
        if alloc_kind != kind {
            throw_ub_format!(
                "deallocating {}, which is {} memory, using {} deallocation operation",
                ptr.alloc_id,
                alloc_kind,
                kind
            );
        }
        if let Some((size, align)) = old_size_and_align {
            if size != alloc.size() || align != alloc.align {
                throw_ub_format!(
                    "incorrect layout on deallocation: {} has size {} and alignment {}, but gave size {} and alignment {}",
                    ptr.alloc_id,
                    alloc.size().bytes(),
                    alloc.align.bytes(),
                    size.bytes(),
                    align.bytes(),
                )
            }
        }

        // Let the machine take some extra action
        let size = alloc.size();
        M::memory_deallocated(&mut self.extra, &mut alloc.extra, ptr, size)?;

        // Don't forget to remember size and align of this now-dead allocation
        let old = self.dead_alloc_map.insert(ptr.alloc_id, (size, alloc.align));
        if old.is_some() {
            bug!("Nothing can be deallocated twice");
        }

        Ok(())
    }

    /// Internal helper function for APIs that offer memory access based on `Scalar` pointers.
    #[inline(always)]
    pub(super) fn check_ptr_access(
        &self,
        sptr: Scalar<M::PointerTag>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Option<Pointer<M::PointerTag>>> {
        let align = M::enforce_alignment(&self.extra).then_some(align);
        self.check_and_deref_ptr(sptr, size, align, CheckInAllocMsg::MemoryAccessTest, |ptr| {
            let (size, align) =
                self.get_size_and_align(ptr.alloc_id, AllocCheck::Dereferenceable)?;
            Ok((size, align, ptr))
        })
    }

    /// Check if the given scalar is allowed to do a memory access of given `size` and `align`
    /// (ignoring `M::enforce_alignment`). The caller can control the error message for the
    /// out-of-bounds case.
    #[inline(always)]
    pub fn check_ptr_access_align(
        &self,
        sptr: Scalar<M::PointerTag>,
        size: Size,
        align: Align,
        msg: CheckInAllocMsg,
    ) -> InterpResult<'tcx> {
        self.check_and_deref_ptr(sptr, size, Some(align), msg, |ptr| {
            let (size, align) =
                self.get_size_and_align(ptr.alloc_id, AllocCheck::Dereferenceable)?;
            Ok((size, align, ()))
        })?;
        Ok(())
    }

    /// Low-level helper function to check if a ptr is in-bounds and potentially return a reference
    /// to the allocation it points to. Supports both shared and mutable references, to the actual
    /// checking is offloaded to a helper closure. `align` defines whether and which alignment check
    /// is done. Returns `None` for size 0, and otherwise `Some` of what `alloc_size` returned.
    fn check_and_deref_ptr<T>(
        &self,
        sptr: Scalar<M::PointerTag>,
        size: Size,
        align: Option<Align>,
        msg: CheckInAllocMsg,
        alloc_size: impl FnOnce(Pointer<M::PointerTag>) -> InterpResult<'tcx, (Size, Align, T)>,
    ) -> InterpResult<'tcx, Option<T>> {
        fn check_offset_align(offset: u64, align: Align) -> InterpResult<'static> {
            if offset % align.bytes() == 0 {
                Ok(())
            } else {
                // The biggest power of two through which `offset` is divisible.
                let offset_pow2 = 1 << offset.trailing_zeros();
                throw_ub!(AlignmentCheckFailed {
                    has: Align::from_bytes(offset_pow2).unwrap(),
                    required: align,
                })
            }
        }

        // Normalize to a `Pointer` if we definitely need one.
        let normalized = if size.bytes() == 0 {
            // Can be an integer, just take what we got.  We do NOT `force_bits` here;
            // if this is already a `Pointer` we want to do the bounds checks!
            sptr
        } else {
            // A "real" access, we must get a pointer to be able to check the bounds.
            Scalar::from(self.force_ptr(sptr)?)
        };
        Ok(match normalized.to_bits_or_ptr(self.pointer_size(), self) {
            Ok(bits) => {
                let bits = u64::try_from(bits).unwrap(); // it's ptr-sized
                assert!(size.bytes() == 0);
                // Must be non-null.
                if bits == 0 {
                    throw_ub!(DanglingIntPointer(0, msg))
                }
                // Must be aligned.
                if let Some(align) = align {
                    check_offset_align(bits, align)?;
                }
                None
            }
            Err(ptr) => {
                let (allocation_size, alloc_align, ret_val) = alloc_size(ptr)?;
                // Test bounds. This also ensures non-null.
                // It is sufficient to check this for the end pointer. The addition
                // checks for overflow.
                let end_ptr = ptr.offset(size, self)?;
                if end_ptr.offset > allocation_size {
                    // equal is okay!
                    throw_ub!(PointerOutOfBounds { ptr: end_ptr.erase_tag(), msg, allocation_size })
                }
                // Test align. Check this last; if both bounds and alignment are violated
                // we want the error to be about the bounds.
                if let Some(align) = align {
                    if M::force_int_for_alignment_check(&self.extra) {
                        let bits = self
                            .force_bits(ptr.into(), self.pointer_size())
                            .expect("ptr-to-int cast for align check should never fail");
                        check_offset_align(bits.try_into().unwrap(), align)?;
                    } else {
                        // Check allocation alignment and offset alignment.
                        if alloc_align.bytes() < align.bytes() {
                            throw_ub!(AlignmentCheckFailed { has: alloc_align, required: align });
                        }
                        check_offset_align(ptr.offset.bytes(), align)?;
                    }
                }

                // We can still be zero-sized in this branch, in which case we have to
                // return `None`.
                if size.bytes() == 0 { None } else { Some(ret_val) }
            }
        })
    }

    /// Test if the pointer might be null.
    pub fn ptr_may_be_null(&self, ptr: Pointer<M::PointerTag>) -> bool {
        let (size, _align) = self
            .get_size_and_align(ptr.alloc_id, AllocCheck::MaybeDead)
            .expect("alloc info with MaybeDead cannot fail");
        // If the pointer is out-of-bounds, it may be null.
        // Note that one-past-the-end (offset == size) is still inbounds, and never null.
        ptr.offset > size
    }
}

/// Allocation accessors
impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    /// Helper function to obtain a global (tcx) allocation.
    /// This attempts to return a reference to an existing allocation if
    /// one can be found in `tcx`. That, however, is only possible if `tcx` and
    /// this machine use the same pointer tag, so it is indirected through
    /// `M::tag_allocation`.
    fn get_global_alloc(
        memory_extra: &M::MemoryExtra,
        tcx: TyCtxt<'tcx>,
        id: AllocId,
        is_write: bool,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation<M::PointerTag, M::AllocExtra>>> {
        let (alloc, def_id) = match tcx.get_global_alloc(id) {
            Some(GlobalAlloc::Memory(mem)) => {
                // Memory of a constant or promoted or anonymous memory referenced by a static.
                (mem, None)
            }
            Some(GlobalAlloc::Function(..)) => throw_ub!(DerefFunctionPointer(id)),
            None => throw_ub!(PointerUseAfterFree(id)),
            Some(GlobalAlloc::Static(def_id)) => {
                assert!(tcx.is_static(def_id));
                assert!(!tcx.is_thread_local_static(def_id));
                // Notice that every static has two `AllocId` that will resolve to the same
                // thing here: one maps to `GlobalAlloc::Static`, this is the "lazy" ID,
                // and the other one is maps to `GlobalAlloc::Memory`, this is returned by
                // `eval_static_initializer` and it is the "resolved" ID.
                // The resolved ID is never used by the interpreted program, it is hidden.
                // This is relied upon for soundness of const-patterns; a pointer to the resolved
                // ID would "sidestep" the checks that make sure consts do not point to statics!
                // The `GlobalAlloc::Memory` branch here is still reachable though; when a static
                // contains a reference to memory that was created during its evaluation (i.e., not
                // to another static), those inner references only exist in "resolved" form.
                if tcx.is_foreign_item(def_id) {
                    throw_unsup!(ReadExternStatic(def_id));
                }

                (tcx.eval_static_initializer(def_id)?, Some(def_id))
            }
        };
        M::before_access_global(memory_extra, id, alloc, def_id, is_write)?;
        let alloc = Cow::Borrowed(alloc);
        // We got tcx memory. Let the machine initialize its "extra" stuff.
        let (alloc, tag) = M::init_allocation_extra(
            memory_extra,
            id, // always use the ID we got as input, not the "hidden" one.
            alloc,
            M::GLOBAL_KIND.map(MemoryKind::Machine),
        );
        // Sanity check that this is the same pointer we would have gotten via `global_base_pointer`.
        debug_assert_eq!(tag, M::tag_global_base_pointer(memory_extra, id));
        Ok(alloc)
    }

    /// Gives raw access to the `Allocation`, without bounds or alignment checks.
    /// The caller is responsible for calling the access hooks!
    fn get_raw(
        &self,
        id: AllocId,
    ) -> InterpResult<'tcx, &Allocation<M::PointerTag, M::AllocExtra>> {
        // The error type of the inner closure here is somewhat funny.  We have two
        // ways of "erroring": An actual error, or because we got a reference from
        // `get_global_alloc` that we can actually use directly without inserting anything anywhere.
        // So the error type is `InterpResult<'tcx, &Allocation<M::PointerTag>>`.
        let a = self.alloc_map.get_or(id, || {
            let alloc = Self::get_global_alloc(&self.extra, self.tcx, id, /*is_write*/ false)
                .map_err(Err)?;
            match alloc {
                Cow::Borrowed(alloc) => {
                    // We got a ref, cheaply return that as an "error" so that the
                    // map does not get mutated.
                    Err(Ok(alloc))
                }
                Cow::Owned(alloc) => {
                    // Need to put it into the map and return a ref to that
                    let kind = M::GLOBAL_KIND.expect(
                        "I got a global allocation that I have to copy but the machine does \
                            not expect that to happen",
                    );
                    Ok((MemoryKind::Machine(kind), alloc))
                }
            }
        });
        // Now unpack that funny error type
        match a {
            Ok(a) => Ok(&a.1),
            Err(a) => a,
        }
    }

    /// "Safe" (bounds and align-checked) allocation access.
    pub fn get<'a>(
        &'a self,
        sptr: Scalar<M::PointerTag>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Option<AllocRef<'a, 'tcx, M::PointerTag, M::AllocExtra>>> {
        let align = M::enforce_alignment(&self.extra).then_some(align);
        let ptr_and_alloc = self.check_and_deref_ptr(
            sptr,
            size,
            align,
            CheckInAllocMsg::MemoryAccessTest,
            |ptr| {
                let alloc = self.get_raw(ptr.alloc_id)?;
                Ok((alloc.size(), alloc.align, (ptr, alloc)))
            },
        )?;
        if let Some((ptr, alloc)) = ptr_and_alloc {
            M::memory_read(&self.extra, &alloc.extra, ptr, size)?;
            let range = alloc_range(ptr.offset, size);
            Ok(Some(AllocRef { alloc, range, tcx: self.tcx, alloc_id: ptr.alloc_id }))
        } else {
            // Even in this branch we have to be sure that we actually access the allocation, in
            // order to ensure that `static FOO: Type = FOO;` causes a cycle error instead of
            // magically pulling *any* ZST value from the ether. However, the `get_raw` above is
            // always called when `sptr` is truly a `Pointer`, so we are good.
            Ok(None)
        }
    }

    /// Return the `extra` field of the given allocation.
    pub fn get_alloc_extra<'a>(&'a self, id: AllocId) -> InterpResult<'tcx, &'a M::AllocExtra> {
        Ok(&self.get_raw(id)?.extra)
    }

    /// Gives raw mutable access to the `Allocation`, without bounds or alignment checks.
    /// The caller is responsible for calling the access hooks!
    ///
    /// Also returns a ptr to `self.extra` so that the caller can use it in parallel with the
    /// allocation.
    fn get_raw_mut(
        &mut self,
        id: AllocId,
    ) -> InterpResult<'tcx, (&mut Allocation<M::PointerTag, M::AllocExtra>, &mut M::MemoryExtra)>
    {
        let tcx = self.tcx;
        let memory_extra = &mut self.extra;
        let a = self.alloc_map.get_mut_or(id, || {
            // Need to make a copy, even if `get_global_alloc` is able
            // to give us a cheap reference.
            let alloc = Self::get_global_alloc(memory_extra, tcx, id, /*is_write*/ true)?;
            let kind = M::GLOBAL_KIND.expect(
                "I got a global allocation that I have to copy but the machine does \
                    not expect that to happen",
            );
            Ok((MemoryKind::Machine(kind), alloc.into_owned()))
        });
        // Unpack the error type manually because type inference doesn't
        // work otherwise (and we cannot help it because `impl Trait`)
        match a {
            Err(e) => Err(e),
            Ok(a) => {
                let a = &mut a.1;
                if a.mutability == Mutability::Not {
                    throw_ub!(WriteToReadOnly(id))
                }
                Ok((a, memory_extra))
            }
        }
    }

    /// "Safe" (bounds and align-checked) allocation access.
    pub fn get_mut<'a>(
        &'a mut self,
        sptr: Scalar<M::PointerTag>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Option<AllocRefMut<'a, 'tcx, M::PointerTag, M::AllocExtra>>> {
        let ptr = self.check_ptr_access(sptr, size, align)?;
        if let Some(ptr) = ptr {
            let tcx = self.tcx;
            // FIXME: can we somehow avoid looking up the allocation twice here?
            // We cannot call `get_raw_mut` inside `check_and_deref_ptr` as that would duplicate `&mut self`.
            let (alloc, extra) = self.get_raw_mut(ptr.alloc_id)?;
            M::memory_written(extra, &mut alloc.extra, ptr, size)?;
            let range = alloc_range(ptr.offset, size);
            Ok(Some(AllocRefMut { alloc, range, tcx, alloc_id: ptr.alloc_id }))
        } else {
            Ok(None)
        }
    }

    /// Return the `extra` field of the given allocation.
    pub fn get_alloc_extra_mut<'a>(
        &'a mut self,
        id: AllocId,
    ) -> InterpResult<'tcx, (&'a mut M::AllocExtra, &'a mut M::MemoryExtra)> {
        let (alloc, memory_extra) = self.get_raw_mut(id)?;
        Ok((&mut alloc.extra, memory_extra))
    }

    /// Obtain the size and alignment of an allocation, even if that allocation has
    /// been deallocated.
    ///
    /// If `liveness` is `AllocCheck::MaybeDead`, this function always returns `Ok`.
    pub fn get_size_and_align(
        &self,
        id: AllocId,
        liveness: AllocCheck,
    ) -> InterpResult<'static, (Size, Align)> {
        // # Regular allocations
        // Don't use `self.get_raw` here as that will
        // a) cause cycles in case `id` refers to a static
        // b) duplicate a global's allocation in miri
        if let Some((_, alloc)) = self.alloc_map.get(id) {
            return Ok((alloc.size(), alloc.align));
        }

        // # Function pointers
        // (both global from `alloc_map` and local from `extra_fn_ptr_map`)
        if self.get_fn_alloc(id).is_some() {
            return if let AllocCheck::Dereferenceable = liveness {
                // The caller requested no function pointers.
                throw_ub!(DerefFunctionPointer(id))
            } else {
                Ok((Size::ZERO, Align::ONE))
            };
        }

        // # Statics
        // Can't do this in the match argument, we may get cycle errors since the lock would
        // be held throughout the match.
        match self.tcx.get_global_alloc(id) {
            Some(GlobalAlloc::Static(did)) => {
                assert!(!self.tcx.is_thread_local_static(did));
                // Use size and align of the type.
                let ty = self.tcx.type_of(did);
                let layout = self.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                Ok((layout.size, layout.align.abi))
            }
            Some(GlobalAlloc::Memory(alloc)) => {
                // Need to duplicate the logic here, because the global allocations have
                // different associated types than the interpreter-local ones.
                Ok((alloc.size(), alloc.align))
            }
            Some(GlobalAlloc::Function(_)) => bug!("We already checked function pointers above"),
            // The rest must be dead.
            None => {
                if let AllocCheck::MaybeDead = liveness {
                    // Deallocated pointers are allowed, we should be able to find
                    // them in the map.
                    Ok(*self
                        .dead_alloc_map
                        .get(&id)
                        .expect("deallocated pointers should all be recorded in `dead_alloc_map`"))
                } else {
                    throw_ub!(PointerUseAfterFree(id))
                }
            }
        }
    }

    fn get_fn_alloc(&self, id: AllocId) -> Option<FnVal<'tcx, M::ExtraFnVal>> {
        trace!("reading fn ptr: {}", id);
        if let Some(extra) = self.extra_fn_ptr_map.get(&id) {
            Some(FnVal::Other(*extra))
        } else {
            match self.tcx.get_global_alloc(id) {
                Some(GlobalAlloc::Function(instance)) => Some(FnVal::Instance(instance)),
                _ => None,
            }
        }
    }

    pub fn get_fn(
        &self,
        ptr: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, FnVal<'tcx, M::ExtraFnVal>> {
        let ptr = self.force_ptr(ptr)?; // We definitely need a pointer value.
        if ptr.offset.bytes() != 0 {
            throw_ub!(InvalidFunctionPointer(ptr.erase_tag()))
        }
        self.get_fn_alloc(ptr.alloc_id)
            .ok_or_else(|| err_ub!(InvalidFunctionPointer(ptr.erase_tag())).into())
    }

    pub fn mark_immutable(&mut self, id: AllocId) -> InterpResult<'tcx> {
        self.get_raw_mut(id)?.0.mutability = Mutability::Not;
        Ok(())
    }

    /// Create a lazy debug printer that prints the given allocation and all allocations it points
    /// to, recursively.
    #[must_use]
    pub fn dump_alloc<'a>(&'a self, id: AllocId) -> DumpAllocs<'a, 'mir, 'tcx, M> {
        self.dump_allocs(vec![id])
    }

    /// Create a lazy debug printer for a list of allocations and all allocations they point to,
    /// recursively.
    #[must_use]
    pub fn dump_allocs<'a>(&'a self, mut allocs: Vec<AllocId>) -> DumpAllocs<'a, 'mir, 'tcx, M> {
        allocs.sort();
        allocs.dedup();
        DumpAllocs { mem: self, allocs }
    }

    /// Print leaked memory. Allocations reachable from `static_roots` or a `Global` allocation
    /// are not considered leaked. Leaks whose kind `may_leak()` returns true are not reported.
    pub fn leak_report(&self, static_roots: &[AllocId]) -> usize {
        // Collect the set of allocations that are *reachable* from `Global` allocations.
        let reachable = {
            let mut reachable = FxHashSet::default();
            let global_kind = M::GLOBAL_KIND.map(MemoryKind::Machine);
            let mut todo: Vec<_> = self.alloc_map.filter_map_collect(move |&id, &(kind, _)| {
                if Some(kind) == global_kind { Some(id) } else { None }
            });
            todo.extend(static_roots);
            while let Some(id) = todo.pop() {
                if reachable.insert(id) {
                    // This is a new allocation, add its relocations to `todo`.
                    if let Some((_, alloc)) = self.alloc_map.get(id) {
                        todo.extend(alloc.relocations().values().map(|&(_, target_id)| target_id));
                    }
                }
            }
            reachable
        };

        // All allocations that are *not* `reachable` and *not* `may_leak` are considered leaking.
        let leaks: Vec<_> = self.alloc_map.filter_map_collect(|&id, &(kind, _)| {
            if kind.may_leak() || reachable.contains(&id) { None } else { Some(id) }
        });
        let n = leaks.len();
        if n > 0 {
            eprintln!("The following memory was leaked: {:?}", self.dump_allocs(leaks));
        }
        n
    }

    /// This is used by [priroda](https://github.com/oli-obk/priroda)
    pub fn alloc_map(&self) -> &M::MemoryMap {
        &self.alloc_map
    }
}

#[doc(hidden)]
/// There's no way to use this directly, it's just a helper struct for the `dump_alloc(s)` methods.
pub struct DumpAllocs<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    mem: &'a Memory<'mir, 'tcx, M>,
    allocs: Vec<AllocId>,
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> std::fmt::Debug for DumpAllocs<'a, 'mir, 'tcx, M> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Cannot be a closure because it is generic in `Tag`, `Extra`.
        fn write_allocation_track_relocs<'tcx, Tag: Copy + fmt::Debug, Extra>(
            fmt: &mut std::fmt::Formatter<'_>,
            tcx: TyCtxt<'tcx>,
            allocs_to_print: &mut VecDeque<AllocId>,
            alloc: &Allocation<Tag, Extra>,
        ) -> std::fmt::Result {
            for &(_, target_id) in alloc.relocations().values() {
                allocs_to_print.push_back(target_id);
            }
            write!(fmt, "{}", pretty::display_allocation(tcx, alloc))
        }

        let mut allocs_to_print: VecDeque<_> = self.allocs.iter().copied().collect();
        // `allocs_printed` contains all allocations that we have already printed.
        let mut allocs_printed = FxHashSet::default();

        while let Some(id) = allocs_to_print.pop_front() {
            if !allocs_printed.insert(id) {
                // Already printed, so skip this.
                continue;
            }

            write!(fmt, "{}", id)?;
            match self.mem.alloc_map.get(id) {
                Some(&(kind, ref alloc)) => {
                    // normal alloc
                    write!(fmt, " ({}, ", kind)?;
                    write_allocation_track_relocs(
                        &mut *fmt,
                        self.mem.tcx,
                        &mut allocs_to_print,
                        alloc,
                    )?;
                }
                None => {
                    // global alloc
                    match self.mem.tcx.get_global_alloc(id) {
                        Some(GlobalAlloc::Memory(alloc)) => {
                            write!(fmt, " (unchanged global, ")?;
                            write_allocation_track_relocs(
                                &mut *fmt,
                                self.mem.tcx,
                                &mut allocs_to_print,
                                alloc,
                            )?;
                        }
                        Some(GlobalAlloc::Function(func)) => {
                            write!(fmt, " (fn: {})", func)?;
                        }
                        Some(GlobalAlloc::Static(did)) => {
                            write!(fmt, " (static: {})", self.mem.tcx.def_path_str(did))?;
                        }
                        None => {
                            write!(fmt, " (deallocated)")?;
                        }
                    }
                }
            }
            writeln!(fmt)?;
        }
        Ok(())
    }
}

/// Reading and writing.
impl<'tcx, 'a, Tag: Copy, Extra> AllocRefMut<'a, 'tcx, Tag, Extra> {
    pub fn write_scalar(
        &mut self,
        range: AllocRange,
        val: ScalarMaybeUninit<Tag>,
    ) -> InterpResult<'tcx> {
        Ok(self
            .alloc
            .write_scalar(&self.tcx, self.range.subrange(range), val)
            .map_err(|e| e.to_interp_error(self.alloc_id))?)
    }

    pub fn write_ptr_sized(
        &mut self,
        offset: Size,
        val: ScalarMaybeUninit<Tag>,
    ) -> InterpResult<'tcx> {
        self.write_scalar(alloc_range(offset, self.tcx.data_layout().pointer_size), val)
    }
}

impl<'tcx, 'a, Tag: Copy, Extra> AllocRef<'a, 'tcx, Tag, Extra> {
    pub fn read_scalar(&self, range: AllocRange) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        Ok(self
            .alloc
            .read_scalar(&self.tcx, self.range.subrange(range))
            .map_err(|e| e.to_interp_error(self.alloc_id))?)
    }

    pub fn read_ptr_sized(&self, offset: Size) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        self.read_scalar(alloc_range(offset, self.tcx.data_layout().pointer_size))
    }

    pub fn check_bytes(&self, range: AllocRange, allow_uninit_and_ptr: bool) -> InterpResult<'tcx> {
        Ok(self
            .alloc
            .check_bytes(&self.tcx, self.range.subrange(range), allow_uninit_and_ptr)
            .map_err(|e| e.to_interp_error(self.alloc_id))?)
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    /// Reads the given number of bytes from memory. Returns them as a slice.
    ///
    /// Performs appropriate bounds checks.
    pub fn read_bytes(&self, sptr: Scalar<M::PointerTag>, size: Size) -> InterpResult<'tcx, &[u8]> {
        let alloc_ref = match self.get(sptr, size, Align::ONE)? {
            Some(a) => a,
            None => return Ok(&[]), // zero-sized access
        };
        // Side-step AllocRef and directly access the underlying bytes more efficiently.
        // (We are staying inside the bounds here so all is good.)
        Ok(alloc_ref
            .alloc
            .get_bytes(&alloc_ref.tcx, alloc_ref.range)
            .map_err(|e| e.to_interp_error(alloc_ref.alloc_id))?)
    }

    /// Writes the given stream of bytes into memory.
    ///
    /// Performs appropriate bounds checks.
    pub fn write_bytes(
        &mut self,
        sptr: Scalar<M::PointerTag>,
        src: impl IntoIterator<Item = u8>,
    ) -> InterpResult<'tcx> {
        let mut src = src.into_iter();
        let (lower, upper) = src.size_hint();
        let len = upper.expect("can only write bounded iterators");
        assert_eq!(lower, len, "can only write iterators with a precise length");

        let size = Size::from_bytes(len);
        let alloc_ref = match self.get_mut(sptr, size, Align::ONE)? {
            Some(alloc_ref) => alloc_ref,
            None => {
                // zero-sized access
                assert_matches!(
                    src.next(),
                    None,
                    "iterator said it was empty but returned an element"
                );
                return Ok(());
            }
        };

        // Side-step AllocRef and directly access the underlying bytes more efficiently.
        // (We are staying inside the bounds here so all is good.)
        let bytes = alloc_ref.alloc.get_bytes_mut(&alloc_ref.tcx, alloc_ref.range);
        // `zip` would stop when the first iterator ends; we want to definitely
        // cover all of `bytes`.
        for dest in bytes {
            *dest = src.next().expect("iterator was shorter than it said it would be");
        }
        assert_matches!(src.next(), None, "iterator was longer than it said it would be");
        Ok(())
    }

    pub fn copy(
        &mut self,
        src: Scalar<M::PointerTag>,
        src_align: Align,
        dest: Scalar<M::PointerTag>,
        dest_align: Align,
        size: Size,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        self.copy_repeatedly(src, src_align, dest, dest_align, size, 1, nonoverlapping)
    }

    pub fn copy_repeatedly(
        &mut self,
        src: Scalar<M::PointerTag>,
        src_align: Align,
        dest: Scalar<M::PointerTag>,
        dest_align: Align,
        size: Size,
        num_copies: u64,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        let tcx = self.tcx;
        // We need to do our own bounds-checks.
        let src = self.check_ptr_access(src, size, src_align)?;
        let dest = self.check_ptr_access(dest, size * num_copies, dest_align)?; // `Size` multiplication

        // FIXME: we look up both allocations twice here, once ebfore for the `check_ptr_access`
        // and once below to get the underlying `&[mut] Allocation`.

        // Source alloc preparations and access hooks.
        let src = match src {
            None => return Ok(()), // Zero-sized *source*, that means dst is also zero-sized and we have nothing to do.
            Some(src_ptr) => src_ptr,
        };
        let src_alloc = self.get_raw(src.alloc_id)?;
        M::memory_read(&self.extra, &src_alloc.extra, src, size)?;
        // We need the `dest` ptr for the next operation, so we get it now.
        // We already did the source checks and called the hooks so we are good to return early.
        let dest = match dest {
            None => return Ok(()), // Zero-sized *destiantion*.
            Some(dest_ptr) => dest_ptr,
        };

        // first copy the relocations to a temporary buffer, because
        // `get_bytes_mut` will clear the relocations, which is correct,
        // since we don't want to keep any relocations at the target.
        // (`get_bytes_with_uninit_and_ptr` below checks that there are no
        // relocations overlapping the edges; those would not be handled correctly).
        let relocations = src_alloc.prepare_relocation_copy(
            self,
            alloc_range(src.offset, size),
            dest.offset,
            num_copies,
        );
        // Prepare a copy of the initialization mask.
        let compressed = src_alloc.compress_uninit_range(src, size);
        // This checks relocation edges on the src.
        let src_bytes = src_alloc
            .get_bytes_with_uninit_and_ptr(&tcx, alloc_range(src.offset, size))
            .map_err(|e| e.to_interp_error(src.alloc_id))?
            .as_ptr(); // raw ptr, so we can also get a ptr to the destination allocation

        // Destination alloc preparations and access hooks.
        let (dest_alloc, extra) = self.get_raw_mut(dest.alloc_id)?;
        M::memory_written(extra, &mut dest_alloc.extra, dest, size * num_copies)?;
        let dest_bytes = dest_alloc
            .get_bytes_mut_ptr(&tcx, alloc_range(dest.offset, size * num_copies))
            .as_mut_ptr();

        if compressed.no_bytes_init() {
            // Fast path: If all bytes are `uninit` then there is nothing to copy. The target range
            // is marked as uninitialized but we otherwise omit changing the byte representation which may
            // be arbitrary for uninitialized bytes.
            // This also avoids writing to the target bytes so that the backing allocation is never
            // touched if the bytes stay uninitialized for the whole interpreter execution. On contemporary
            // operating system this can avoid physically allocating the page.
            dest_alloc.mark_init(alloc_range(dest.offset, size * num_copies), false); // `Size` multiplication
            dest_alloc.mark_relocation_range(relocations);
            return Ok(());
        }

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
        // The pointers above remain valid even if the `HashMap` table is moved around because they
        // point into the `Vec` storing the bytes.
        unsafe {
            if src.alloc_id == dest.alloc_id {
                if nonoverlapping {
                    // `Size` additions
                    if (src.offset <= dest.offset && src.offset + size > dest.offset)
                        || (dest.offset <= src.offset && dest.offset + size > src.offset)
                    {
                        throw_ub_format!("copy_nonoverlapping called on overlapping ranges")
                    }
                }

                for i in 0..num_copies {
                    ptr::copy(
                        src_bytes,
                        dest_bytes.add((size * i).bytes_usize()), // `Size` multiplication
                        size.bytes_usize(),
                    );
                }
            } else {
                for i in 0..num_copies {
                    ptr::copy_nonoverlapping(
                        src_bytes,
                        dest_bytes.add((size * i).bytes_usize()), // `Size` multiplication
                        size.bytes_usize(),
                    );
                }
            }
        }

        // now fill in all the "init" data
        dest_alloc.mark_compressed_init_range(&compressed, dest, size, num_copies);
        // copy the relocations to the destination
        dest_alloc.mark_relocation_range(relocations);

        Ok(())
    }
}

/// Machine pointer introspection.
impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    pub fn force_ptr(
        &self,
        scalar: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        match scalar {
            Scalar::Ptr(ptr) => Ok(ptr),
            _ => M::int_to_ptr(&self, scalar.to_machine_usize(self)?),
        }
    }

    pub fn force_bits(
        &self,
        scalar: Scalar<M::PointerTag>,
        size: Size,
    ) -> InterpResult<'tcx, u128> {
        match scalar.to_bits_or_ptr(size, self) {
            Ok(bits) => Ok(bits),
            Err(ptr) => Ok(M::ptr_to_int(&self, ptr)?.into()),
        }
    }
}
