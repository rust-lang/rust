//! The memory subsystem.
//!
//! Generally, we use `Pointer` to denote memory addresses. However, some operations
//! have a "size"-like parameter, and they take `Scalar` for the address because
//! if the size is 0, then the pointer can also be a (properly aligned, non-null)
//! integer. It is crucial that these operations call `check_align` *before*
//! short-circuiting the empty case!

use std::assert_matches::assert_matches;
use std::borrow::{Borrow, Cow};
use std::cell::Cell;
use std::collections::VecDeque;
use std::{fmt, ptr};

use rustc_abi::{Align, HasDataLayout, Size};
use rustc_ast::Mutability;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_middle::bug;
use rustc_middle::mir::display_allocation;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use tracing::{debug, instrument, trace};

use super::{
    AllocBytes, AllocId, AllocInit, AllocMap, AllocRange, Allocation, CheckAlignMsg,
    CheckInAllocMsg, CtfeProvenance, GlobalAlloc, InterpCx, InterpResult, Machine, MayLeak,
    Misalignment, Pointer, PointerArithmetic, Provenance, Scalar, alloc_range, err_ub,
    err_ub_custom, interp_ok, throw_ub, throw_ub_custom, throw_unsup, throw_unsup_format,
};
use crate::fluent_generated as fluent;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum MemoryKind<T> {
    /// Stack memory. Error if deallocated except during a stack pop.
    Stack,
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
            MemoryKind::CallerLocation => true,
            MemoryKind::Machine(k) => k.may_leak(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for MemoryKind<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryKind::Stack => write!(f, "stack variable"),
            MemoryKind::CallerLocation => write!(f, "caller location"),
            MemoryKind::Machine(m) => write!(f, "{m}"),
        }
    }
}

/// The return value of `get_alloc_info` indicates the "kind" of the allocation.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AllocKind {
    /// A regular live data allocation.
    LiveData,
    /// A function allocation (that fn ptrs point to).
    Function,
    /// A (symbolic) vtable allocation.
    VTable,
    /// A dead allocation.
    Dead,
}

/// Metadata about an `AllocId`.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct AllocInfo {
    pub size: Size,
    pub align: Align,
    pub kind: AllocKind,
    pub mutbl: Mutability,
}

impl AllocInfo {
    fn new(size: Size, align: Align, kind: AllocKind, mutbl: Mutability) -> Self {
        Self { size, align, kind, mutbl }
    }
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
            FnVal::Instance(instance) => interp_ok(instance),
            FnVal::Other(_) => {
                throw_unsup_format!("'foreign' function pointers are not supported in this context")
            }
        }
    }
}

// `Memory` has to depend on the `Machine` because some of its operations
// (e.g., `get`) call a `Machine` hook.
pub struct Memory<'tcx, M: Machine<'tcx>> {
    /// Allocations local to this instance of the interpreter. The kind
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
    extra_fn_ptr_map: FxIndexMap<AllocId, M::ExtraFnVal>,

    /// To be able to compare pointers with null, and to check alignment for accesses
    /// to ZSTs (where pointers may dangle), we keep track of the size even for allocations
    /// that do not exist any more.
    // FIXME: this should not be public, but interning currently needs access to it
    pub(super) dead_alloc_map: FxIndexMap<AllocId, (Size, Align)>,

    /// This stores whether we are currently doing reads purely for the purpose of validation.
    /// Those reads do not trigger the machine's hooks for memory reads.
    /// Needless to say, this must only be set with great care!
    validation_in_progress: Cell<bool>,
}

/// A reference to some allocation that was already bounds-checked for the given region
/// and had the on-access machine hooks run.
#[derive(Copy, Clone)]
pub struct AllocRef<'a, 'tcx, Prov: Provenance, Extra, Bytes: AllocBytes = Box<[u8]>> {
    alloc: &'a Allocation<Prov, Extra, Bytes>,
    range: AllocRange,
    tcx: TyCtxt<'tcx>,
    alloc_id: AllocId,
}
/// A reference to some allocation that was already bounds-checked for the given region
/// and had the on-access machine hooks run.
pub struct AllocRefMut<'a, 'tcx, Prov: Provenance, Extra, Bytes: AllocBytes = Box<[u8]>> {
    alloc: &'a mut Allocation<Prov, Extra, Bytes>,
    range: AllocRange,
    tcx: TyCtxt<'tcx>,
    alloc_id: AllocId,
}

impl<'tcx, M: Machine<'tcx>> Memory<'tcx, M> {
    pub fn new() -> Self {
        Memory {
            alloc_map: M::MemoryMap::default(),
            extra_fn_ptr_map: FxIndexMap::default(),
            dead_alloc_map: FxIndexMap::default(),
            validation_in_progress: Cell::new(false),
        }
    }

    /// This is used by [priroda](https://github.com/oli-obk/priroda)
    pub fn alloc_map(&self) -> &M::MemoryMap {
        &self.alloc_map
    }
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Call this to turn untagged "global" pointers (obtained via `tcx`) into
    /// the machine pointer to the allocation. Must never be used
    /// for any other pointers, nor for TLS statics.
    ///
    /// Using the resulting pointer represents a *direct* access to that memory
    /// (e.g. by directly using a `static`),
    /// as opposed to access through a pointer that was created by the program.
    ///
    /// This function can fail only if `ptr` points to an `extern static`.
    #[inline]
    pub fn global_root_pointer(
        &self,
        ptr: Pointer<CtfeProvenance>,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let alloc_id = ptr.provenance.alloc_id();
        // We need to handle `extern static`.
        match self.tcx.try_get_global_alloc(alloc_id) {
            Some(GlobalAlloc::Static(def_id)) if self.tcx.is_thread_local_static(def_id) => {
                // Thread-local statics do not have a constant address. They *must* be accessed via
                // `ThreadLocalRef`; we can never have a pointer to them as a regular constant value.
                bug!("global memory cannot point to thread-local static")
            }
            Some(GlobalAlloc::Static(def_id)) if self.tcx.is_foreign_item(def_id) => {
                return M::extern_static_pointer(self, def_id);
            }
            None => {
                assert!(
                    self.memory.extra_fn_ptr_map.contains_key(&alloc_id),
                    "{alloc_id:?} is neither global nor a function pointer"
                );
            }
            _ => {}
        }
        // And we need to get the provenance.
        M::adjust_alloc_root_pointer(self, ptr, M::GLOBAL_KIND.map(MemoryKind::Machine))
    }

    pub fn fn_ptr(&mut self, fn_val: FnVal<'tcx, M::ExtraFnVal>) -> Pointer<M::Provenance> {
        let id = match fn_val {
            FnVal::Instance(instance) => {
                let salt = M::get_global_alloc_salt(self, Some(instance));
                self.tcx.reserve_and_set_fn_alloc(instance, salt)
            }
            FnVal::Other(extra) => {
                // FIXME(RalfJung): Should we have a cache here?
                let id = self.tcx.reserve_alloc_id();
                let old = self.memory.extra_fn_ptr_map.insert(id, extra);
                assert!(old.is_none());
                id
            }
        };
        // Functions are global allocations, so make sure we get the right root pointer.
        // We know this is not an `extern static` so this cannot fail.
        self.global_root_pointer(Pointer::from(id)).unwrap()
    }

    pub fn allocate_ptr(
        &mut self,
        size: Size,
        align: Align,
        kind: MemoryKind<M::MemoryKind>,
        init: AllocInit,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let alloc = if M::PANIC_ON_ALLOC_FAIL {
            Allocation::new(size, align, init)
        } else {
            Allocation::try_new(size, align, init)?
        };
        self.insert_allocation(alloc, kind)
    }

    pub fn allocate_bytes_ptr(
        &mut self,
        bytes: &[u8],
        align: Align,
        kind: MemoryKind<M::MemoryKind>,
        mutability: Mutability,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let alloc = Allocation::from_bytes(bytes, align, mutability);
        self.insert_allocation(alloc, kind)
    }

    pub fn insert_allocation(
        &mut self,
        alloc: Allocation<M::Provenance, (), M::Bytes>,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        assert!(alloc.size() <= self.max_size_of_val());
        let id = self.tcx.reserve_alloc_id();
        debug_assert_ne!(
            Some(kind),
            M::GLOBAL_KIND.map(MemoryKind::Machine),
            "dynamically allocating global memory"
        );
        // This cannot be merged with the `adjust_global_allocation` code path
        // since here we have an allocation that already uses `M::Bytes`.
        let extra = M::init_local_allocation(self, id, kind, alloc.size(), alloc.align)?;
        let alloc = alloc.with_extra(extra);
        self.memory.alloc_map.insert(id, (kind, alloc));
        M::adjust_alloc_root_pointer(self, Pointer::from(id), Some(kind))
    }

    /// If this grows the allocation, `init_growth` determines
    /// whether the additional space will be initialized.
    pub fn reallocate_ptr(
        &mut self,
        ptr: Pointer<Option<M::Provenance>>,
        old_size_and_align: Option<(Size, Align)>,
        new_size: Size,
        new_align: Align,
        kind: MemoryKind<M::MemoryKind>,
        init_growth: AllocInit,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let (alloc_id, offset, _prov) = self.ptr_get_alloc_id(ptr, 0)?;
        if offset.bytes() != 0 {
            throw_ub_custom!(
                fluent::const_eval_realloc_or_alloc_with_offset,
                ptr = format!("{ptr:?}"),
                kind = "realloc"
            );
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc".
        // This happens so rarely, the perf advantage is outweighed by the maintenance cost.
        // If requested, we zero-init the entire allocation, to ensure that a growing
        // allocation has its new bytes properly set. For the part that is copied,
        // `mem_copy` below will de-initialize things as necessary.
        let new_ptr = self.allocate_ptr(new_size, new_align, kind, init_growth)?;
        let old_size = match old_size_and_align {
            Some((size, _align)) => size,
            None => self.get_alloc_raw(alloc_id)?.size(),
        };
        // This will also call the access hooks.
        self.mem_copy(ptr, new_ptr.into(), old_size.min(new_size), /*nonoverlapping*/ true)?;
        self.deallocate_ptr(ptr, old_size_and_align, kind)?;

        interp_ok(new_ptr)
    }

    #[instrument(skip(self), level = "debug")]
    pub fn deallocate_ptr(
        &mut self,
        ptr: Pointer<Option<M::Provenance>>,
        old_size_and_align: Option<(Size, Align)>,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx> {
        let (alloc_id, offset, prov) = self.ptr_get_alloc_id(ptr, 0)?;
        trace!("deallocating: {alloc_id:?}");

        if offset.bytes() != 0 {
            throw_ub_custom!(
                fluent::const_eval_realloc_or_alloc_with_offset,
                ptr = format!("{ptr:?}"),
                kind = "dealloc",
            );
        }

        let Some((alloc_kind, mut alloc)) = self.memory.alloc_map.remove(&alloc_id) else {
            // Deallocating global memory -- always an error
            return Err(match self.tcx.try_get_global_alloc(alloc_id) {
                Some(GlobalAlloc::Function { .. }) => {
                    err_ub_custom!(
                        fluent::const_eval_invalid_dealloc,
                        alloc_id = alloc_id,
                        kind = "fn",
                    )
                }
                Some(GlobalAlloc::VTable(..)) => {
                    err_ub_custom!(
                        fluent::const_eval_invalid_dealloc,
                        alloc_id = alloc_id,
                        kind = "vtable",
                    )
                }
                Some(GlobalAlloc::Static(..) | GlobalAlloc::Memory(..)) => {
                    err_ub_custom!(
                        fluent::const_eval_invalid_dealloc,
                        alloc_id = alloc_id,
                        kind = "static_mem"
                    )
                }
                None => err_ub!(PointerUseAfterFree(alloc_id, CheckInAllocMsg::MemoryAccessTest)),
            })
            .into();
        };

        if alloc.mutability.is_not() {
            throw_ub_custom!(fluent::const_eval_dealloc_immutable, alloc = alloc_id,);
        }
        if alloc_kind != kind {
            throw_ub_custom!(
                fluent::const_eval_dealloc_kind_mismatch,
                alloc = alloc_id,
                alloc_kind = format!("{alloc_kind}"),
                kind = format!("{kind}"),
            );
        }
        if let Some((size, align)) = old_size_and_align {
            if size != alloc.size() || align != alloc.align {
                throw_ub_custom!(
                    fluent::const_eval_dealloc_incorrect_layout,
                    alloc = alloc_id,
                    size = alloc.size().bytes(),
                    align = alloc.align.bytes(),
                    size_found = size.bytes(),
                    align_found = align.bytes(),
                )
            }
        }

        // Let the machine take some extra action
        let size = alloc.size();
        M::before_memory_deallocation(
            self.tcx,
            &mut self.machine,
            &mut alloc.extra,
            ptr,
            (alloc_id, prov),
            size,
            alloc.align,
            kind,
        )?;

        // Don't forget to remember size and align of this now-dead allocation
        let old = self.memory.dead_alloc_map.insert(alloc_id, (size, alloc.align));
        if old.is_some() {
            bug!("Nothing can be deallocated twice");
        }

        interp_ok(())
    }

    /// Internal helper function to determine the allocation and offset of a pointer (if any).
    #[inline(always)]
    fn get_ptr_access(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
    ) -> InterpResult<'tcx, Option<(AllocId, Size, M::ProvenanceExtra)>> {
        let size = i64::try_from(size.bytes()).unwrap(); // it would be an error to even ask for more than isize::MAX bytes
        Self::check_and_deref_ptr(
            self,
            ptr,
            size,
            CheckInAllocMsg::MemoryAccessTest,
            |this, alloc_id, offset, prov| {
                let (size, align) = this
                    .get_live_alloc_size_and_align(alloc_id, CheckInAllocMsg::MemoryAccessTest)?;
                interp_ok((size, align, (alloc_id, offset, prov)))
            },
        )
    }

    /// Check if the given pointer points to live memory of the given `size`.
    /// The caller can control the error message for the out-of-bounds case.
    #[inline(always)]
    pub fn check_ptr_access(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
        msg: CheckInAllocMsg,
    ) -> InterpResult<'tcx> {
        let size = i64::try_from(size.bytes()).unwrap(); // it would be an error to even ask for more than isize::MAX bytes
        Self::check_and_deref_ptr(self, ptr, size, msg, |this, alloc_id, _, _| {
            let (size, align) = this.get_live_alloc_size_and_align(alloc_id, msg)?;
            interp_ok((size, align, ()))
        })?;
        interp_ok(())
    }

    /// Check whether the given pointer points to live memory for a signed amount of bytes.
    /// A negative amounts means that the given range of memory to the left of the pointer
    /// needs to be dereferenceable.
    pub fn check_ptr_access_signed(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: i64,
        msg: CheckInAllocMsg,
    ) -> InterpResult<'tcx> {
        Self::check_and_deref_ptr(self, ptr, size, msg, |this, alloc_id, _, _| {
            let (size, align) = this.get_live_alloc_size_and_align(alloc_id, msg)?;
            interp_ok((size, align, ()))
        })?;
        interp_ok(())
    }

    /// Low-level helper function to check if a ptr is in-bounds and potentially return a reference
    /// to the allocation it points to. Supports both shared and mutable references, as the actual
    /// checking is offloaded to a helper closure. Supports signed sizes for checks "to the left" of
    /// a pointer.
    ///
    /// `alloc_size` will only get called for non-zero-sized accesses.
    ///
    /// Returns `None` if and only if the size is 0.
    fn check_and_deref_ptr<T, R: Borrow<Self>>(
        this: R,
        ptr: Pointer<Option<M::Provenance>>,
        size: i64,
        msg: CheckInAllocMsg,
        alloc_size: impl FnOnce(
            R,
            AllocId,
            Size,
            M::ProvenanceExtra,
        ) -> InterpResult<'tcx, (Size, Align, T)>,
    ) -> InterpResult<'tcx, Option<T>> {
        // Everything is okay with size 0.
        if size == 0 {
            return interp_ok(None);
        }

        interp_ok(match this.borrow().ptr_try_get_alloc_id(ptr, size) {
            Err(addr) => {
                // We couldn't get a proper allocation.
                throw_ub!(DanglingIntPointer { addr, inbounds_size: size, msg });
            }
            Ok((alloc_id, offset, prov)) => {
                let tcx = this.borrow().tcx;
                let (alloc_size, _alloc_align, ret_val) = alloc_size(this, alloc_id, offset, prov)?;
                let offset = offset.bytes();
                // Compute absolute begin and end of the range.
                let (begin, end) = if size >= 0 {
                    (Some(offset), offset.checked_add(size as u64))
                } else {
                    (offset.checked_sub(size.unsigned_abs()), Some(offset))
                };
                // Ensure both are within bounds.
                let in_bounds = begin.is_some() && end.is_some_and(|e| e <= alloc_size.bytes());
                if !in_bounds {
                    throw_ub!(PointerOutOfBounds {
                        alloc_id,
                        alloc_size,
                        ptr_offset: tcx.sign_extend_to_target_isize(offset),
                        inbounds_size: size,
                        msg,
                    })
                }

                Some(ret_val)
            }
        })
    }

    pub(super) fn check_misalign(
        &self,
        misaligned: Option<Misalignment>,
        msg: CheckAlignMsg,
    ) -> InterpResult<'tcx> {
        if let Some(misaligned) = misaligned {
            throw_ub!(AlignmentCheckFailed(misaligned, msg))
        }
        interp_ok(())
    }

    pub(super) fn is_ptr_misaligned(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        align: Align,
    ) -> Option<Misalignment> {
        if !M::enforce_alignment(self) || align.bytes() == 1 {
            return None;
        }

        #[inline]
        fn is_offset_misaligned(offset: u64, align: Align) -> Option<Misalignment> {
            if offset % align.bytes() == 0 {
                None
            } else {
                // The biggest power of two through which `offset` is divisible.
                let offset_pow2 = 1 << offset.trailing_zeros();
                Some(Misalignment { has: Align::from_bytes(offset_pow2).unwrap(), required: align })
            }
        }

        match self.ptr_try_get_alloc_id(ptr, 0) {
            Err(addr) => is_offset_misaligned(addr, align),
            Ok((alloc_id, offset, _prov)) => {
                let alloc_info = self.get_alloc_info(alloc_id);
                if let Some(misalign) = M::alignment_check(
                    self,
                    alloc_id,
                    alloc_info.align,
                    alloc_info.kind,
                    offset,
                    align,
                ) {
                    Some(misalign)
                } else if M::Provenance::OFFSET_IS_ADDR {
                    is_offset_misaligned(ptr.addr().bytes(), align)
                } else {
                    // Check allocation alignment and offset alignment.
                    if alloc_info.align.bytes() < align.bytes() {
                        Some(Misalignment { has: alloc_info.align, required: align })
                    } else {
                        is_offset_misaligned(offset.bytes(), align)
                    }
                }
            }
        }
    }

    /// Checks a pointer for misalignment.
    ///
    /// The error assumes this is checking the pointer used directly for an access.
    pub fn check_ptr_align(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        align: Align,
    ) -> InterpResult<'tcx> {
        self.check_misalign(self.is_ptr_misaligned(ptr, align), CheckAlignMsg::AccessedPtr)
    }
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// This function is used by Miri's provenance GC to remove unreachable entries from the dead_alloc_map.
    pub fn remove_unreachable_allocs(&mut self, reachable_allocs: &FxHashSet<AllocId>) {
        // Unlike all the other GC helpers where we check if an `AllocId` is found in the interpreter or
        // is live, here all the IDs in the map are for dead allocations so we don't
        // need to check for liveness.
        #[allow(rustc::potential_query_instability)] // Only used from Miri, not queries.
        self.memory.dead_alloc_map.retain(|id, _| reachable_allocs.contains(id));
    }
}

/// Allocation accessors
impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Helper function to obtain a global (tcx) allocation.
    /// This attempts to return a reference to an existing allocation if
    /// one can be found in `tcx`. That, however, is only possible if `tcx` and
    /// this machine use the same pointer provenance, so it is indirected through
    /// `M::adjust_allocation`.
    fn get_global_alloc(
        &self,
        id: AllocId,
        is_write: bool,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation<M::Provenance, M::AllocExtra, M::Bytes>>> {
        let (alloc, def_id) = match self.tcx.try_get_global_alloc(id) {
            Some(GlobalAlloc::Memory(mem)) => {
                // Memory of a constant or promoted or anonymous memory referenced by a static.
                (mem, None)
            }
            Some(GlobalAlloc::Function { .. }) => throw_ub!(DerefFunctionPointer(id)),
            Some(GlobalAlloc::VTable(..)) => throw_ub!(DerefVTablePointer(id)),
            None => throw_ub!(PointerUseAfterFree(id, CheckInAllocMsg::MemoryAccessTest)),
            Some(GlobalAlloc::Static(def_id)) => {
                assert!(self.tcx.is_static(def_id));
                // Thread-local statics do not have a constant address. They *must* be accessed via
                // `ThreadLocalRef`; we can never have a pointer to them as a regular constant value.
                assert!(!self.tcx.is_thread_local_static(def_id));
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
                if self.tcx.is_foreign_item(def_id) {
                    // This is unreachable in Miri, but can happen in CTFE where we actually *do* support
                    // referencing arbitrary (declared) extern statics.
                    throw_unsup!(ExternStatic(def_id));
                }

                // We don't give a span -- statics don't need that, they cannot be generic or associated.
                let val = self.ctfe_query(|tcx| tcx.eval_static_initializer(def_id))?;
                (val, Some(def_id))
            }
        };
        M::before_access_global(self.tcx, &self.machine, id, alloc, def_id, is_write)?;
        // We got tcx memory. Let the machine initialize its "extra" stuff.
        M::adjust_global_allocation(
            self,
            id, // always use the ID we got as input, not the "hidden" one.
            alloc.inner(),
        )
    }

    /// Gives raw access to the `Allocation`, without bounds or alignment checks.
    /// The caller is responsible for calling the access hooks!
    ///
    /// You almost certainly want to use `get_ptr_alloc`/`get_ptr_alloc_mut` instead.
    fn get_alloc_raw(
        &self,
        id: AllocId,
    ) -> InterpResult<'tcx, &Allocation<M::Provenance, M::AllocExtra, M::Bytes>> {
        // The error type of the inner closure here is somewhat funny. We have two
        // ways of "erroring": An actual error, or because we got a reference from
        // `get_global_alloc` that we can actually use directly without inserting anything anywhere.
        // So the error type is `InterpResult<'tcx, &Allocation<M::Provenance>>`.
        let a = self.memory.alloc_map.get_or(id, || {
            // We have to funnel the `InterpErrorInfo` through a `Result` to match the `get_or` API,
            // so we use `report_err` for that.
            let alloc = self.get_global_alloc(id, /*is_write*/ false).report_err().map_err(Err)?;
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
            Ok(a) => interp_ok(&a.1),
            Err(a) => a.into(),
        }
    }

    /// Gives raw, immutable access to the `Allocation` address, without bounds or alignment checks.
    /// The caller is responsible for calling the access hooks!
    pub fn get_alloc_bytes_unchecked_raw(&self, id: AllocId) -> InterpResult<'tcx, *const u8> {
        let alloc = self.get_alloc_raw(id)?;
        interp_ok(alloc.get_bytes_unchecked_raw())
    }

    /// Bounds-checked *but not align-checked* allocation access.
    pub fn get_ptr_alloc<'a>(
        &'a self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
    ) -> InterpResult<'tcx, Option<AllocRef<'a, 'tcx, M::Provenance, M::AllocExtra, M::Bytes>>>
    {
        let size_i64 = i64::try_from(size.bytes()).unwrap(); // it would be an error to even ask for more than isize::MAX bytes
        let ptr_and_alloc = Self::check_and_deref_ptr(
            self,
            ptr,
            size_i64,
            CheckInAllocMsg::MemoryAccessTest,
            |this, alloc_id, offset, prov| {
                let alloc = this.get_alloc_raw(alloc_id)?;
                interp_ok((alloc.size(), alloc.align, (alloc_id, offset, prov, alloc)))
            },
        )?;
        // We want to call the hook on *all* accesses that involve an AllocId, including zero-sized
        // accesses. That means we cannot rely on the closure above or the `Some` branch below. We
        // do this after `check_and_deref_ptr` to ensure some basic sanity has already been checked.
        if !self.memory.validation_in_progress.get() {
            if let Ok((alloc_id, ..)) = self.ptr_try_get_alloc_id(ptr, size_i64) {
                M::before_alloc_read(self, alloc_id)?;
            }
        }

        if let Some((alloc_id, offset, prov, alloc)) = ptr_and_alloc {
            let range = alloc_range(offset, size);
            if !self.memory.validation_in_progress.get() {
                M::before_memory_read(
                    self.tcx,
                    &self.machine,
                    &alloc.extra,
                    ptr,
                    (alloc_id, prov),
                    range,
                )?;
            }
            interp_ok(Some(AllocRef { alloc, range, tcx: *self.tcx, alloc_id }))
        } else {
            interp_ok(None)
        }
    }

    /// Return the `extra` field of the given allocation.
    pub fn get_alloc_extra<'a>(&'a self, id: AllocId) -> InterpResult<'tcx, &'a M::AllocExtra> {
        interp_ok(&self.get_alloc_raw(id)?.extra)
    }

    /// Return the `mutability` field of the given allocation.
    pub fn get_alloc_mutability<'a>(&'a self, id: AllocId) -> InterpResult<'tcx, Mutability> {
        interp_ok(self.get_alloc_raw(id)?.mutability)
    }

    /// Gives raw mutable access to the `Allocation`, without bounds or alignment checks.
    /// The caller is responsible for calling the access hooks!
    ///
    /// Also returns a ptr to `self.extra` so that the caller can use it in parallel with the
    /// allocation.
    fn get_alloc_raw_mut(
        &mut self,
        id: AllocId,
    ) -> InterpResult<'tcx, (&mut Allocation<M::Provenance, M::AllocExtra, M::Bytes>, &mut M)> {
        // We have "NLL problem case #3" here, which cannot be worked around without loss of
        // efficiency even for the common case where the key is in the map.
        // <https://rust-lang.github.io/rfcs/2094-nll.html#problem-case-3-conditional-control-flow-across-functions>
        // (Cannot use `get_mut_or` since `get_global_alloc` needs `&self`, and that boils down to
        // Miri's `adjust_alloc_root_pointer` needing to look up the size of the allocation.
        // It could be avoided with a totally separate codepath in Miri for handling the absolute address
        // of global allocations, but that's not worth it.)
        if self.memory.alloc_map.get_mut(id).is_none() {
            // Slow path.
            // Allocation not found locally, go look global.
            let alloc = self.get_global_alloc(id, /*is_write*/ true)?;
            let kind = M::GLOBAL_KIND.expect(
                "I got a global allocation that I have to copy but the machine does \
                    not expect that to happen",
            );
            self.memory.alloc_map.insert(id, (MemoryKind::Machine(kind), alloc.into_owned()));
        }

        let (_kind, alloc) = self.memory.alloc_map.get_mut(id).unwrap();
        if alloc.mutability.is_not() {
            throw_ub!(WriteToReadOnly(id))
        }
        interp_ok((alloc, &mut self.machine))
    }

    /// Gives raw, mutable access to the `Allocation` address, without bounds or alignment checks.
    /// The caller is responsible for calling the access hooks!
    pub fn get_alloc_bytes_unchecked_raw_mut(
        &mut self,
        id: AllocId,
    ) -> InterpResult<'tcx, *mut u8> {
        let alloc = self.get_alloc_raw_mut(id)?.0;
        interp_ok(alloc.get_bytes_unchecked_raw_mut())
    }

    /// Bounds-checked *but not align-checked* allocation access.
    pub fn get_ptr_alloc_mut<'a>(
        &'a mut self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
    ) -> InterpResult<'tcx, Option<AllocRefMut<'a, 'tcx, M::Provenance, M::AllocExtra, M::Bytes>>>
    {
        let tcx = self.tcx;
        let validation_in_progress = self.memory.validation_in_progress.get();

        let size_i64 = i64::try_from(size.bytes()).unwrap(); // it would be an error to even ask for more than isize::MAX bytes
        let ptr_and_alloc = Self::check_and_deref_ptr(
            self,
            ptr,
            size_i64,
            CheckInAllocMsg::MemoryAccessTest,
            |this, alloc_id, offset, prov| {
                let (alloc, machine) = this.get_alloc_raw_mut(alloc_id)?;
                interp_ok((alloc.size(), alloc.align, (alloc_id, offset, prov, alloc, machine)))
            },
        )?;

        if let Some((alloc_id, offset, prov, alloc, machine)) = ptr_and_alloc {
            let range = alloc_range(offset, size);
            if !validation_in_progress {
                M::before_memory_write(
                    tcx,
                    machine,
                    &mut alloc.extra,
                    ptr,
                    (alloc_id, prov),
                    range,
                )?;
            }
            interp_ok(Some(AllocRefMut { alloc, range, tcx: *tcx, alloc_id }))
        } else {
            interp_ok(None)
        }
    }

    /// Return the `extra` field of the given allocation.
    pub fn get_alloc_extra_mut<'a>(
        &'a mut self,
        id: AllocId,
    ) -> InterpResult<'tcx, (&'a mut M::AllocExtra, &'a mut M)> {
        let (alloc, machine) = self.get_alloc_raw_mut(id)?;
        interp_ok((&mut alloc.extra, machine))
    }

    /// Check whether an allocation is live. This is faster than calling
    /// [`InterpCx::get_alloc_info`] if all you need to check is whether the kind is
    /// [`AllocKind::Dead`] because it doesn't have to look up the type and layout of statics.
    pub fn is_alloc_live(&self, id: AllocId) -> bool {
        self.memory.alloc_map.contains_key_ref(&id)
            || self.memory.extra_fn_ptr_map.contains_key(&id)
            // We check `tcx` last as that has to acquire a lock in `many-seeds` mode.
            // This also matches the order in `get_alloc_info`.
            || self.tcx.try_get_global_alloc(id).is_some()
    }

    /// Obtain the size and alignment of an allocation, even if that allocation has
    /// been deallocated.
    pub fn get_alloc_info(&self, id: AllocId) -> AllocInfo {
        // # Regular allocations
        // Don't use `self.get_raw` here as that will
        // a) cause cycles in case `id` refers to a static
        // b) duplicate a global's allocation in miri
        if let Some((_, alloc)) = self.memory.alloc_map.get(id) {
            return AllocInfo::new(
                alloc.size(),
                alloc.align,
                AllocKind::LiveData,
                alloc.mutability,
            );
        }

        // # Function pointers
        // (both global from `alloc_map` and local from `extra_fn_ptr_map`)
        if let Some(fn_val) = self.get_fn_alloc(id) {
            let align = match fn_val {
                FnVal::Instance(instance) => {
                    // Function alignment can be set globally with the `-Zmin-function-alignment=<n>` flag;
                    // the alignment from a `#[repr(align(<n>))]` is used if it specifies a higher alignment.
                    let fn_align = self.tcx.codegen_fn_attrs(instance.def_id()).alignment;
                    let global_align = self.tcx.sess.opts.unstable_opts.min_function_alignment;

                    Ord::max(global_align, fn_align).unwrap_or(Align::ONE)
                }
                // Machine-specific extra functions currently do not support alignment restrictions.
                FnVal::Other(_) => Align::ONE,
            };

            return AllocInfo::new(Size::ZERO, align, AllocKind::Function, Mutability::Not);
        }

        // # Global allocations
        if let Some(global_alloc) = self.tcx.try_get_global_alloc(id) {
            let (size, align) = global_alloc.size_and_align(*self.tcx, self.typing_env);
            let mutbl = global_alloc.mutability(*self.tcx, self.typing_env);
            let kind = match global_alloc {
                GlobalAlloc::Static { .. } | GlobalAlloc::Memory { .. } => AllocKind::LiveData,
                GlobalAlloc::Function { .. } => bug!("We already checked function pointers above"),
                GlobalAlloc::VTable { .. } => AllocKind::VTable,
            };
            return AllocInfo::new(size, align, kind, mutbl);
        }

        // # Dead pointers
        let (size, align) = *self
            .memory
            .dead_alloc_map
            .get(&id)
            .expect("deallocated pointers should all be recorded in `dead_alloc_map`");
        AllocInfo::new(size, align, AllocKind::Dead, Mutability::Not)
    }

    /// Obtain the size and alignment of a *live* allocation.
    fn get_live_alloc_size_and_align(
        &self,
        id: AllocId,
        msg: CheckInAllocMsg,
    ) -> InterpResult<'tcx, (Size, Align)> {
        let info = self.get_alloc_info(id);
        if matches!(info.kind, AllocKind::Dead) {
            throw_ub!(PointerUseAfterFree(id, msg))
        }
        interp_ok((info.size, info.align))
    }

    fn get_fn_alloc(&self, id: AllocId) -> Option<FnVal<'tcx, M::ExtraFnVal>> {
        if let Some(extra) = self.memory.extra_fn_ptr_map.get(&id) {
            Some(FnVal::Other(*extra))
        } else {
            match self.tcx.try_get_global_alloc(id) {
                Some(GlobalAlloc::Function { instance, .. }) => Some(FnVal::Instance(instance)),
                _ => None,
            }
        }
    }

    pub fn get_ptr_fn(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
    ) -> InterpResult<'tcx, FnVal<'tcx, M::ExtraFnVal>> {
        trace!("get_ptr_fn({:?})", ptr);
        let (alloc_id, offset, _prov) = self.ptr_get_alloc_id(ptr, 0)?;
        if offset.bytes() != 0 {
            throw_ub!(InvalidFunctionPointer(Pointer::new(alloc_id, offset)))
        }
        self.get_fn_alloc(alloc_id)
            .ok_or_else(|| err_ub!(InvalidFunctionPointer(Pointer::new(alloc_id, offset))))
            .into()
    }

    /// Get the dynamic type of the given vtable pointer.
    /// If `expected_trait` is `Some`, it must be a vtable for the given trait.
    pub fn get_ptr_vtable_ty(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        expected_trait: Option<&'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>>,
    ) -> InterpResult<'tcx, Ty<'tcx>> {
        trace!("get_ptr_vtable({:?})", ptr);
        let (alloc_id, offset, _tag) = self.ptr_get_alloc_id(ptr, 0)?;
        if offset.bytes() != 0 {
            throw_ub!(InvalidVTablePointer(Pointer::new(alloc_id, offset)))
        }
        let Some(GlobalAlloc::VTable(ty, vtable_dyn_type)) =
            self.tcx.try_get_global_alloc(alloc_id)
        else {
            throw_ub!(InvalidVTablePointer(Pointer::new(alloc_id, offset)))
        };
        if let Some(expected_dyn_type) = expected_trait {
            self.check_vtable_for_type(vtable_dyn_type, expected_dyn_type)?;
        }
        interp_ok(ty)
    }

    pub fn alloc_mark_immutable(&mut self, id: AllocId) -> InterpResult<'tcx> {
        self.get_alloc_raw_mut(id)?.0.mutability = Mutability::Not;
        interp_ok(())
    }

    /// Handle the effect an FFI call might have on the state of allocations.
    /// This overapproximates the modifications which external code might make to memory:
    /// We set all reachable allocations as initialized, mark all reachable provenances as exposed
    /// and overwrite them with `Provenance::WILDCARD`.
    ///
    /// The allocations in `ids` are assumed to be already exposed.
    pub fn prepare_for_native_call(&mut self, ids: Vec<AllocId>) -> InterpResult<'tcx> {
        let mut done = FxHashSet::default();
        let mut todo = ids;
        while let Some(id) = todo.pop() {
            if !done.insert(id) {
                // We already saw this allocation before, don't process it again.
                continue;
            }
            let info = self.get_alloc_info(id);

            // If there is no data behind this pointer, skip this.
            if !matches!(info.kind, AllocKind::LiveData) {
                continue;
            }

            // Expose all provenances in this allocation, and add them to `todo`.
            let alloc = self.get_alloc_raw(id)?;
            for prov in alloc.provenance().provenances() {
                M::expose_provenance(self, prov)?;
                if let Some(id) = prov.get_alloc_id() {
                    todo.push(id);
                }
            }
            // Also expose the provenance of the interpreter-level allocation, so it can
            // be read by FFI. The `black_box` is defensive programming as LLVM likes
            // to (incorrectly) optimize away ptr2int casts whose result is unused.
            std::hint::black_box(alloc.get_bytes_unchecked_raw().expose_provenance());

            // Prepare for possible write from native code if mutable.
            if info.mutbl.is_mut() {
                self.get_alloc_raw_mut(id)?
                    .0
                    .prepare_for_native_write()
                    .map_err(|e| e.to_interp_error(id))?;
            }
        }
        interp_ok(())
    }

    /// Create a lazy debug printer that prints the given allocation and all allocations it points
    /// to, recursively.
    #[must_use]
    pub fn dump_alloc<'a>(&'a self, id: AllocId) -> DumpAllocs<'a, 'tcx, M> {
        self.dump_allocs(vec![id])
    }

    /// Create a lazy debug printer for a list of allocations and all allocations they point to,
    /// recursively.
    #[must_use]
    pub fn dump_allocs<'a>(&'a self, mut allocs: Vec<AllocId>) -> DumpAllocs<'a, 'tcx, M> {
        allocs.sort();
        allocs.dedup();
        DumpAllocs { ecx: self, allocs }
    }

    /// Print the allocation's bytes, without any nested allocations.
    pub fn print_alloc_bytes_for_diagnostics(&self, id: AllocId) -> String {
        // Using the "raw" access to avoid the `before_alloc_read` hook, we specifically
        // want to be able to read all memory for diagnostics, even if that is cyclic.
        let alloc = self.get_alloc_raw(id).unwrap();
        let mut bytes = String::new();
        if alloc.size() != Size::ZERO {
            bytes = "\n".into();
            // FIXME(translation) there might be pieces that are translatable.
            rustc_middle::mir::pretty::write_allocation_bytes(*self.tcx, alloc, &mut bytes, "    ")
                .unwrap();
        }
        bytes
    }

    /// Find leaked allocations, remove them from memory and return them. Allocations reachable from
    /// `static_roots` or a `Global` allocation are not considered leaked, as well as leaks whose
    /// kind's `may_leak()` returns true.
    ///
    /// This is highly destructive, no more execution can happen after this!
    pub fn take_leaked_allocations(
        &mut self,
        static_roots: impl FnOnce(&Self) -> &[AllocId],
    ) -> Vec<(AllocId, MemoryKind<M::MemoryKind>, Allocation<M::Provenance, M::AllocExtra, M::Bytes>)>
    {
        // Collect the set of allocations that are *reachable* from `Global` allocations.
        let reachable = {
            let mut reachable = FxHashSet::default();
            let global_kind = M::GLOBAL_KIND.map(MemoryKind::Machine);
            let mut todo: Vec<_> =
                self.memory.alloc_map.filter_map_collect(move |&id, &(kind, _)| {
                    if Some(kind) == global_kind { Some(id) } else { None }
                });
            todo.extend(static_roots(self));
            while let Some(id) = todo.pop() {
                if reachable.insert(id) {
                    // This is a new allocation, add the allocation it points to `todo`.
                    if let Some((_, alloc)) = self.memory.alloc_map.get(id) {
                        todo.extend(
                            alloc.provenance().provenances().filter_map(|prov| prov.get_alloc_id()),
                        );
                    }
                }
            }
            reachable
        };

        // All allocations that are *not* `reachable` and *not* `may_leak` are considered leaking.
        let leaked: Vec<_> = self.memory.alloc_map.filter_map_collect(|&id, &(kind, _)| {
            if kind.may_leak() || reachable.contains(&id) { None } else { Some(id) }
        });
        let mut result = Vec::new();
        for &id in leaked.iter() {
            let (kind, alloc) = self.memory.alloc_map.remove(&id).unwrap();
            result.push((id, kind, alloc));
        }
        result
    }

    /// Runs the closure in "validation" mode, which means the machine's memory read hooks will be
    /// suppressed. Needless to say, this must only be set with great care! Cannot be nested.
    ///
    /// We do this so Miri's allocation access tracking does not show the validation
    /// reads as spurious accesses.
    pub fn run_for_validation_mut<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        // This deliberately uses `==` on `bool` to follow the pattern
        // `assert!(val.replace(new) == old)`.
        assert!(
            self.memory.validation_in_progress.replace(true) == false,
            "`validation_in_progress` was already set"
        );
        let res = f(self);
        assert!(
            self.memory.validation_in_progress.replace(false) == true,
            "`validation_in_progress` was unset by someone else"
        );
        res
    }

    /// Runs the closure in "validation" mode, which means the machine's memory read hooks will be
    /// suppressed. Needless to say, this must only be set with great care! Cannot be nested.
    ///
    /// We do this so Miri's allocation access tracking does not show the validation
    /// reads as spurious accesses.
    pub fn run_for_validation_ref<R>(&self, f: impl FnOnce(&Self) -> R) -> R {
        // This deliberately uses `==` on `bool` to follow the pattern
        // `assert!(val.replace(new) == old)`.
        assert!(
            self.memory.validation_in_progress.replace(true) == false,
            "`validation_in_progress` was already set"
        );
        let res = f(self);
        assert!(
            self.memory.validation_in_progress.replace(false) == true,
            "`validation_in_progress` was unset by someone else"
        );
        res
    }

    pub(super) fn validation_in_progress(&self) -> bool {
        self.memory.validation_in_progress.get()
    }
}

#[doc(hidden)]
/// There's no way to use this directly, it's just a helper struct for the `dump_alloc(s)` methods.
pub struct DumpAllocs<'a, 'tcx, M: Machine<'tcx>> {
    ecx: &'a InterpCx<'tcx, M>,
    allocs: Vec<AllocId>,
}

impl<'a, 'tcx, M: Machine<'tcx>> std::fmt::Debug for DumpAllocs<'a, 'tcx, M> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Cannot be a closure because it is generic in `Prov`, `Extra`.
        fn write_allocation_track_relocs<'tcx, Prov: Provenance, Extra, Bytes: AllocBytes>(
            fmt: &mut std::fmt::Formatter<'_>,
            tcx: TyCtxt<'tcx>,
            allocs_to_print: &mut VecDeque<AllocId>,
            alloc: &Allocation<Prov, Extra, Bytes>,
        ) -> std::fmt::Result {
            for alloc_id in alloc.provenance().provenances().filter_map(|prov| prov.get_alloc_id())
            {
                allocs_to_print.push_back(alloc_id);
            }
            write!(fmt, "{}", display_allocation(tcx, alloc))
        }

        let mut allocs_to_print: VecDeque<_> = self.allocs.iter().copied().collect();
        // `allocs_printed` contains all allocations that we have already printed.
        let mut allocs_printed = FxHashSet::default();

        while let Some(id) = allocs_to_print.pop_front() {
            if !allocs_printed.insert(id) {
                // Already printed, so skip this.
                continue;
            }

            write!(fmt, "{id:?}")?;
            match self.ecx.memory.alloc_map.get(id) {
                Some((kind, alloc)) => {
                    // normal alloc
                    write!(fmt, " ({kind}, ")?;
                    write_allocation_track_relocs(
                        &mut *fmt,
                        *self.ecx.tcx,
                        &mut allocs_to_print,
                        alloc,
                    )?;
                }
                None => {
                    // global alloc
                    match self.ecx.tcx.try_get_global_alloc(id) {
                        Some(GlobalAlloc::Memory(alloc)) => {
                            write!(fmt, " (unchanged global, ")?;
                            write_allocation_track_relocs(
                                &mut *fmt,
                                *self.ecx.tcx,
                                &mut allocs_to_print,
                                alloc.inner(),
                            )?;
                        }
                        Some(GlobalAlloc::Function { instance, .. }) => {
                            write!(fmt, " (fn: {instance})")?;
                        }
                        Some(GlobalAlloc::VTable(ty, dyn_ty)) => {
                            write!(fmt, " (vtable: impl {dyn_ty} for {ty})")?;
                        }
                        Some(GlobalAlloc::Static(did)) => {
                            write!(fmt, " (static: {})", self.ecx.tcx.def_path_str(did))?;
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
impl<'a, 'tcx, Prov: Provenance, Extra, Bytes: AllocBytes>
    AllocRefMut<'a, 'tcx, Prov, Extra, Bytes>
{
    pub fn as_ref<'b>(&'b self) -> AllocRef<'b, 'tcx, Prov, Extra, Bytes> {
        AllocRef { alloc: self.alloc, range: self.range, tcx: self.tcx, alloc_id: self.alloc_id }
    }

    /// `range` is relative to this allocation reference, not the base of the allocation.
    pub fn write_scalar(&mut self, range: AllocRange, val: Scalar<Prov>) -> InterpResult<'tcx> {
        let range = self.range.subrange(range);
        debug!("write_scalar at {:?}{range:?}: {val:?}", self.alloc_id);

        self.alloc
            .write_scalar(&self.tcx, range, val)
            .map_err(|e| e.to_interp_error(self.alloc_id))
            .into()
    }

    /// `offset` is relative to this allocation reference, not the base of the allocation.
    pub fn write_ptr_sized(&mut self, offset: Size, val: Scalar<Prov>) -> InterpResult<'tcx> {
        self.write_scalar(alloc_range(offset, self.tcx.data_layout().pointer_size), val)
    }

    /// Mark the given sub-range (relative to this allocation reference) as uninitialized.
    pub fn write_uninit(&mut self, range: AllocRange) -> InterpResult<'tcx> {
        let range = self.range.subrange(range);

        self.alloc
            .write_uninit(&self.tcx, range)
            .map_err(|e| e.to_interp_error(self.alloc_id))
            .into()
    }

    /// Mark the entire referenced range as uninitialized
    pub fn write_uninit_full(&mut self) -> InterpResult<'tcx> {
        self.alloc
            .write_uninit(&self.tcx, self.range)
            .map_err(|e| e.to_interp_error(self.alloc_id))
            .into()
    }

    /// Remove all provenance in the reference range.
    pub fn clear_provenance(&mut self) -> InterpResult<'tcx> {
        self.alloc
            .clear_provenance(&self.tcx, self.range)
            .map_err(|e| e.to_interp_error(self.alloc_id))
            .into()
    }
}

impl<'a, 'tcx, Prov: Provenance, Extra, Bytes: AllocBytes> AllocRef<'a, 'tcx, Prov, Extra, Bytes> {
    /// `range` is relative to this allocation reference, not the base of the allocation.
    pub fn read_scalar(
        &self,
        range: AllocRange,
        read_provenance: bool,
    ) -> InterpResult<'tcx, Scalar<Prov>> {
        let range = self.range.subrange(range);
        self.alloc
            .read_scalar(&self.tcx, range, read_provenance)
            .map_err(|e| e.to_interp_error(self.alloc_id))
            .into()
    }

    /// `range` is relative to this allocation reference, not the base of the allocation.
    pub fn read_integer(&self, range: AllocRange) -> InterpResult<'tcx, Scalar<Prov>> {
        self.read_scalar(range, /*read_provenance*/ false)
    }

    /// `offset` is relative to this allocation reference, not the base of the allocation.
    pub fn read_pointer(&self, offset: Size) -> InterpResult<'tcx, Scalar<Prov>> {
        self.read_scalar(
            alloc_range(offset, self.tcx.data_layout().pointer_size),
            /*read_provenance*/ true,
        )
    }

    /// `range` is relative to this allocation reference, not the base of the allocation.
    pub fn get_bytes_strip_provenance<'b>(&'b self) -> InterpResult<'tcx, &'a [u8]> {
        self.alloc
            .get_bytes_strip_provenance(&self.tcx, self.range)
            .map_err(|e| e.to_interp_error(self.alloc_id))
            .into()
    }

    /// Returns whether the allocation has provenance anywhere in the range of the `AllocRef`.
    pub fn has_provenance(&self) -> bool {
        !self.alloc.provenance().range_empty(self.range, &self.tcx)
    }
}

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Reads the given number of bytes from memory, and strips their provenance if possible.
    /// Returns them as a slice.
    ///
    /// Performs appropriate bounds checks.
    pub fn read_bytes_ptr_strip_provenance(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
    ) -> InterpResult<'tcx, &[u8]> {
        let Some(alloc_ref) = self.get_ptr_alloc(ptr, size)? else {
            // zero-sized access
            return interp_ok(&[]);
        };
        // Side-step AllocRef and directly access the underlying bytes more efficiently.
        // (We are staying inside the bounds here so all is good.)
        interp_ok(
            alloc_ref
                .alloc
                .get_bytes_strip_provenance(&alloc_ref.tcx, alloc_ref.range)
                .map_err(|e| e.to_interp_error(alloc_ref.alloc_id))?,
        )
    }

    /// Writes the given stream of bytes into memory.
    ///
    /// Performs appropriate bounds checks.
    pub fn write_bytes_ptr(
        &mut self,
        ptr: Pointer<Option<M::Provenance>>,
        src: impl IntoIterator<Item = u8>,
    ) -> InterpResult<'tcx> {
        let mut src = src.into_iter();
        let (lower, upper) = src.size_hint();
        let len = upper.expect("can only write bounded iterators");
        assert_eq!(lower, len, "can only write iterators with a precise length");

        let size = Size::from_bytes(len);
        let Some(alloc_ref) = self.get_ptr_alloc_mut(ptr, size)? else {
            // zero-sized access
            assert_matches!(src.next(), None, "iterator said it was empty but returned an element");
            return interp_ok(());
        };

        // Side-step AllocRef and directly access the underlying bytes more efficiently.
        // (We are staying inside the bounds here and all bytes do get overwritten so all is good.)
        let alloc_id = alloc_ref.alloc_id;
        let bytes = alloc_ref
            .alloc
            .get_bytes_unchecked_for_overwrite(&alloc_ref.tcx, alloc_ref.range)
            .map_err(move |e| e.to_interp_error(alloc_id))?;
        // `zip` would stop when the first iterator ends; we want to definitely
        // cover all of `bytes`.
        for dest in bytes {
            *dest = src.next().expect("iterator was shorter than it said it would be");
        }
        assert_matches!(src.next(), None, "iterator was longer than it said it would be");
        interp_ok(())
    }

    pub fn mem_copy(
        &mut self,
        src: Pointer<Option<M::Provenance>>,
        dest: Pointer<Option<M::Provenance>>,
        size: Size,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        self.mem_copy_repeatedly(src, dest, size, 1, nonoverlapping)
    }

    /// Performs `num_copies` many copies of `size` many bytes from `src` to `dest + i*size` (where
    /// `i` is the index of the copy).
    ///
    /// Either `nonoverlapping` must be true or `num_copies` must be 1; doing repeated copies that
    /// may overlap is not supported.
    pub fn mem_copy_repeatedly(
        &mut self,
        src: Pointer<Option<M::Provenance>>,
        dest: Pointer<Option<M::Provenance>>,
        size: Size,
        num_copies: u64,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        let tcx = self.tcx;
        // We need to do our own bounds-checks.
        let src_parts = self.get_ptr_access(src, size)?;
        let dest_parts = self.get_ptr_access(dest, size * num_copies)?; // `Size` multiplication

        // FIXME: we look up both allocations twice here, once before for the `check_ptr_access`
        // and once below to get the underlying `&[mut] Allocation`.

        // Source alloc preparations and access hooks.
        let Some((src_alloc_id, src_offset, src_prov)) = src_parts else {
            // Zero-sized *source*, that means dest is also zero-sized and we have nothing to do.
            return interp_ok(());
        };
        let src_alloc = self.get_alloc_raw(src_alloc_id)?;
        let src_range = alloc_range(src_offset, size);
        assert!(!self.memory.validation_in_progress.get(), "we can't be copying during validation");
        // For the overlapping case, it is crucial that we trigger the read hook
        // before the write hook -- the aliasing model cares about the order.
        M::before_memory_read(
            tcx,
            &self.machine,
            &src_alloc.extra,
            src,
            (src_alloc_id, src_prov),
            src_range,
        )?;
        // We need the `dest` ptr for the next operation, so we get it now.
        // We already did the source checks and called the hooks so we are good to return early.
        let Some((dest_alloc_id, dest_offset, dest_prov)) = dest_parts else {
            // Zero-sized *destination*.
            return interp_ok(());
        };

        // Prepare getting source provenance.
        let src_bytes = src_alloc.get_bytes_unchecked(src_range).as_ptr(); // raw ptr, so we can also get a ptr to the destination allocation
        // first copy the provenance to a temporary buffer, because
        // `get_bytes_mut` will clear the provenance, which is correct,
        // since we don't want to keep any provenance at the target.
        // This will also error if copying partial provenance is not supported.
        let provenance = src_alloc
            .provenance()
            .prepare_copy(src_range, dest_offset, num_copies, self)
            .map_err(|e| e.to_interp_error(dest_alloc_id))?;
        // Prepare a copy of the initialization mask.
        let init = src_alloc.init_mask().prepare_copy(src_range);

        // Destination alloc preparations and access hooks.
        let (dest_alloc, extra) = self.get_alloc_raw_mut(dest_alloc_id)?;
        let dest_range = alloc_range(dest_offset, size * num_copies);
        M::before_memory_write(
            tcx,
            extra,
            &mut dest_alloc.extra,
            dest,
            (dest_alloc_id, dest_prov),
            dest_range,
        )?;
        // Yes we do overwrite all bytes in `dest_bytes`.
        let dest_bytes = dest_alloc
            .get_bytes_unchecked_for_overwrite_ptr(&tcx, dest_range)
            .map_err(|e| e.to_interp_error(dest_alloc_id))?
            .as_mut_ptr();

        if init.no_bytes_init() {
            // Fast path: If all bytes are `uninit` then there is nothing to copy. The target range
            // is marked as uninitialized but we otherwise omit changing the byte representation which may
            // be arbitrary for uninitialized bytes.
            // This also avoids writing to the target bytes so that the backing allocation is never
            // touched if the bytes stay uninitialized for the whole interpreter execution. On contemporary
            // operating system this can avoid physically allocating the page.
            dest_alloc
                .write_uninit(&tcx, dest_range)
                .map_err(|e| e.to_interp_error(dest_alloc_id))?;
            // We can forget about the provenance, this is all not initialized anyway.
            return interp_ok(());
        }

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
        // The pointers above remain valid even if the `HashMap` table is moved around because they
        // point into the `Vec` storing the bytes.
        unsafe {
            if src_alloc_id == dest_alloc_id {
                if nonoverlapping {
                    // `Size` additions
                    if (src_offset <= dest_offset && src_offset + size > dest_offset)
                        || (dest_offset <= src_offset && dest_offset + size > src_offset)
                    {
                        throw_ub_custom!(fluent::const_eval_copy_nonoverlapping_overlapping);
                    }
                }
            }
            if num_copies > 1 {
                assert!(nonoverlapping, "multi-copy only supported in non-overlapping mode");
            }

            let size_in_bytes = size.bytes_usize();
            // For particularly large arrays (where this is perf-sensitive) it's common that
            // we're writing a single byte repeatedly. So, optimize that case to a memset.
            if size_in_bytes == 1 {
                debug_assert!(num_copies >= 1); // we already handled the zero-sized cases above.
                // SAFETY: `src_bytes` would be read from anyway by `copy` below (num_copies >= 1).
                let value = *src_bytes;
                dest_bytes.write_bytes(value, (size * num_copies).bytes_usize());
            } else if src_alloc_id == dest_alloc_id {
                let mut dest_ptr = dest_bytes;
                for _ in 0..num_copies {
                    // Here we rely on `src` and `dest` being non-overlapping if there is more than
                    // one copy.
                    ptr::copy(src_bytes, dest_ptr, size_in_bytes);
                    dest_ptr = dest_ptr.add(size_in_bytes);
                }
            } else {
                let mut dest_ptr = dest_bytes;
                for _ in 0..num_copies {
                    ptr::copy_nonoverlapping(src_bytes, dest_ptr, size_in_bytes);
                    dest_ptr = dest_ptr.add(size_in_bytes);
                }
            }
        }

        // now fill in all the "init" data
        dest_alloc.init_mask_apply_copy(
            init,
            alloc_range(dest_offset, size), // just a single copy (i.e., not full `dest_range`)
            num_copies,
        );
        // copy the provenance to the destination
        dest_alloc.provenance_apply_copy(provenance);

        interp_ok(())
    }
}

/// Machine pointer introspection.
impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Test if this value might be null.
    /// If the machine does not support ptr-to-int casts, this is conservative.
    pub fn scalar_may_be_null(&self, scalar: Scalar<M::Provenance>) -> InterpResult<'tcx, bool> {
        match scalar.try_to_scalar_int() {
            Ok(int) => interp_ok(int.is_null()),
            Err(_) => {
                // We can't cast this pointer to an integer. Can only happen during CTFE.
                let ptr = scalar.to_pointer(self)?;
                match self.ptr_try_get_alloc_id(ptr, 0) {
                    Ok((alloc_id, offset, _)) => {
                        let info = self.get_alloc_info(alloc_id);
                        // If the pointer is in-bounds (including "at the end"), it is definitely not null.
                        if offset <= info.size {
                            return interp_ok(false);
                        }
                        // If the allocation is N-aligned, and the offset is not divisible by N,
                        // then `base + offset` has a non-zero remainder after division by `N`,
                        // which means `base + offset` cannot be null.
                        if offset.bytes() % info.align.bytes() != 0 {
                            return interp_ok(false);
                        }
                        // We don't know enough, this might be null.
                        interp_ok(true)
                    }
                    Err(_offset) => bug!("a non-int scalar is always a pointer"),
                }
            }
        }
    }

    /// Turning a "maybe pointer" into a proper pointer (and some information
    /// about where it points), or an absolute address.
    ///
    /// `size` says how many bytes of memory are expected at that pointer. This is largely only used
    /// for error messages; however, the *sign* of `size` can be used to disambiguate situations
    /// where a wildcard pointer sits right in between two allocations.
    /// It is almost always okay to just set the size to 0; this will be treated like a positive size
    /// for handling wildcard pointers.
    ///
    /// The result must be used immediately; it is not allowed to convert
    /// the returned data back into a `Pointer` and store that in machine state.
    /// (In fact that's not even possible since `M::ProvenanceExtra` is generic and
    /// we don't have an operation to turn it back into `M::Provenance`.)
    pub fn ptr_try_get_alloc_id(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: i64,
    ) -> Result<(AllocId, Size, M::ProvenanceExtra), u64> {
        match ptr.into_pointer_or_addr() {
            Ok(ptr) => match M::ptr_get_alloc(self, ptr, size) {
                Some((alloc_id, offset, extra)) => Ok((alloc_id, offset, extra)),
                None => {
                    assert!(M::Provenance::OFFSET_IS_ADDR);
                    let (_, addr) = ptr.into_parts();
                    Err(addr.bytes())
                }
            },
            Err(addr) => Err(addr.bytes()),
        }
    }

    /// Turning a "maybe pointer" into a proper pointer (and some information about where it points).
    ///
    /// `size` says how many bytes of memory are expected at that pointer. This is largely only used
    /// for error messages; however, the *sign* of `size` can be used to disambiguate situations
    /// where a wildcard pointer sits right in between two allocations.
    /// It is almost always okay to just set the size to 0; this will be treated like a positive size
    /// for handling wildcard pointers.
    ///
    /// The result must be used immediately; it is not allowed to convert
    /// the returned data back into a `Pointer` and store that in machine state.
    /// (In fact that's not even possible since `M::ProvenanceExtra` is generic and
    /// we don't have an operation to turn it back into `M::Provenance`.)
    #[inline(always)]
    pub fn ptr_get_alloc_id(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: i64,
    ) -> InterpResult<'tcx, (AllocId, Size, M::ProvenanceExtra)> {
        self.ptr_try_get_alloc_id(ptr, size)
            .map_err(|offset| {
                err_ub!(DanglingIntPointer {
                    addr: offset,
                    inbounds_size: size,
                    msg: CheckInAllocMsg::InboundsTest
                })
            })
            .into()
    }
}
