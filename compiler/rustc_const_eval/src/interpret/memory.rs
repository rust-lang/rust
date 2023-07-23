//! The memory subsystem.
//!
//! Generally, we use `Pointer` to denote memory addresses. However, some operations
//! have a "size"-like parameter, and they take `Scalar` for the address because
//! if the size is 0, then the pointer can also be a (properly aligned, non-null)
//! integer. It is crucial that these operations call `check_align` *before*
//! short-circuiting the empty case!

use std::assert_matches::assert_matches;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt;
use std::ptr;

use rustc_ast::Mutability;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::mir::display_allocation;
use rustc_middle::ty::{self, Instance, ParamEnv, Ty, TyCtxt};
use rustc_target::abi::{Align, HasDataLayout, Size};

use crate::const_eval::CheckAlignment;
use crate::fluent_generated as fluent;

use super::{
    alloc_range, AllocBytes, AllocId, AllocMap, AllocRange, Allocation, CheckInAllocMsg,
    GlobalAlloc, InterpCx, InterpResult, Machine, MayLeak, Pointer, PointerArithmetic, Provenance,
    Scalar,
};

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
            MemoryKind::Machine(m) => write!(f, "{}", m),
        }
    }
}

/// The return value of `get_alloc_info` indicates the "kind" of the allocation.
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

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    pub fn new() -> Self {
        Memory {
            alloc_map: M::MemoryMap::default(),
            extra_fn_ptr_map: FxHashMap::default(),
            dead_alloc_map: FxHashMap::default(),
        }
    }

    /// This is used by [priroda](https://github.com/oli-obk/priroda)
    pub fn alloc_map(&self) -> &M::MemoryMap {
        &self.alloc_map
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
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
    pub fn global_base_pointer(
        &self,
        ptr: Pointer<AllocId>,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let alloc_id = ptr.provenance;
        // We need to handle `extern static`.
        match self.tcx.try_get_global_alloc(alloc_id) {
            Some(GlobalAlloc::Static(def_id)) if self.tcx.is_thread_local_static(def_id) => {
                bug!("global memory cannot point to thread-local static")
            }
            Some(GlobalAlloc::Static(def_id)) if self.tcx.is_foreign_item(def_id) => {
                return M::extern_static_base_pointer(self, def_id);
            }
            _ => {}
        }
        // And we need to get the provenance.
        M::adjust_alloc_base_pointer(self, ptr)
    }

    pub fn create_fn_alloc_ptr(
        &mut self,
        fn_val: FnVal<'tcx, M::ExtraFnVal>,
    ) -> Pointer<M::Provenance> {
        let id = match fn_val {
            FnVal::Instance(instance) => self.tcx.create_fn_alloc(instance),
            FnVal::Other(extra) => {
                // FIXME(RalfJung): Should we have a cache here?
                let id = self.tcx.reserve_alloc_id();
                let old = self.memory.extra_fn_ptr_map.insert(id, extra);
                assert!(old.is_none());
                id
            }
        };
        // Functions are global allocations, so make sure we get the right base pointer.
        // We know this is not an `extern static` so this cannot fail.
        self.global_base_pointer(Pointer::from(id)).unwrap()
    }

    pub fn allocate_ptr(
        &mut self,
        size: Size,
        align: Align,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let alloc = if M::PANIC_ON_ALLOC_FAIL {
            Allocation::uninit(size, align)
        } else {
            Allocation::try_uninit(size, align)?
        };
        self.allocate_raw_ptr(alloc, kind)
    }

    pub fn allocate_bytes_ptr(
        &mut self,
        bytes: &[u8],
        align: Align,
        kind: MemoryKind<M::MemoryKind>,
        mutability: Mutability,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let alloc = Allocation::from_bytes(bytes, align, mutability);
        self.allocate_raw_ptr(alloc, kind)
    }

    /// This can fail only if `alloc` contains provenance.
    pub fn allocate_raw_ptr(
        &mut self,
        alloc: Allocation,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let id = self.tcx.reserve_alloc_id();
        debug_assert_ne!(
            Some(kind),
            M::GLOBAL_KIND.map(MemoryKind::Machine),
            "dynamically allocating global memory"
        );
        let alloc = M::adjust_allocation(self, id, Cow::Owned(alloc), Some(kind))?;
        self.memory.alloc_map.insert(id, (kind, alloc.into_owned()));
        M::adjust_alloc_base_pointer(self, Pointer::from(id))
    }

    pub fn reallocate_ptr(
        &mut self,
        ptr: Pointer<Option<M::Provenance>>,
        old_size_and_align: Option<(Size, Align)>,
        new_size: Size,
        new_align: Align,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx, Pointer<M::Provenance>> {
        let (alloc_id, offset, _prov) = self.ptr_get_alloc_id(ptr)?;
        if offset.bytes() != 0 {
            throw_ub_custom!(
                fluent::const_eval_realloc_or_alloc_with_offset,
                ptr = format!("{ptr:?}"),
                kind = "realloc"
            );
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc".
        // This happens so rarely, the perf advantage is outweighed by the maintenance cost.
        let new_ptr = self.allocate_ptr(new_size, new_align, kind)?;
        let old_size = match old_size_and_align {
            Some((size, _align)) => size,
            None => self.get_alloc_raw(alloc_id)?.size(),
        };
        // This will also call the access hooks.
        self.mem_copy(
            ptr,
            Align::ONE,
            new_ptr.into(),
            Align::ONE,
            old_size.min(new_size),
            /*nonoverlapping*/ true,
        )?;
        self.deallocate_ptr(ptr, old_size_and_align, kind)?;

        Ok(new_ptr)
    }

    #[instrument(skip(self), level = "debug")]
    pub fn deallocate_ptr(
        &mut self,
        ptr: Pointer<Option<M::Provenance>>,
        old_size_and_align: Option<(Size, Align)>,
        kind: MemoryKind<M::MemoryKind>,
    ) -> InterpResult<'tcx> {
        let (alloc_id, offset, prov) = self.ptr_get_alloc_id(ptr)?;
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
                Some(GlobalAlloc::Function(..)) => {
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
                None => err_ub!(PointerUseAfterFree(alloc_id)),
            }
            .into());
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
            *self.tcx,
            &mut self.machine,
            &mut alloc.extra,
            (alloc_id, prov),
            alloc_range(Size::ZERO, size),
        )?;

        // Don't forget to remember size and align of this now-dead allocation
        let old = self.memory.dead_alloc_map.insert(alloc_id, (size, alloc.align));
        if old.is_some() {
            bug!("Nothing can be deallocated twice");
        }

        Ok(())
    }

    /// Internal helper function to determine the allocation and offset of a pointer (if any).
    #[inline(always)]
    fn get_ptr_access(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Option<(AllocId, Size, M::ProvenanceExtra)>> {
        self.check_and_deref_ptr(
            ptr,
            size,
            align,
            M::enforce_alignment(self),
            CheckInAllocMsg::MemoryAccessTest,
            |alloc_id, offset, prov| {
                let (size, align) = self.get_live_alloc_size_and_align(alloc_id)?;
                Ok((size, align, (alloc_id, offset, prov)))
            },
        )
    }

    /// Check if the given pointer points to live memory of given `size` and `align`
    /// (ignoring `M::enforce_alignment`). The caller can control the error message for the
    /// out-of-bounds case.
    #[inline(always)]
    pub fn check_ptr_access_align(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
        align: Align,
        msg: CheckInAllocMsg,
    ) -> InterpResult<'tcx> {
        self.check_and_deref_ptr(
            ptr,
            size,
            align,
            CheckAlignment::Error,
            msg,
            |alloc_id, _, _| {
                let (size, align) = self.get_live_alloc_size_and_align(alloc_id)?;
                Ok((size, align, ()))
            },
        )?;
        Ok(())
    }

    /// Low-level helper function to check if a ptr is in-bounds and potentially return a reference
    /// to the allocation it points to. Supports both shared and mutable references, as the actual
    /// checking is offloaded to a helper closure. `align` defines whether and which alignment check
    /// is done. Returns `None` for size 0, and otherwise `Some` of what `alloc_size` returned.
    fn check_and_deref_ptr<T>(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
        align: Align,
        check: CheckAlignment,
        msg: CheckInAllocMsg,
        alloc_size: impl FnOnce(
            AllocId,
            Size,
            M::ProvenanceExtra,
        ) -> InterpResult<'tcx, (Size, Align, T)>,
    ) -> InterpResult<'tcx, Option<T>> {
        Ok(match self.ptr_try_get_alloc_id(ptr) {
            Err(addr) => {
                // We couldn't get a proper allocation. This is only okay if the access size is 0,
                // and the address is not null.
                if size.bytes() > 0 || addr == 0 {
                    throw_ub!(DanglingIntPointer(addr, msg));
                }
                // Must be aligned.
                if check.should_check() {
                    self.check_offset_align(addr, align, check)?;
                }
                None
            }
            Ok((alloc_id, offset, prov)) => {
                let (alloc_size, alloc_align, ret_val) = alloc_size(alloc_id, offset, prov)?;
                // Test bounds. This also ensures non-null.
                // It is sufficient to check this for the end pointer. Also check for overflow!
                if offset.checked_add(size, &self.tcx).map_or(true, |end| end > alloc_size) {
                    throw_ub!(PointerOutOfBounds {
                        alloc_id,
                        alloc_size,
                        ptr_offset: self.target_usize_to_isize(offset.bytes()),
                        ptr_size: size,
                        msg,
                    })
                }
                // Ensure we never consider the null pointer dereferenceable.
                if M::Provenance::OFFSET_IS_ADDR {
                    assert_ne!(ptr.addr(), Size::ZERO);
                }
                // Test align. Check this last; if both bounds and alignment are violated
                // we want the error to be about the bounds.
                if check.should_check() {
                    if M::use_addr_for_alignment_check(self) {
                        // `use_addr_for_alignment_check` can only be true if `OFFSET_IS_ADDR` is true.
                        self.check_offset_align(ptr.addr().bytes(), align, check)?;
                    } else {
                        // Check allocation alignment and offset alignment.
                        if alloc_align.bytes() < align.bytes() {
                            M::alignment_check_failed(self, alloc_align, align, check)?;
                        }
                        self.check_offset_align(offset.bytes(), align, check)?;
                    }
                }

                // We can still be zero-sized in this branch, in which case we have to
                // return `None`.
                if size.bytes() == 0 { None } else { Some(ret_val) }
            }
        })
    }

    fn check_offset_align(
        &self,
        offset: u64,
        align: Align,
        check: CheckAlignment,
    ) -> InterpResult<'tcx> {
        if offset % align.bytes() == 0 {
            Ok(())
        } else {
            // The biggest power of two through which `offset` is divisible.
            let offset_pow2 = 1 << offset.trailing_zeros();
            M::alignment_check_failed(self, Align::from_bytes(offset_pow2).unwrap(), align, check)
        }
    }
}

/// Allocation accessors
impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
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
            Some(GlobalAlloc::Function(..)) => throw_ub!(DerefFunctionPointer(id)),
            Some(GlobalAlloc::VTable(..)) => throw_ub!(DerefVTablePointer(id)),
            None => throw_ub!(PointerUseAfterFree(id)),
            Some(GlobalAlloc::Static(def_id)) => {
                assert!(self.tcx.is_static(def_id));
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
                    throw_unsup!(ReadExternStatic(def_id));
                }

                // We don't give a span -- statics don't need that, they cannot be generic or associated.
                let val = self.ctfe_query(None, |tcx| tcx.eval_static_initializer(def_id))?;
                (val, Some(def_id))
            }
        };
        M::before_access_global(*self.tcx, &self.machine, id, alloc, def_id, is_write)?;
        // We got tcx memory. Let the machine initialize its "extra" stuff.
        M::adjust_allocation(
            self,
            id, // always use the ID we got as input, not the "hidden" one.
            Cow::Borrowed(alloc.inner()),
            M::GLOBAL_KIND.map(MemoryKind::Machine),
        )
    }

    /// Get the base address for the bytes in an `Allocation` specified by the
    /// `AllocID` passed in; error if no such allocation exists.
    ///
    /// It is up to the caller to take sufficient care when using this address:
    /// there could be provenance or uninit memory in there, and other memory
    /// accesses could invalidate the exposed pointer.
    pub fn alloc_base_addr(&self, id: AllocId) -> InterpResult<'tcx, *const u8> {
        let alloc = self.get_alloc_raw(id)?;
        Ok(alloc.base_addr())
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
            let alloc = self.get_global_alloc(id, /*is_write*/ false).map_err(Err)?;
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
    pub fn get_ptr_alloc<'a>(
        &'a self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Option<AllocRef<'a, 'tcx, M::Provenance, M::AllocExtra, M::Bytes>>>
    {
        let ptr_and_alloc = self.check_and_deref_ptr(
            ptr,
            size,
            align,
            M::enforce_alignment(self),
            CheckInAllocMsg::MemoryAccessTest,
            |alloc_id, offset, prov| {
                let alloc = self.get_alloc_raw(alloc_id)?;
                Ok((alloc.size(), alloc.align, (alloc_id, offset, prov, alloc)))
            },
        )?;
        if let Some((alloc_id, offset, prov, alloc)) = ptr_and_alloc {
            let range = alloc_range(offset, size);
            M::before_memory_read(*self.tcx, &self.machine, &alloc.extra, (alloc_id, prov), range)?;
            Ok(Some(AllocRef { alloc, range, tcx: *self.tcx, alloc_id }))
        } else {
            // Even in this branch we have to be sure that we actually access the allocation, in
            // order to ensure that `static FOO: Type = FOO;` causes a cycle error instead of
            // magically pulling *any* ZST value from the ether. However, the `get_raw` above is
            // always called when `ptr` has an `AllocId`.
            Ok(None)
        }
    }

    /// Return the `extra` field of the given allocation.
    pub fn get_alloc_extra<'a>(&'a self, id: AllocId) -> InterpResult<'tcx, &'a M::AllocExtra> {
        Ok(&self.get_alloc_raw(id)?.extra)
    }

    /// Return the `mutability` field of the given allocation.
    pub fn get_alloc_mutability<'a>(&'a self, id: AllocId) -> InterpResult<'tcx, Mutability> {
        Ok(self.get_alloc_raw(id)?.mutability)
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
        // (Cannot use `get_mut_or` since `get_global_alloc` needs `&self`.)
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
        Ok((alloc, &mut self.machine))
    }

    /// "Safe" (bounds and align-checked) allocation access.
    pub fn get_ptr_alloc_mut<'a>(
        &'a mut self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Option<AllocRefMut<'a, 'tcx, M::Provenance, M::AllocExtra, M::Bytes>>>
    {
        let parts = self.get_ptr_access(ptr, size, align)?;
        if let Some((alloc_id, offset, prov)) = parts {
            let tcx = *self.tcx;
            // FIXME: can we somehow avoid looking up the allocation twice here?
            // We cannot call `get_raw_mut` inside `check_and_deref_ptr` as that would duplicate `&mut self`.
            let (alloc, machine) = self.get_alloc_raw_mut(alloc_id)?;
            let range = alloc_range(offset, size);
            M::before_memory_write(tcx, machine, &mut alloc.extra, (alloc_id, prov), range)?;
            Ok(Some(AllocRefMut { alloc, range, tcx, alloc_id }))
        } else {
            Ok(None)
        }
    }

    /// Return the `extra` field of the given allocation.
    pub fn get_alloc_extra_mut<'a>(
        &'a mut self,
        id: AllocId,
    ) -> InterpResult<'tcx, (&'a mut M::AllocExtra, &'a mut M)> {
        let (alloc, machine) = self.get_alloc_raw_mut(id)?;
        Ok((&mut alloc.extra, machine))
    }

    /// Obtain the size and alignment of an allocation, even if that allocation has
    /// been deallocated.
    pub fn get_alloc_info(&self, id: AllocId) -> (Size, Align, AllocKind) {
        // # Regular allocations
        // Don't use `self.get_raw` here as that will
        // a) cause cycles in case `id` refers to a static
        // b) duplicate a global's allocation in miri
        if let Some((_, alloc)) = self.memory.alloc_map.get(id) {
            return (alloc.size(), alloc.align, AllocKind::LiveData);
        }

        // # Function pointers
        // (both global from `alloc_map` and local from `extra_fn_ptr_map`)
        if self.get_fn_alloc(id).is_some() {
            return (Size::ZERO, Align::ONE, AllocKind::Function);
        }

        // # Statics
        // Can't do this in the match argument, we may get cycle errors since the lock would
        // be held throughout the match.
        match self.tcx.try_get_global_alloc(id) {
            Some(GlobalAlloc::Static(def_id)) => {
                assert!(self.tcx.is_static(def_id));
                assert!(!self.tcx.is_thread_local_static(def_id));
                // Use size and align of the type.
                let ty = self
                    .tcx
                    .type_of(def_id)
                    .no_bound_vars()
                    .expect("statics should not have generic parameters");
                let layout = self.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                assert!(layout.is_sized());
                (layout.size, layout.align.abi, AllocKind::LiveData)
            }
            Some(GlobalAlloc::Memory(alloc)) => {
                // Need to duplicate the logic here, because the global allocations have
                // different associated types than the interpreter-local ones.
                let alloc = alloc.inner();
                (alloc.size(), alloc.align, AllocKind::LiveData)
            }
            Some(GlobalAlloc::Function(_)) => bug!("We already checked function pointers above"),
            Some(GlobalAlloc::VTable(..)) => {
                // No data to be accessed here. But vtables are pointer-aligned.
                return (Size::ZERO, self.tcx.data_layout.pointer_align.abi, AllocKind::VTable);
            }
            // The rest must be dead.
            None => {
                // Deallocated pointers are allowed, we should be able to find
                // them in the map.
                let (size, align) = *self
                    .memory
                    .dead_alloc_map
                    .get(&id)
                    .expect("deallocated pointers should all be recorded in `dead_alloc_map`");
                (size, align, AllocKind::Dead)
            }
        }
    }

    /// Obtain the size and alignment of a live allocation.
    pub fn get_live_alloc_size_and_align(&self, id: AllocId) -> InterpResult<'tcx, (Size, Align)> {
        let (size, align, kind) = self.get_alloc_info(id);
        if matches!(kind, AllocKind::Dead) {
            throw_ub!(PointerUseAfterFree(id))
        }
        Ok((size, align))
    }

    fn get_fn_alloc(&self, id: AllocId) -> Option<FnVal<'tcx, M::ExtraFnVal>> {
        if let Some(extra) = self.memory.extra_fn_ptr_map.get(&id) {
            Some(FnVal::Other(*extra))
        } else {
            match self.tcx.try_get_global_alloc(id) {
                Some(GlobalAlloc::Function(instance)) => Some(FnVal::Instance(instance)),
                _ => None,
            }
        }
    }

    pub fn get_ptr_fn(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
    ) -> InterpResult<'tcx, FnVal<'tcx, M::ExtraFnVal>> {
        trace!("get_ptr_fn({:?})", ptr);
        let (alloc_id, offset, _prov) = self.ptr_get_alloc_id(ptr)?;
        if offset.bytes() != 0 {
            throw_ub!(InvalidFunctionPointer(Pointer::new(alloc_id, offset)))
        }
        self.get_fn_alloc(alloc_id)
            .ok_or_else(|| err_ub!(InvalidFunctionPointer(Pointer::new(alloc_id, offset))).into())
    }

    pub fn get_ptr_vtable(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
    ) -> InterpResult<'tcx, (Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>)> {
        trace!("get_ptr_vtable({:?})", ptr);
        let (alloc_id, offset, _tag) = self.ptr_get_alloc_id(ptr)?;
        if offset.bytes() != 0 {
            throw_ub!(InvalidVTablePointer(Pointer::new(alloc_id, offset)))
        }
        match self.tcx.try_get_global_alloc(alloc_id) {
            Some(GlobalAlloc::VTable(ty, trait_ref)) => Ok((ty, trait_ref)),
            _ => throw_ub!(InvalidVTablePointer(Pointer::new(alloc_id, offset))),
        }
    }

    pub fn alloc_mark_immutable(&mut self, id: AllocId) -> InterpResult<'tcx> {
        self.get_alloc_raw_mut(id)?.0.mutability = Mutability::Not;
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
        DumpAllocs { ecx: self, allocs }
    }

    /// Find leaked allocations. Allocations reachable from `static_roots` or a `Global` allocation
    /// are not considered leaked, as well as leaks whose kind's `may_leak()` returns true.
    pub fn find_leaked_allocations(
        &self,
        static_roots: &[AllocId],
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
            todo.extend(static_roots);
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
        self.memory.alloc_map.filter_map_collect(|id, (kind, alloc)| {
            if kind.may_leak() || reachable.contains(id) {
                None
            } else {
                Some((*id, *kind, alloc.clone()))
            }
        })
    }
}

#[doc(hidden)]
/// There's no way to use this directly, it's just a helper struct for the `dump_alloc(s)` methods.
pub struct DumpAllocs<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    ecx: &'a InterpCx<'mir, 'tcx, M>,
    allocs: Vec<AllocId>,
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> std::fmt::Debug for DumpAllocs<'a, 'mir, 'tcx, M> {
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
                    write!(fmt, " ({}, ", kind)?;
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
                        Some(GlobalAlloc::Function(func)) => {
                            write!(fmt, " (fn: {func})")?;
                        }
                        Some(GlobalAlloc::VTable(ty, Some(trait_ref))) => {
                            write!(fmt, " (vtable: impl {trait_ref} for {ty})")?;
                        }
                        Some(GlobalAlloc::VTable(ty, None)) => {
                            write!(fmt, " (vtable: impl <auto trait> for {ty})")?;
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
impl<'tcx, 'a, Prov: Provenance, Extra, Bytes: AllocBytes>
    AllocRefMut<'a, 'tcx, Prov, Extra, Bytes>
{
    /// `range` is relative to this allocation reference, not the base of the allocation.
    pub fn write_scalar(&mut self, range: AllocRange, val: Scalar<Prov>) -> InterpResult<'tcx> {
        let range = self.range.subrange(range);
        debug!("write_scalar at {:?}{range:?}: {val:?}", self.alloc_id);
        Ok(self
            .alloc
            .write_scalar(&self.tcx, range, val)
            .map_err(|e| e.to_interp_error(self.alloc_id))?)
    }

    /// `offset` is relative to this allocation reference, not the base of the allocation.
    pub fn write_ptr_sized(&mut self, offset: Size, val: Scalar<Prov>) -> InterpResult<'tcx> {
        self.write_scalar(alloc_range(offset, self.tcx.data_layout().pointer_size), val)
    }

    /// Mark the entire referenced range as uninitialized
    pub fn write_uninit(&mut self) -> InterpResult<'tcx> {
        Ok(self
            .alloc
            .write_uninit(&self.tcx, self.range)
            .map_err(|e| e.to_interp_error(self.alloc_id))?)
    }
}

impl<'tcx, 'a, Prov: Provenance, Extra, Bytes: AllocBytes> AllocRef<'a, 'tcx, Prov, Extra, Bytes> {
    /// `range` is relative to this allocation reference, not the base of the allocation.
    pub fn read_scalar(
        &self,
        range: AllocRange,
        read_provenance: bool,
    ) -> InterpResult<'tcx, Scalar<Prov>> {
        let range = self.range.subrange(range);
        let res = self
            .alloc
            .read_scalar(&self.tcx, range, read_provenance)
            .map_err(|e| e.to_interp_error(self.alloc_id))?;
        debug!("read_scalar at {:?}{range:?}: {res:?}", self.alloc_id);
        Ok(res)
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
        Ok(self
            .alloc
            .get_bytes_strip_provenance(&self.tcx, self.range)
            .map_err(|e| e.to_interp_error(self.alloc_id))?)
    }

    /// Returns whether the allocation has provenance anywhere in the range of the `AllocRef`.
    pub(crate) fn has_provenance(&self) -> bool {
        !self.alloc.provenance().range_empty(self.range, &self.tcx)
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Reads the given number of bytes from memory, and strips their provenance if possible.
    /// Returns them as a slice.
    ///
    /// Performs appropriate bounds checks.
    pub fn read_bytes_ptr_strip_provenance(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
        size: Size,
    ) -> InterpResult<'tcx, &[u8]> {
        let Some(alloc_ref) = self.get_ptr_alloc(ptr, size, Align::ONE)? else {
            // zero-sized access
            return Ok(&[]);
        };
        // Side-step AllocRef and directly access the underlying bytes more efficiently.
        // (We are staying inside the bounds here so all is good.)
        Ok(alloc_ref
            .alloc
            .get_bytes_strip_provenance(&alloc_ref.tcx, alloc_ref.range)
            .map_err(|e| e.to_interp_error(alloc_ref.alloc_id))?)
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
        let Some(alloc_ref) = self.get_ptr_alloc_mut(ptr, size, Align::ONE)? else {
            // zero-sized access
            assert_matches!(src.next(), None, "iterator said it was empty but returned an element");
            return Ok(());
        };

        // Side-step AllocRef and directly access the underlying bytes more efficiently.
        // (We are staying inside the bounds here so all is good.)
        let alloc_id = alloc_ref.alloc_id;
        let bytes = alloc_ref
            .alloc
            .get_bytes_mut(&alloc_ref.tcx, alloc_ref.range)
            .map_err(move |e| e.to_interp_error(alloc_id))?;
        // `zip` would stop when the first iterator ends; we want to definitely
        // cover all of `bytes`.
        for dest in bytes {
            *dest = src.next().expect("iterator was shorter than it said it would be");
        }
        assert_matches!(src.next(), None, "iterator was longer than it said it would be");
        Ok(())
    }

    pub fn mem_copy(
        &mut self,
        src: Pointer<Option<M::Provenance>>,
        src_align: Align,
        dest: Pointer<Option<M::Provenance>>,
        dest_align: Align,
        size: Size,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        self.mem_copy_repeatedly(src, src_align, dest, dest_align, size, 1, nonoverlapping)
    }

    pub fn mem_copy_repeatedly(
        &mut self,
        src: Pointer<Option<M::Provenance>>,
        src_align: Align,
        dest: Pointer<Option<M::Provenance>>,
        dest_align: Align,
        size: Size,
        num_copies: u64,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        let tcx = self.tcx;
        // We need to do our own bounds-checks.
        let src_parts = self.get_ptr_access(src, size, src_align)?;
        let dest_parts = self.get_ptr_access(dest, size * num_copies, dest_align)?; // `Size` multiplication

        // FIXME: we look up both allocations twice here, once before for the `check_ptr_access`
        // and once below to get the underlying `&[mut] Allocation`.

        // Source alloc preparations and access hooks.
        let Some((src_alloc_id, src_offset, src_prov)) = src_parts else {
            // Zero-sized *source*, that means dest is also zero-sized and we have nothing to do.
            return Ok(());
        };
        let src_alloc = self.get_alloc_raw(src_alloc_id)?;
        let src_range = alloc_range(src_offset, size);
        M::before_memory_read(
            *tcx,
            &self.machine,
            &src_alloc.extra,
            (src_alloc_id, src_prov),
            src_range,
        )?;
        // We need the `dest` ptr for the next operation, so we get it now.
        // We already did the source checks and called the hooks so we are good to return early.
        let Some((dest_alloc_id, dest_offset, dest_prov)) = dest_parts else {
            // Zero-sized *destination*.
            return Ok(());
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
            *tcx,
            extra,
            &mut dest_alloc.extra,
            (dest_alloc_id, dest_prov),
            dest_range,
        )?;
        let dest_bytes = dest_alloc
            .get_bytes_mut_ptr(&tcx, dest_range)
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
            return Ok(());
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
        dest_alloc.init_mask_apply_copy(
            init,
            alloc_range(dest_offset, size), // just a single copy (i.e., not full `dest_range`)
            num_copies,
        );
        // copy the provenance to the destination
        dest_alloc.provenance_apply_copy(provenance);

        Ok(())
    }
}

/// Machine pointer introspection.
impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Test if this value might be null.
    /// If the machine does not support ptr-to-int casts, this is conservative.
    pub fn scalar_may_be_null(&self, scalar: Scalar<M::Provenance>) -> InterpResult<'tcx, bool> {
        Ok(match scalar.try_to_int() {
            Ok(int) => int.is_null(),
            Err(_) => {
                // Can only happen during CTFE.
                let ptr = scalar.to_pointer(self)?;
                match self.ptr_try_get_alloc_id(ptr) {
                    Ok((alloc_id, offset, _)) => {
                        let (size, _align, _kind) = self.get_alloc_info(alloc_id);
                        // If the pointer is out-of-bounds, it may be null.
                        // Note that one-past-the-end (offset == size) is still inbounds, and never null.
                        offset > size
                    }
                    Err(_offset) => bug!("a non-int scalar is always a pointer"),
                }
            }
        })
    }

    /// Turning a "maybe pointer" into a proper pointer (and some information
    /// about where it points), or an absolute address.
    pub fn ptr_try_get_alloc_id(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
    ) -> Result<(AllocId, Size, M::ProvenanceExtra), u64> {
        match ptr.into_pointer_or_addr() {
            Ok(ptr) => match M::ptr_get_alloc(self, ptr) {
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
    #[inline(always)]
    pub fn ptr_get_alloc_id(
        &self,
        ptr: Pointer<Option<M::Provenance>>,
    ) -> InterpResult<'tcx, (AllocId, Size, M::ProvenanceExtra)> {
        self.ptr_try_get_alloc_id(ptr).map_err(|offset| {
            err_ub!(DanglingIntPointer(offset, CheckInAllocMsg::InboundsTest)).into()
        })
    }
}
