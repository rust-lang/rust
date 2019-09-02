//! The memory subsystem.
//!
//! Generally, we use `Pointer` to denote memory addresses. However, some operations
//! have a "size"-like parameter, and they take `Scalar` for the address because
//! if the size is 0, then the pointer can also be a (properly aligned, non-NULL)
//! integer. It is crucial that these operations call `check_align` *before*
//! short-circuiting the empty case!

use std::collections::VecDeque;
use std::ptr;
use std::borrow::Cow;

use rustc::ty::{self, Instance, ParamEnv, query::TyCtxtAt};
use rustc::ty::layout::{Align, TargetDataLayout, Size, HasDataLayout};
use rustc_data_structures::fx::{FxHashSet, FxHashMap};

use syntax::ast::Mutability;

use super::{
    Pointer, AllocId, Allocation, GlobalId, AllocationExtra,
    InterpResult, Scalar, GlobalAlloc, PointerArithmetic,
    Machine, AllocMap, MayLeak, ErrorHandled, CheckInAllocMsg,
};

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum MemoryKind<T> {
    /// Error if deallocated except during a stack pop
    Stack,
    /// Error if ever deallocated
    Vtable,
    /// Additional memory kinds a machine wishes to distinguish from the builtin ones
    Machine(T),
}

impl<T: MayLeak> MayLeak for MemoryKind<T> {
    #[inline]
    fn may_leak(self) -> bool {
        match self {
            MemoryKind::Stack => false,
            MemoryKind::Vtable => true,
            MemoryKind::Machine(k) => k.may_leak()
        }
    }
}

/// Used by `get_size_and_align` to indicate whether the allocation needs to be live.
#[derive(Debug, Copy, Clone)]
pub enum AllocCheck {
    /// Allocation must be live and not a function pointer.
    Dereferencable,
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
            FnVal::Instance(instance) =>
                Ok(instance),
            FnVal::Other(_) => throw_unsup_format!(
                "'foreign' function pointers are not supported in this context"
            ),
        }
    }
}

// `Memory` has to depend on the `Machine` because some of its operations
// (e.g., `get`) call a `Machine` hook.
pub struct Memory<'mir, 'tcx, M: Machine<'mir, 'tcx>> {
    /// Allocations local to this instance of the miri engine. The kind
    /// helps ensure that the same mechanism is used for allocation and
    /// deallocation. When an allocation is not found here, it is a
    /// static and looked up in the `tcx` for read access. Some machines may
    /// have to mutate this map even on a read-only access to a static (because
    /// they do pointer provenance tracking and the allocations in `tcx` have
    /// the wrong type), so we let the machine override this type.
    /// Either way, if the machine allows writing to a static, doing so will
    /// create a copy of the static allocation here.
    // FIXME: this should not be public, but interning currently needs access to it
    pub(super) alloc_map: M::MemoryMap,

    /// Map for "extra" function pointers.
    extra_fn_ptr_map: FxHashMap<AllocId, M::ExtraFnVal>,

    /// To be able to compare pointers with NULL, and to check alignment for accesses
    /// to ZSTs (where pointers may dangle), we keep track of the size even for allocations
    /// that do not exist any more.
    // FIXME: this should not be public, but interning currently needs access to it
    pub(super) dead_alloc_map: FxHashMap<AllocId, (Size, Align)>,

    /// Extra data added by the machine.
    pub extra: M::MemoryExtra,

    /// Lets us implement `HasDataLayout`, which is awfully convenient.
    pub tcx: TyCtxtAt<'tcx>,
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> HasDataLayout for Memory<'mir, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

// FIXME: Really we shouldn't clone memory, ever. Snapshot machinery should instead
// carefully copy only the reachable parts.
impl<'mir, 'tcx, M> Clone for Memory<'mir, 'tcx, M>
where
    M: Machine<'mir, 'tcx, PointerTag = (), AllocExtra = (), MemoryExtra = ()>,
    M::MemoryMap: AllocMap<AllocId, (MemoryKind<M::MemoryKinds>, Allocation)>,
{
    fn clone(&self) -> Self {
        Memory {
            alloc_map: self.alloc_map.clone(),
            extra_fn_ptr_map: self.extra_fn_ptr_map.clone(),
            dead_alloc_map: self.dead_alloc_map.clone(),
            extra: (),
            tcx: self.tcx,
        }
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    pub fn new(tcx: TyCtxtAt<'tcx>, extra: M::MemoryExtra) -> Self {
        Memory {
            alloc_map: M::MemoryMap::default(),
            extra_fn_ptr_map: FxHashMap::default(),
            dead_alloc_map: FxHashMap::default(),
            extra,
            tcx,
        }
    }

    #[inline]
    pub fn tag_static_base_pointer(&self, ptr: Pointer) -> Pointer<M::PointerTag> {
        ptr.with_tag(M::tag_static_base_pointer(&self.extra, ptr.alloc_id))
    }

    pub fn create_fn_alloc(
        &mut self,
        fn_val: FnVal<'tcx, M::ExtraFnVal>,
    ) -> Pointer<M::PointerTag>
    {
        let id = match fn_val {
            FnVal::Instance(instance) => self.tcx.alloc_map.lock().create_fn_alloc(instance),
            FnVal::Other(extra) => {
                // FIXME(RalfJung): Should we have a cache here?
                let id = self.tcx.alloc_map.lock().reserve();
                let old = self.extra_fn_ptr_map.insert(id, extra);
                assert!(old.is_none());
                id
            }
        };
        self.tag_static_base_pointer(Pointer::from(id))
    }

    pub fn allocate(
        &mut self,
        size: Size,
        align: Align,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> Pointer<M::PointerTag> {
        let alloc = Allocation::undef(size, align);
        self.allocate_with(alloc, kind)
    }

    pub fn allocate_static_bytes(
        &mut self,
        bytes: &[u8],
        kind: MemoryKind<M::MemoryKinds>,
    ) -> Pointer<M::PointerTag> {
        let alloc = Allocation::from_byte_aligned_bytes(bytes);
        self.allocate_with(alloc, kind)
    }

    pub fn allocate_with(
        &mut self,
        alloc: Allocation,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> Pointer<M::PointerTag> {
        let id = self.tcx.alloc_map.lock().reserve();
        let (alloc, tag) = M::tag_allocation(&self.extra, id, Cow::Owned(alloc), Some(kind));
        self.alloc_map.insert(id, (kind, alloc.into_owned()));
        Pointer::from(id).with_tag(tag)
    }

    pub fn reallocate(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        old_size_and_align: Option<(Size, Align)>,
        new_size: Size,
        new_align: Align,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        if ptr.offset.bytes() != 0 {
            throw_unsup!(ReallocateNonBasePtr)
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc".
        // This happens so rarely, the perf advantage is outweighed by the maintenance cost.
        let new_ptr = self.allocate(new_size, new_align, kind);
        let old_size = match old_size_and_align {
            Some((size, _align)) => size,
            None => self.get(ptr.alloc_id)?.size,
        };
        self.copy(
            ptr,
            new_ptr,
            old_size.min(new_size),
            /*nonoverlapping*/ true,
        )?;
        self.deallocate(ptr, old_size_and_align, kind)?;

        Ok(new_ptr)
    }

    /// Deallocate a local, or do nothing if that local has been made into a static
    pub fn deallocate_local(&mut self, ptr: Pointer<M::PointerTag>) -> InterpResult<'tcx> {
        // The allocation might be already removed by static interning.
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
        kind: MemoryKind<M::MemoryKinds>,
    ) -> InterpResult<'tcx> {
        trace!("deallocating: {}", ptr.alloc_id);

        if ptr.offset.bytes() != 0 {
            throw_unsup!(DeallocateNonBasePtr)
        }

        let (alloc_kind, mut alloc) = match self.alloc_map.remove(&ptr.alloc_id) {
            Some(alloc) => alloc,
            None => {
                // Deallocating static memory -- always an error
                return Err(match self.tcx.alloc_map.lock().get(ptr.alloc_id) {
                    Some(GlobalAlloc::Function(..)) => err_unsup!(DeallocatedWrongMemoryKind(
                        "function".to_string(),
                        format!("{:?}", kind),
                    )),
                    Some(GlobalAlloc::Static(..)) | Some(GlobalAlloc::Memory(..)) => err_unsup!(
                        DeallocatedWrongMemoryKind("static".to_string(), format!("{:?}", kind))
                    ),
                    None => err_unsup!(DoubleFree),
                }
                .into());
            }
        };

        if alloc_kind != kind {
            throw_unsup!(DeallocatedWrongMemoryKind(
                format!("{:?}", alloc_kind),
                format!("{:?}", kind),
            ))
        }
        if let Some((size, align)) = old_size_and_align {
            if size != alloc.size || align != alloc.align {
                let bytes = alloc.size;
                throw_unsup!(IncorrectAllocationInformation(size, bytes, align, alloc.align))
            }
        }

        // Let the machine take some extra action
        let size = alloc.size;
        AllocationExtra::memory_deallocated(&mut alloc, ptr, size)?;

        // Don't forget to remember size and align of this now-dead allocation
        let old = self.dead_alloc_map.insert(
            ptr.alloc_id,
            (alloc.size, alloc.align)
        );
        if old.is_some() {
            bug!("Nothing can be deallocated twice");
        }

        Ok(())
    }

    /// Check if the given scalar is allowed to do a memory access of given `size`
    /// and `align`. On success, returns `None` for zero-sized accesses (where
    /// nothing else is left to do) and a `Pointer` to use for the actual access otherwise.
    /// Crucially, if the input is a `Pointer`, we will test it for liveness
    /// *even if* the size is 0.
    ///
    /// Everyone accessing memory based on a `Scalar` should use this method to get the
    /// `Pointer` they need. And even if you already have a `Pointer`, call this method
    /// to make sure it is sufficiently aligned and not dangling.  Not doing that may
    /// cause ICEs.
    ///
    /// Most of the time you should use `check_mplace_access`, but when you just have a pointer,
    /// this method is still appropriate.
    #[inline(always)]
    pub fn check_ptr_access(
        &self,
        sptr: Scalar<M::PointerTag>,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Option<Pointer<M::PointerTag>>> {
        let align = if M::CHECK_ALIGN { Some(align) } else { None };
        self.check_ptr_access_align(sptr, size, align)
    }

    /// Like `check_ptr_access`, but *definitely* checks alignment when `align`
    /// is `Some` (overriding `M::CHECK_ALIGN`).
    pub(super) fn check_ptr_access_align(
        &self,
        sptr: Scalar<M::PointerTag>,
        size: Size,
        align: Option<Align>,
    ) -> InterpResult<'tcx, Option<Pointer<M::PointerTag>>> {
        fn check_offset_align(offset: u64, align: Align) -> InterpResult<'static> {
            if offset % align.bytes() == 0 {
                Ok(())
            } else {
                // The biggest power of two through which `offset` is divisible.
                let offset_pow2 = 1 << offset.trailing_zeros();
                throw_unsup!(AlignmentCheckFailed {
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
            // A "real" access, we must get a pointer.
            Scalar::Ptr(self.force_ptr(sptr)?)
        };
        Ok(match normalized.to_bits_or_ptr(self.pointer_size(), self) {
            Ok(bits) => {
                let bits = bits as u64; // it's ptr-sized
                assert!(size.bytes() == 0);
                // Must be non-NULL.
                if bits == 0 {
                    throw_unsup!(InvalidNullPointerUsage)
                }
                // Must be aligned.
                if let Some(align) = align {
                    check_offset_align(bits, align)?;
                }
                None
            }
            Err(ptr) => {
                let (allocation_size, alloc_align) =
                    self.get_size_and_align(ptr.alloc_id, AllocCheck::Dereferencable)?;
                // Test bounds. This also ensures non-NULL.
                // It is sufficient to check this for the end pointer. The addition
                // checks for overflow.
                let end_ptr = ptr.offset(size, self)?;
                end_ptr.check_inbounds_alloc(allocation_size, CheckInAllocMsg::MemoryAccessTest)?;
                // Test align. Check this last; if both bounds and alignment are violated
                // we want the error to be about the bounds.
                if let Some(align) = align {
                    if alloc_align.bytes() < align.bytes() {
                        // The allocation itself is not aligned enough.
                        // FIXME: Alignment check is too strict, depending on the base address that
                        // got picked we might be aligned even if this check fails.
                        // We instead have to fall back to converting to an integer and checking
                        // the "real" alignment.
                        throw_unsup!(AlignmentCheckFailed {
                            has: alloc_align,
                            required: align,
                        });
                    }
                    check_offset_align(ptr.offset.bytes(), align)?;
                }

                // We can still be zero-sized in this branch, in which case we have to
                // return `None`.
                if size.bytes() == 0 { None } else { Some(ptr) }
            }
        })
    }

    /// Test if the pointer might be NULL.
    pub fn ptr_may_be_null(
        &self,
        ptr: Pointer<M::PointerTag>,
    ) -> bool {
        let (size, _align) = self.get_size_and_align(ptr.alloc_id, AllocCheck::MaybeDead)
            .expect("alloc info with MaybeDead cannot fail");
        ptr.check_inbounds_alloc(size, CheckInAllocMsg::NullPointerTest).is_err()
    }
}

/// Allocation accessors
impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    /// Helper function to obtain the global (tcx) allocation for a static.
    /// This attempts to return a reference to an existing allocation if
    /// one can be found in `tcx`. That, however, is only possible if `tcx` and
    /// this machine use the same pointer tag, so it is indirected through
    /// `M::tag_allocation`.
    ///
    /// Notice that every static has two `AllocId` that will resolve to the same
    /// thing here: one maps to `GlobalAlloc::Static`, this is the "lazy" ID,
    /// and the other one is maps to `GlobalAlloc::Memory`, this is returned by
    /// `const_eval_raw` and it is the "resolved" ID.
    /// The resolved ID is never used by the interpreted progrma, it is hidden.
    /// The `GlobalAlloc::Memory` branch here is still reachable though; when a static
    /// contains a reference to memory that was created during its evaluation (i.e., not to
    /// another static), those inner references only exist in "resolved" form.
    fn get_static_alloc(
        memory_extra: &M::MemoryExtra,
        tcx: TyCtxtAt<'tcx>,
        id: AllocId,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation<M::PointerTag, M::AllocExtra>>> {
        let alloc = tcx.alloc_map.lock().get(id);
        let alloc = match alloc {
            Some(GlobalAlloc::Memory(mem)) =>
                Cow::Borrowed(mem),
            Some(GlobalAlloc::Function(..)) =>
                throw_unsup!(DerefFunctionPointer),
            None =>
                throw_unsup!(DanglingPointerDeref),
            Some(GlobalAlloc::Static(def_id)) => {
                // We got a "lazy" static that has not been computed yet.
                if tcx.is_foreign_item(def_id) {
                    trace!("static_alloc: foreign item {:?}", def_id);
                    M::find_foreign_static(tcx.tcx, def_id)?
                } else {
                    trace!("static_alloc: Need to compute {:?}", def_id);
                    let instance = Instance::mono(tcx.tcx, def_id);
                    let gid = GlobalId {
                        instance,
                        promoted: None,
                    };
                    // use the raw query here to break validation cycles. Later uses of the static
                    // will call the full query anyway
                    let raw_const = tcx.const_eval_raw(ty::ParamEnv::reveal_all().and(gid))
                        .map_err(|err| {
                            // no need to report anything, the const_eval call takes care of that
                            // for statics
                            assert!(tcx.is_static(def_id));
                            match err {
                                ErrorHandled::Reported =>
                                    err_inval!(ReferencedConstant),
                                ErrorHandled::TooGeneric =>
                                    err_inval!(TooGeneric),
                            }
                        })?;
                    // Make sure we use the ID of the resolved memory, not the lazy one!
                    let id = raw_const.alloc_id;
                    let allocation = tcx.alloc_map.lock().unwrap_memory(id);
                    Cow::Borrowed(allocation)
                }
            }
        };
        // We got tcx memory. Let the machine figure out whether and how to
        // turn that into memory with the right pointer tag.
        Ok(M::tag_allocation(
            memory_extra,
            id, // always use the ID we got as input, not the "hidden" one.
            alloc,
            M::STATIC_KIND.map(MemoryKind::Machine),
        ).0)
    }

    pub fn get(
        &self,
        id: AllocId,
    ) -> InterpResult<'tcx, &Allocation<M::PointerTag, M::AllocExtra>> {
        // The error type of the inner closure here is somewhat funny.  We have two
        // ways of "erroring": An actual error, or because we got a reference from
        // `get_static_alloc` that we can actually use directly without inserting anything anywhere.
        // So the error type is `InterpResult<'tcx, &Allocation<M::PointerTag>>`.
        let a = self.alloc_map.get_or(id, || {
            let alloc = Self::get_static_alloc(&self.extra, self.tcx, id).map_err(Err)?;
            match alloc {
                Cow::Borrowed(alloc) => {
                    // We got a ref, cheaply return that as an "error" so that the
                    // map does not get mutated.
                    Err(Ok(alloc))
                }
                Cow::Owned(alloc) => {
                    // Need to put it into the map and return a ref to that
                    let kind = M::STATIC_KIND.expect(
                        "I got an owned allocation that I have to copy but the machine does \
                            not expect that to happen"
                    );
                    Ok((MemoryKind::Machine(kind), alloc))
                }
            }
        });
        // Now unpack that funny error type
        match a {
            Ok(a) => Ok(&a.1),
            Err(a) => a
        }
    }

    pub fn get_mut(
        &mut self,
        id: AllocId,
    ) -> InterpResult<'tcx, &mut Allocation<M::PointerTag, M::AllocExtra>> {
        let tcx = self.tcx;
        let memory_extra = &self.extra;
        let a = self.alloc_map.get_mut_or(id, || {
            // Need to make a copy, even if `get_static_alloc` is able
            // to give us a cheap reference.
            let alloc = Self::get_static_alloc(memory_extra, tcx, id)?;
            if alloc.mutability == Mutability::Immutable {
                throw_unsup!(ModifiedConstantMemory)
            }
            match M::STATIC_KIND {
                Some(kind) => Ok((MemoryKind::Machine(kind), alloc.into_owned())),
                None => throw_unsup!(ModifiedStatic),
            }
        });
        // Unpack the error type manually because type inference doesn't
        // work otherwise (and we cannot help it because `impl Trait`)
        match a {
            Err(e) => Err(e),
            Ok(a) => {
                let a = &mut a.1;
                if a.mutability == Mutability::Immutable {
                    throw_unsup!(ModifiedConstantMemory)
                }
                Ok(a)
            }
        }
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
        // Don't use `self.get` here as that will
        // a) cause cycles in case `id` refers to a static
        // b) duplicate a static's allocation in miri
        if let Some((_, alloc)) = self.alloc_map.get(id) {
            return Ok((alloc.size, alloc.align));
        }

        // # Function pointers
        // (both global from `alloc_map` and local from `extra_fn_ptr_map`)
        if let Ok(_) = self.get_fn_alloc(id) {
            return if let AllocCheck::Dereferencable = liveness {
                // The caller requested no function pointers.
                throw_unsup!(DerefFunctionPointer)
            } else {
                Ok((Size::ZERO, Align::from_bytes(1).unwrap()))
            };
        }

        // # Statics
        // Can't do this in the match argument, we may get cycle errors since the lock would
        // be held throughout the match.
        let alloc = self.tcx.alloc_map.lock().get(id);
        match alloc {
            Some(GlobalAlloc::Static(did)) => {
                // Use size and align of the type.
                let ty = self.tcx.type_of(did);
                let layout = self.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                Ok((layout.size, layout.align.abi))
            },
            Some(GlobalAlloc::Memory(alloc)) =>
                // Need to duplicate the logic here, because the global allocations have
                // different associated types than the interpreter-local ones.
                Ok((alloc.size, alloc.align)),
            Some(GlobalAlloc::Function(_)) =>
                bug!("We already checked function pointers above"),
            // The rest must be dead.
            None => if let AllocCheck::MaybeDead = liveness {
                // Deallocated pointers are allowed, we should be able to find
                // them in the map.
                Ok(*self.dead_alloc_map.get(&id)
                    .expect("deallocated pointers should all be recorded in \
                            `dead_alloc_map`"))
            } else {
                throw_unsup!(DanglingPointerDeref)
            },
        }
    }

    fn get_fn_alloc(&self, id: AllocId) -> InterpResult<'tcx, FnVal<'tcx, M::ExtraFnVal>> {
        trace!("reading fn ptr: {}", id);
        if let Some(extra) = self.extra_fn_ptr_map.get(&id) {
            Ok(FnVal::Other(*extra))
        } else {
            match self.tcx.alloc_map.lock().get(id) {
                Some(GlobalAlloc::Function(instance)) => Ok(FnVal::Instance(instance)),
                _ => throw_unsup!(ExecuteMemory),
            }
        }
    }

    pub fn get_fn(
        &self,
        ptr: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, FnVal<'tcx, M::ExtraFnVal>> {
        let ptr = self.force_ptr(ptr)?; // We definitely need a pointer value.
        if ptr.offset.bytes() != 0 {
            throw_unsup!(InvalidFunctionPointer)
        }
        self.get_fn_alloc(ptr.alloc_id)
    }

    pub fn mark_immutable(&mut self, id: AllocId) -> InterpResult<'tcx> {
        self.get_mut(id)?.mutability = Mutability::Immutable;
        Ok(())
    }

    /// For debugging, print an allocation and all allocations it points to, recursively.
    pub fn dump_alloc(&self, id: AllocId) {
        self.dump_allocs(vec![id]);
    }

    fn dump_alloc_helper<Tag, Extra>(
        &self,
        allocs_seen: &mut FxHashSet<AllocId>,
        allocs_to_print: &mut VecDeque<AllocId>,
        mut msg: String,
        alloc: &Allocation<Tag, Extra>,
        extra: String,
    ) {
        use std::fmt::Write;

        let prefix_len = msg.len();
        let mut relocations = vec![];

        for i in 0..alloc.size.bytes() {
            let i = Size::from_bytes(i);
            if let Some(&(_, target_id)) = alloc.relocations().get(&i) {
                if allocs_seen.insert(target_id) {
                    allocs_to_print.push_back(target_id);
                }
                relocations.push((i, target_id));
            }
            if alloc.undef_mask().is_range_defined(i, i + Size::from_bytes(1)).is_ok() {
                // this `as usize` is fine, since `i` came from a `usize`
                let i = i.bytes() as usize;

                // Checked definedness (and thus range) and relocations. This access also doesn't
                // influence interpreter execution but is only for debugging.
                let bytes = alloc.inspect_with_undef_and_ptr_outside_interpreter(i..i+1);
                write!(msg, "{:02x} ", bytes[0]).unwrap();
            } else {
                msg.push_str("__ ");
            }
        }

        trace!(
            "{}({} bytes, alignment {}){}",
            msg,
            alloc.size.bytes(),
            alloc.align.bytes(),
            extra
        );

        if !relocations.is_empty() {
            msg.clear();
            write!(msg, "{:1$}", "", prefix_len).unwrap(); // Print spaces.
            let mut pos = Size::ZERO;
            let relocation_width = (self.pointer_size().bytes() - 1) * 3;
            for (i, target_id) in relocations {
                // this `as usize` is fine, since we can't print more chars than `usize::MAX`
                write!(msg, "{:1$}", "", ((i - pos) * 3).bytes() as usize).unwrap();
                let target = format!("({})", target_id);
                // this `as usize` is fine, since we can't print more chars than `usize::MAX`
                write!(msg, "└{0:─^1$}┘ ", target, relocation_width as usize).unwrap();
                pos = i + self.pointer_size();
            }
            trace!("{}", msg);
        }
    }

    /// For debugging, print a list of allocations and all allocations they point to, recursively.
    pub fn dump_allocs(&self, mut allocs: Vec<AllocId>) {
        if !log_enabled!(::log::Level::Trace) {
            return;
        }
        allocs.sort();
        allocs.dedup();
        let mut allocs_to_print = VecDeque::from(allocs);
        let mut allocs_seen = FxHashSet::default();

        while let Some(id) = allocs_to_print.pop_front() {
            let msg = format!("Alloc {:<5} ", format!("{}:", id));

            // normal alloc?
            match self.alloc_map.get_or(id, || Err(())) {
                Ok((kind, alloc)) => {
                    let extra = match kind {
                        MemoryKind::Stack => " (stack)".to_owned(),
                        MemoryKind::Vtable => " (vtable)".to_owned(),
                        MemoryKind::Machine(m) => format!(" ({:?})", m),
                    };
                    self.dump_alloc_helper(
                        &mut allocs_seen, &mut allocs_to_print,
                        msg, alloc, extra
                    );
                },
                Err(()) => {
                    // static alloc?
                    match self.tcx.alloc_map.lock().get(id) {
                        Some(GlobalAlloc::Memory(alloc)) => {
                            self.dump_alloc_helper(
                                &mut allocs_seen, &mut allocs_to_print,
                                msg, alloc, " (immutable)".to_owned()
                            );
                        }
                        Some(GlobalAlloc::Function(func)) => {
                            trace!("{} {}", msg, func);
                        }
                        Some(GlobalAlloc::Static(did)) => {
                            trace!("{} {:?}", msg, did);
                        }
                        None => {
                            trace!("{} (deallocated)", msg);
                        }
                    }
                },
            };

        }
    }

    pub fn leak_report(&self) -> usize {
        trace!("### LEAK REPORT ###");
        let leaks: Vec<_> = self.alloc_map.filter_map_collect(|&id, &(kind, _)| {
            if kind.may_leak() { None } else { Some(id) }
        });
        let n = leaks.len();
        self.dump_allocs(leaks);
        n
    }

    /// This is used by [priroda](https://github.com/oli-obk/priroda)
    pub fn alloc_map(&self) -> &M::MemoryMap {
        &self.alloc_map
    }
}

/// Reading and writing.
impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    /// Reads the given number of bytes from memory. Returns them as a slice.
    ///
    /// Performs appropriate bounds checks.
    pub fn read_bytes(
        &self,
        ptr: Scalar<M::PointerTag>,
        size: Size,
    ) -> InterpResult<'tcx, &[u8]> {
        let ptr = match self.check_ptr_access(ptr, size, Align::from_bytes(1).unwrap())? {
            Some(ptr) => ptr,
            None => return Ok(&[]), // zero-sized access
        };
        self.get(ptr.alloc_id)?.get_bytes(self, ptr, size)
    }

    /// Reads a 0-terminated sequence of bytes from memory. Returns them as a slice.
    ///
    /// Performs appropriate bounds checks.
    pub fn read_c_str(&self, ptr: Scalar<M::PointerTag>) -> InterpResult<'tcx, &[u8]> {
        let ptr = self.force_ptr(ptr)?; // We need to read at least 1 byte, so we *need* a ptr.
        self.get(ptr.alloc_id)?.read_c_str(self, ptr)
    }

    /// Expects the caller to have checked bounds and alignment.
    pub fn copy(
        &mut self,
        src: Pointer<M::PointerTag>,
        dest: Pointer<M::PointerTag>,
        size: Size,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        self.copy_repeatedly(src, dest, size, 1, nonoverlapping)
    }

    /// Expects the caller to have checked bounds and alignment.
    pub fn copy_repeatedly(
        &mut self,
        src: Pointer<M::PointerTag>,
        dest: Pointer<M::PointerTag>,
        size: Size,
        length: u64,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        // first copy the relocations to a temporary buffer, because
        // `get_bytes_mut` will clear the relocations, which is correct,
        // since we don't want to keep any relocations at the target.
        // (`get_bytes_with_undef_and_ptr` below checks that there are no
        // relocations overlapping the edges; those would not be handled correctly).
        let relocations = self.get(src.alloc_id)?
            .prepare_relocation_copy(self, src, size, dest, length);

        let tcx = self.tcx.tcx;

        // This checks relocation edges on the src.
        let src_bytes = self.get(src.alloc_id)?
            .get_bytes_with_undef_and_ptr(&tcx, src, size)?
            .as_ptr();
        let dest_bytes = self.get_mut(dest.alloc_id)?
            .get_bytes_mut(&tcx, dest, size * length)?
            .as_mut_ptr();

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
        // The pointers above remain valid even if the `HashMap` table is moved around because they
        // point into the `Vec` storing the bytes.
        unsafe {
            assert_eq!(size.bytes() as usize as u64, size.bytes());
            if src.alloc_id == dest.alloc_id {
                if nonoverlapping {
                    if (src.offset <= dest.offset && src.offset + size > dest.offset) ||
                        (dest.offset <= src.offset && dest.offset + size > src.offset)
                    {
                        throw_ub_format!(
                            "copy_nonoverlapping called on overlapping ranges"
                        )
                    }
                }

                for i in 0..length {
                    ptr::copy(src_bytes,
                              dest_bytes.offset((size.bytes() * i) as isize),
                              size.bytes() as usize);
                }
            } else {
                for i in 0..length {
                    ptr::copy_nonoverlapping(src_bytes,
                                             dest_bytes.offset((size.bytes() * i) as isize),
                                             size.bytes() as usize);
                }
            }
        }

        // copy definedness to the destination
        self.copy_undef_mask(src, dest, size, length)?;
        // copy the relocations to the destination
        self.get_mut(dest.alloc_id)?.mark_relocation_range(relocations);

        Ok(())
    }
}

/// Undefined bytes
impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    // FIXME: Add a fast version for the common, nonoverlapping case
    fn copy_undef_mask(
        &mut self,
        src: Pointer<M::PointerTag>,
        dest: Pointer<M::PointerTag>,
        size: Size,
        repeat: u64,
    ) -> InterpResult<'tcx> {
        // The bits have to be saved locally before writing to dest in case src and dest overlap.
        assert_eq!(size.bytes() as usize as u64, size.bytes());

        let src_alloc = self.get(src.alloc_id)?;
        let compressed = src_alloc.compress_undef_range(src, size);

        // now fill in all the data
        let dest_allocation = self.get_mut(dest.alloc_id)?;
        dest_allocation.mark_compressed_undef_range(&compressed, dest, size, repeat);

        Ok(())
    }

    pub fn force_ptr(
        &self,
        scalar: Scalar<M::PointerTag>,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        match scalar {
            Scalar::Ptr(ptr) => Ok(ptr),
            _ => M::int_to_ptr(&self, scalar.to_usize(self)?)
        }
    }

    pub fn force_bits(
        &self,
        scalar: Scalar<M::PointerTag>,
        size: Size
    ) -> InterpResult<'tcx, u128> {
        match scalar.to_bits_or_ptr(size, self) {
            Ok(bits) => Ok(bits),
            Err(ptr) => Ok(M::ptr_to_int(&self, ptr)? as u128)
        }
    }
}
