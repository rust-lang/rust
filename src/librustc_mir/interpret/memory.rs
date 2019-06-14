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
    InterpResult, Scalar, InterpError, GlobalAlloc, PointerArithmetic,
    Machine, AllocMap, MayLeak, ErrorHandled, CheckInAllocMsg, InboundsCheck,
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
    alloc_map: M::MemoryMap,

    /// To be able to compare pointers with NULL, and to check alignment for accesses
    /// to ZSTs (where pointers may dangle), we keep track of the size even for allocations
    /// that do not exist any more.
    dead_alloc_map: FxHashMap<AllocId, (Size, Align)>,

    /// Extra data added by the machine.
    pub extra: M::MemoryExtra,

    /// Lets us implement `HasDataLayout`, which is awfully convenient.
    pub(super) tcx: TyCtxtAt<'tcx>,
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
            dead_alloc_map: self.dead_alloc_map.clone(),
            extra: (),
            tcx: self.tcx,
        }
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    pub fn new(tcx: TyCtxtAt<'tcx>) -> Self {
        Memory {
            alloc_map: M::MemoryMap::default(),
            dead_alloc_map: FxHashMap::default(),
            extra: M::MemoryExtra::default(),
            tcx,
        }
    }

    #[inline]
    pub fn tag_static_base_pointer(&self, ptr: Pointer) -> Pointer<M::PointerTag> {
        ptr.with_tag(M::tag_static_base_pointer(ptr.alloc_id, &self.extra))
    }

    pub fn create_fn_alloc(&mut self, instance: Instance<'tcx>) -> Pointer<M::PointerTag> {
        let id = self.tcx.alloc_map.lock().create_fn_alloc(instance);
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
        let (alloc, tag) = M::tag_allocation(id, Cow::Owned(alloc), Some(kind), &self.extra);
        self.alloc_map.insert(id, (kind, alloc.into_owned()));
        Pointer::from(id).with_tag(tag)
    }

    pub fn reallocate(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        old_size: Size,
        old_align: Align,
        new_size: Size,
        new_align: Align,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> InterpResult<'tcx, Pointer<M::PointerTag>> {
        if ptr.offset.bytes() != 0 {
            return err!(ReallocateNonBasePtr);
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc".
        // This happens so rarely, the perf advantage is outweighed by the maintenance cost.
        let new_ptr = self.allocate(new_size, new_align, kind);
        self.copy(
            ptr.into(),
            old_align,
            new_ptr.into(),
            new_align,
            old_size.min(new_size),
            /*nonoverlapping*/ true,
        )?;
        self.deallocate(ptr, Some((old_size, old_align)), kind)?;

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
        size_and_align: Option<(Size, Align)>,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> InterpResult<'tcx> {
        trace!("deallocating: {}", ptr.alloc_id);

        if ptr.offset.bytes() != 0 {
            return err!(DeallocateNonBasePtr);
        }

        let (alloc_kind, mut alloc) = match self.alloc_map.remove(&ptr.alloc_id) {
            Some(alloc) => alloc,
            None => {
                // Deallocating static memory -- always an error
                return match self.tcx.alloc_map.lock().get(ptr.alloc_id) {
                    Some(GlobalAlloc::Function(..)) => err!(DeallocatedWrongMemoryKind(
                        "function".to_string(),
                        format!("{:?}", kind),
                    )),
                    Some(GlobalAlloc::Static(..)) |
                    Some(GlobalAlloc::Memory(..)) => err!(DeallocatedWrongMemoryKind(
                        "static".to_string(),
                        format!("{:?}", kind),
                    )),
                    None => err!(DoubleFree)
                }
            }
        };

        if alloc_kind != kind {
            return err!(DeallocatedWrongMemoryKind(
                format!("{:?}", alloc_kind),
                format!("{:?}", kind),
            ));
        }
        if let Some((size, align)) = size_and_align {
            if size.bytes() != alloc.bytes.len() as u64 || align != alloc.align {
                let bytes = Size::from_bytes(alloc.bytes.len() as u64);
                return err!(IncorrectAllocationInformation(size,
                                                           bytes,
                                                           align,
                                                           alloc.align));
            }
        }

        // Let the machine take some extra action
        let size = Size::from_bytes(alloc.bytes.len() as u64);
        AllocationExtra::memory_deallocated(&mut alloc, ptr, size)?;

        // Don't forget to remember size and align of this now-dead allocation
        let old = self.dead_alloc_map.insert(
            ptr.alloc_id,
            (Size::from_bytes(alloc.bytes.len() as u64), alloc.align)
        );
        if old.is_some() {
            bug!("Nothing can be deallocated twice");
        }

        Ok(())
    }

    /// Checks that the pointer is aligned AND non-NULL. This supports ZSTs in two ways:
    /// You can pass a scalar, and a `Pointer` does not have to actually still be allocated.
    pub fn check_align(
        &self,
        ptr: Scalar<M::PointerTag>,
        required_align: Align
    ) -> InterpResult<'tcx> {
        // Check non-NULL/Undef, extract offset
        let (offset, alloc_align) = match ptr.to_bits_or_ptr(self.pointer_size(), self) {
            Err(ptr) => {
                // check this is not NULL -- which we can ensure only if this is in-bounds
                // of some (potentially dead) allocation.
                let align = self.check_bounds_ptr(ptr, InboundsCheck::MaybeDead,
                                                  CheckInAllocMsg::NullPointerTest)?;
                (ptr.offset.bytes(), align)
            }
            Ok(data) => {
                // check this is not NULL
                if data == 0 {
                    return err!(InvalidNullPointerUsage);
                }
                // the "base address" is 0 and hence always aligned
                (data as u64, required_align)
            }
        };
        // Check alignment
        if alloc_align.bytes() < required_align.bytes() {
            return err!(AlignmentCheckFailed {
                has: alloc_align,
                required: required_align,
            });
        }
        if offset % required_align.bytes() == 0 {
            Ok(())
        } else {
            let has = offset % required_align.bytes();
            err!(AlignmentCheckFailed {
                has: Align::from_bytes(has).unwrap(),
                required: required_align,
            })
        }
    }

    /// Checks if the pointer is "in-bounds". Notice that a pointer pointing at the end
    /// of an allocation (i.e., at the first *inaccessible* location) *is* considered
    /// in-bounds!  This follows C's/LLVM's rules.
    /// If you want to check bounds before doing a memory access, better first obtain
    /// an `Allocation` and call `check_bounds`.
    pub fn check_bounds_ptr(
        &self,
        ptr: Pointer<M::PointerTag>,
        liveness: InboundsCheck,
        msg: CheckInAllocMsg,
    ) -> InterpResult<'tcx, Align> {
        let (allocation_size, align) = self.get_size_and_align(ptr.alloc_id, liveness)?;
        ptr.check_in_alloc(allocation_size, msg)?;
        Ok(align)
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
        id: AllocId,
        tcx: TyCtxtAt<'tcx>,
        memory_extra: &M::MemoryExtra,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation<M::PointerTag, M::AllocExtra>>> {
        let alloc = tcx.alloc_map.lock().get(id);
        let alloc = match alloc {
            Some(GlobalAlloc::Memory(mem)) =>
                Cow::Borrowed(mem),
            Some(GlobalAlloc::Function(..)) =>
                return err!(DerefFunctionPointer),
            None =>
                return err!(DanglingPointerDeref),
            Some(GlobalAlloc::Static(def_id)) => {
                // We got a "lazy" static that has not been computed yet.
                if tcx.is_foreign_item(def_id) {
                    trace!("static_alloc: foreign item {:?}", def_id);
                    M::find_foreign_static(def_id, tcx)?
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
                                ErrorHandled::Reported => InterpError::ReferencedConstant,
                                ErrorHandled::TooGeneric => InterpError::TooGeneric,
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
            id, // always use the ID we got as input, not the "hidden" one.
            alloc,
            M::STATIC_KIND.map(MemoryKind::Machine),
            memory_extra
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
            let alloc = Self::get_static_alloc(id, self.tcx, &self.extra).map_err(Err)?;
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
            let alloc = Self::get_static_alloc(id, tcx, memory_extra)?;
            if alloc.mutability == Mutability::Immutable {
                return err!(ModifiedConstantMemory);
            }
            match M::STATIC_KIND {
                Some(kind) => Ok((MemoryKind::Machine(kind), alloc.into_owned())),
                None => err!(ModifiedStatic),
            }
        });
        // Unpack the error type manually because type inference doesn't
        // work otherwise (and we cannot help it because `impl Trait`)
        match a {
            Err(e) => Err(e),
            Ok(a) => {
                let a = &mut a.1;
                if a.mutability == Mutability::Immutable {
                    return err!(ModifiedConstantMemory);
                }
                Ok(a)
            }
        }
    }

    /// Obtain the size and alignment of an allocation, even if that allocation has been deallocated
    ///
    /// If `liveness` is `InboundsCheck::MaybeDead`, this function always returns `Ok`
    pub fn get_size_and_align(
        &self,
        id: AllocId,
        liveness: InboundsCheck,
    ) -> InterpResult<'static, (Size, Align)> {
        if let Ok(alloc) = self.get(id) {
            return Ok((Size::from_bytes(alloc.bytes.len() as u64), alloc.align));
        }
        // Could also be a fn ptr or extern static
        match self.tcx.alloc_map.lock().get(id) {
            Some(GlobalAlloc::Function(..)) => Ok((Size::ZERO, Align::from_bytes(1).unwrap())),
            Some(GlobalAlloc::Static(did)) => {
                // The only way `get` couldn't have worked here is if this is an extern static
                assert!(self.tcx.is_foreign_item(did));
                // Use size and align of the type
                let ty = self.tcx.type_of(did);
                let layout = self.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                Ok((layout.size, layout.align.abi))
            }
            _ => match liveness {
                InboundsCheck::MaybeDead => {
                    // Must be a deallocated pointer
                    Ok(*self.dead_alloc_map.get(&id).expect(
                        "allocation missing in dead_alloc_map"
                    ))
                },
                InboundsCheck::Live => err!(DanglingPointerDeref),
            },
        }
    }

    pub fn get_fn(&self, ptr: Pointer<M::PointerTag>) -> InterpResult<'tcx, Instance<'tcx>> {
        if ptr.offset.bytes() != 0 {
            return err!(InvalidFunctionPointer);
        }
        trace!("reading fn ptr: {}", ptr.alloc_id);
        match self.tcx.alloc_map.lock().get(ptr.alloc_id) {
            Some(GlobalAlloc::Function(instance)) => Ok(instance),
            _ => Err(InterpError::ExecuteMemory.into()),
        }
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

        for i in 0..(alloc.bytes.len() as u64) {
            let i = Size::from_bytes(i);
            if let Some(&(_, target_id)) = alloc.relocations.get(&i) {
                if allocs_seen.insert(target_id) {
                    allocs_to_print.push_back(target_id);
                }
                relocations.push((i, target_id));
            }
            if alloc.undef_mask.is_range_defined(i, i + Size::from_bytes(1)).is_ok() {
                // this `as usize` is fine, since `i` came from a `usize`
                write!(msg, "{:02x} ", alloc.bytes[i.bytes() as usize]).unwrap();
            } else {
                msg.push_str("__ ");
            }
        }

        trace!(
            "{}({} bytes, alignment {}){}",
            msg,
            alloc.bytes.len(),
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

/// Byte Accessors
impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
    pub fn read_bytes(
        &self,
        ptr: Scalar<M::PointerTag>,
        size: Size,
    ) -> InterpResult<'tcx, &[u8]> {
        if size.bytes() == 0 {
            Ok(&[])
        } else {
            let ptr = ptr.to_ptr()?;
            self.get(ptr.alloc_id)?.get_bytes(self, ptr, size)
        }
    }
}

/// Interning (for CTFE)
impl<'mir, 'tcx, M> Memory<'mir, 'tcx, M>
where
    M: Machine<'mir, 'tcx, PointerTag = (), AllocExtra = (), MemoryExtra = ()>,
    // FIXME: Working around https://github.com/rust-lang/rust/issues/24159
    M::MemoryMap: AllocMap<AllocId, (MemoryKind<M::MemoryKinds>, Allocation)>,
{
    /// mark an allocation as static and initialized, either mutable or not
    pub fn intern_static(
        &mut self,
        alloc_id: AllocId,
        mutability: Mutability,
    ) -> InterpResult<'tcx> {
        trace!(
            "mark_static_initialized {:?}, mutability: {:?}",
            alloc_id,
            mutability
        );
        // remove allocation
        let (kind, mut alloc) = self.alloc_map.remove(&alloc_id).unwrap();
        match kind {
            MemoryKind::Machine(_) => bug!("Static cannot refer to machine memory"),
            MemoryKind::Stack | MemoryKind::Vtable => {},
        }
        // ensure llvm knows not to put this into immutable memory
        alloc.mutability = mutability;
        let alloc = self.tcx.intern_const_alloc(alloc);
        self.tcx.alloc_map.lock().set_alloc_id_memory(alloc_id, alloc);
        // recurse into inner allocations
        for &(_, alloc) in alloc.relocations.values() {
            // FIXME: Reusing the mutability here is likely incorrect.  It is originally
            // determined via `is_freeze`, and data is considered frozen if there is no
            // `UnsafeCell` *immediately* in that data -- however, this search stops
            // at references.  So whenever we follow a reference, we should likely
            // assume immutability -- and we should make sure that the compiler
            // does not permit code that would break this!
            if self.alloc_map.contains_key(&alloc) {
                // Not yet interned, so proceed recursively
                self.intern_static(alloc, mutability)?;
            } else if self.dead_alloc_map.contains_key(&alloc) {
                // dangling pointer
                return err!(ValidationFailure(
                    "encountered dangling pointer in final constant".into(),
                ))
            }
        }
        Ok(())
    }
}

/// Reading and writing.
impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'mir, 'tcx, M> {
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
        length: u64,
        nonoverlapping: bool,
    ) -> InterpResult<'tcx> {
        self.check_align(src, src_align)?;
        self.check_align(dest, dest_align)?;
        if size.bytes() == 0 {
            // Nothing to do for ZST, other than checking alignment and
            // non-NULLness which already happened.
            return Ok(());
        }
        let src = src.to_ptr()?;
        let dest = dest.to_ptr()?;

        // first copy the relocations to a temporary buffer, because
        // `get_bytes_mut` will clear the relocations, which is correct,
        // since we don't want to keep any relocations at the target.
        // (`get_bytes_with_undef_and_ptr` below checks that there are no
        // relocations overlapping the edges; those would not be handled correctly).
        let relocations = {
            let relocations = self.get(src.alloc_id)?.relocations(self, src, size);
            if relocations.is_empty() {
                // nothing to copy, ignore even the `length` loop
                Vec::new()
            } else {
                let mut new_relocations = Vec::with_capacity(relocations.len() * (length as usize));
                for i in 0..length {
                    new_relocations.extend(
                        relocations
                        .iter()
                        .map(|&(offset, reloc)| {
                            // compute offset for current repetition
                            let dest_offset = dest.offset + (i * size);
                            (
                                // shift offsets from source allocation to destination allocation
                                offset + dest_offset - src.offset,
                                reloc,
                            )
                        })
                    );
                }

                new_relocations
            }
        };

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
                        return err!(Intrinsic(
                            "copy_nonoverlapping called on overlapping ranges".to_string(),
                        ));
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
        self.get_mut(dest.alloc_id)?.relocations.insert_presorted(relocations);

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

        let undef_mask = &self.get(src.alloc_id)?.undef_mask;

        // Since we are copying `size` bytes from `src` to `dest + i * size` (`for i in 0..repeat`),
        // a naive undef mask copying algorithm would repeatedly have to read the undef mask from
        // the source and write it to the destination. Even if we optimized the memory accesses,
        // we'd be doing all of this `repeat` times.
        // Therefor we precompute a compressed version of the undef mask of the source value and
        // then write it back `repeat` times without computing any more information from the source.

        // a precomputed cache for ranges of defined/undefined bits
        // 0000010010001110 will become
        // [5, 1, 2, 1, 3, 3, 1]
        // where each element toggles the state
        let mut ranges = smallvec::SmallVec::<[u64; 1]>::new();
        let first = undef_mask.get(src.offset);
        let mut cur_len = 1;
        let mut cur = first;
        for i in 1..size.bytes() {
            // FIXME: optimize to bitshift the current undef block's bits and read the top bit
            if undef_mask.get(src.offset + Size::from_bytes(i)) == cur {
                cur_len += 1;
            } else {
                ranges.push(cur_len);
                cur_len = 1;
                cur = !cur;
            }
        }

        // now fill in all the data
        let dest_allocation = self.get_mut(dest.alloc_id)?;
        // an optimization where we can just overwrite an entire range of definedness bits if
        // they are going to be uniformly `1` or `0`.
        if ranges.is_empty() {
            dest_allocation.undef_mask.set_range_inbounds(
                dest.offset,
                dest.offset + size * repeat,
                first,
            );
            return Ok(())
        }

        // remember to fill in the trailing bits
        ranges.push(cur_len);

        for mut j in 0..repeat {
            j *= size.bytes();
            j += dest.offset.bytes();
            let mut cur = first;
            for range in &ranges {
                let old_j = j;
                j += range;
                dest_allocation.undef_mask.set_range_inbounds(
                    Size::from_bytes(old_j),
                    Size::from_bytes(j),
                    cur,
                );
                cur = !cur;
            }
        }
        Ok(())
    }
}
