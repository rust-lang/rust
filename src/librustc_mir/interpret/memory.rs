// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The memory subsystem.
//!
//! Generally, we use `Pointer` to denote memory addresses. However, some operations
//! have a "size"-like parameter, and they take `Scalar` for the address because
//! if the size is 0, then the pointer can also be a (properly aligned, non-NULL)
//! integer.  It is crucial that these operations call `check_align` *before*
//! short-circuiting the empty case!

use std::collections::VecDeque;
use std::ptr;
use std::borrow::Cow;

use rustc::ty::{self, Instance, ParamEnv, query::TyCtxtAt};
use rustc::ty::layout::{self, Align, TargetDataLayout, Size, HasDataLayout};
pub use rustc::mir::interpret::{truncate, write_target_uint, read_target_uint};
use rustc_data_structures::fx::{FxHashSet, FxHashMap};

use syntax::ast::Mutability;

use super::{
    Pointer, AllocId, Allocation, GlobalId, AllocationExtra, InboundsCheck,
    EvalResult, Scalar, EvalErrorKind, AllocType, PointerArithmetic,
    Machine, AllocMap, MayLeak, ScalarMaybeUndef, ErrorHandled,
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
// (e.g. `get`) call a `Machine` hook.
pub struct Memory<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'a, 'mir, 'tcx>> {
    /// Allocations local to this instance of the miri engine.  The kind
    /// helps ensure that the same mechanism is used for allocation and
    /// deallocation.  When an allocation is not found here, it is a
    /// static and looked up in the `tcx` for read access.  Some machines may
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

    /// Lets us implement `HasDataLayout`, which is awfully convenient.
    pub(super) tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
}

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> HasDataLayout
    for Memory<'a, 'mir, 'tcx, M>
{
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

// FIXME: Really we shouldn't clone memory, ever. Snapshot machinery should instead
// carefully copy only the reachable parts.
impl<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'a, 'mir, 'tcx>>
    Clone for Memory<'a, 'mir, 'tcx, M>
{
    fn clone(&self) -> Self {
        Memory {
            alloc_map: self.alloc_map.clone(),
            dead_alloc_map: self.dead_alloc_map.clone(),
            tcx: self.tcx,
        }
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    pub fn new(tcx: TyCtxtAt<'a, 'tcx, 'tcx>) -> Self {
        Memory {
            alloc_map: Default::default(),
            dead_alloc_map: FxHashMap::default(),
            tcx,
        }
    }

    pub fn create_fn_alloc(&mut self, instance: Instance<'tcx>) -> Pointer {
        Pointer::from(self.tcx.alloc_map.lock().create_fn_alloc(instance))
    }

    pub fn allocate_static_bytes(&mut self, bytes: &[u8]) -> Pointer {
        Pointer::from(self.tcx.allocate_bytes(bytes))
    }

    pub fn allocate_with(
        &mut self,
        alloc: Allocation<M::PointerTag, M::AllocExtra>,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, AllocId> {
        let id = self.tcx.alloc_map.lock().reserve();
        self.alloc_map.insert(id, (kind, alloc));
        Ok(id)
    }

    pub fn allocate(
        &mut self,
        size: Size,
        align: Align,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, Pointer> {
        Ok(Pointer::from(self.allocate_with(Allocation::undef(size, align), kind)?))
    }

    pub fn reallocate(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        old_size: Size,
        old_align: Align,
        new_size: Size,
        new_align: Align,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, Pointer> {
        if ptr.offset.bytes() != 0 {
            return err!(ReallocateNonBasePtr);
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc".
        // This happens so rarely, the perf advantage is outweighed by the maintenance cost.
        let new_ptr = self.allocate(new_size, new_align, kind)?;
        self.copy(
            ptr.into(),
            old_align,
            new_ptr.with_default_tag().into(),
            new_align,
            old_size.min(new_size),
            /*nonoverlapping*/ true,
        )?;
        self.deallocate(ptr, Some((old_size, old_align)), kind)?;

        Ok(new_ptr)
    }

    /// Deallocate a local, or do nothing if that local has been made into a static
    pub fn deallocate_local(&mut self, ptr: Pointer<M::PointerTag>) -> EvalResult<'tcx> {
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
    ) -> EvalResult<'tcx> {
        trace!("deallocating: {}", ptr.alloc_id);

        if ptr.offset.bytes() != 0 {
            return err!(DeallocateNonBasePtr);
        }

        let (alloc_kind, mut alloc) = match self.alloc_map.remove(&ptr.alloc_id) {
            Some(alloc) => alloc,
            None => {
                // Deallocating static memory -- always an error
                return match self.tcx.alloc_map.lock().get(ptr.alloc_id) {
                    Some(AllocType::Function(..)) => err!(DeallocatedWrongMemoryKind(
                        "function".to_string(),
                        format!("{:?}", kind),
                    )),
                    Some(AllocType::Static(..)) |
                    Some(AllocType::Memory(..)) => err!(DeallocatedWrongMemoryKind(
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

    /// Check that the pointer is aligned AND non-NULL. This supports ZSTs in two ways:
    /// You can pass a scalar, and a `Pointer` does not have to actually still be allocated.
    pub fn check_align(
        &self,
        ptr: Scalar<M::PointerTag>,
        required_align: Align
    ) -> EvalResult<'tcx> {
        // Check non-NULL/Undef, extract offset
        let (offset, alloc_align) = match ptr {
            Scalar::Ptr(ptr) => {
                // check this is not NULL -- which we can ensure only if this is in-bounds
                // of some (potentially dead) allocation.
                self.check_bounds_ptr(ptr, InboundsCheck::MaybeDead)?;
                // data required for alignment check
                let (_, align) = self.get_size_and_align(ptr.alloc_id);
                (ptr.offset.bytes(), align)
            }
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, self.pointer_size().bytes());
                assert!(bits < (1u128 << self.pointer_size().bits()));
                // check this is not NULL
                if bits == 0 {
                    return err!(InvalidNullPointerUsage);
                }
                // the "base address" is 0 and hence always aligned
                (bits as u64, required_align)
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

    /// Check if the pointer is "in-bounds". Notice that a pointer pointing at the end
    /// of an allocation (i.e., at the first *inaccessible* location) *is* considered
    /// in-bounds!  This follows C's/LLVM's rules.  `check` indicates whether we
    /// additionally require the pointer to be pointing to a *live* (still allocated)
    /// allocation.
    /// If you want to check bounds before doing a memory access, better use `check_bounds`.
    pub fn check_bounds_ptr(
        &self,
        ptr: Pointer<M::PointerTag>,
        check: InboundsCheck,
    ) -> EvalResult<'tcx> {
        let allocation_size = match check {
            InboundsCheck::Live => {
                let alloc = self.get(ptr.alloc_id)?;
                alloc.bytes.len() as u64
            }
            InboundsCheck::MaybeDead => {
                self.get_size_and_align(ptr.alloc_id).0.bytes()
            }
        };
        if ptr.offset.bytes() > allocation_size {
            return err!(PointerOutOfBounds {
                ptr: ptr.erase_tag(),
                check,
                allocation_size: Size::from_bytes(allocation_size),
            });
        }
        Ok(())
    }

    /// Check if the memory range beginning at `ptr` and of size `Size` is "in-bounds".
    #[inline(always)]
    pub fn check_bounds(
        &self,
        ptr: Pointer<M::PointerTag>,
        size: Size,
        check: InboundsCheck,
    ) -> EvalResult<'tcx> {
        // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        self.check_bounds_ptr(ptr.offset(size, &*self)?, check)
    }
}

/// Allocation accessors
impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    /// Helper function to obtain the global (tcx) allocation for a static.
    /// This attempts to return a reference to an existing allocation if
    /// one can be found in `tcx`. That, however, is only possible if `tcx` and
    /// this machine use the same pointer tag, so it is indirected through
    /// `M::static_with_default_tag`.
    fn get_static_alloc(
        tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        id: AllocId,
    ) -> EvalResult<'tcx, Cow<'tcx, Allocation<M::PointerTag, M::AllocExtra>>> {
        let alloc = tcx.alloc_map.lock().get(id);
        let def_id = match alloc {
            Some(AllocType::Memory(mem)) => {
                // We got tcx memory. Let the machine figure out whether and how to
                // turn that into memory with the right pointer tag.
                return Ok(M::adjust_static_allocation(mem))
            }
            Some(AllocType::Function(..)) => {
                return err!(DerefFunctionPointer)
            }
            Some(AllocType::Static(did)) => {
                did
            }
            None =>
                return err!(DanglingPointerDeref),
        };
        // We got a "lazy" static that has not been computed yet, do some work
        trace!("static_alloc: Need to compute {:?}", def_id);
        if tcx.is_foreign_item(def_id) {
            return M::find_foreign_static(tcx, def_id);
        }
        let instance = Instance::mono(tcx.tcx, def_id);
        let gid = GlobalId {
            instance,
            promoted: None,
        };
        // use the raw query here to break validation cycles. Later uses of the static will call the
        // full query anyway
        tcx.const_eval_raw(ty::ParamEnv::reveal_all().and(gid)).map_err(|err| {
            // no need to report anything, the const_eval call takes care of that for statics
            assert!(tcx.is_static(def_id).is_some());
            match err {
                ErrorHandled::Reported => EvalErrorKind::ReferencedConstant.into(),
                ErrorHandled::TooGeneric => EvalErrorKind::TooGeneric.into(),
            }
        }).map(|raw_const| {
            let allocation = tcx.alloc_map.lock().unwrap_memory(raw_const.alloc_id);
            // We got tcx memory. Let the machine figure out whether and how to
            // turn that into memory with the right pointer tag.
            M::adjust_static_allocation(allocation)
        })
    }

    pub fn get(&self, id: AllocId) -> EvalResult<'tcx, &Allocation<M::PointerTag, M::AllocExtra>> {
        // The error type of the inner closure here is somewhat funny.  We have two
        // ways of "erroring": An actual error, or because we got a reference from
        // `get_static_alloc` that we can actually use directly without inserting anything anywhere.
        // So the error type is `EvalResult<'tcx, &Allocation<M::PointerTag>>`.
        let a = self.alloc_map.get_or(id, || {
            let alloc = Self::get_static_alloc(self.tcx, id).map_err(Err)?;
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
    ) -> EvalResult<'tcx, &mut Allocation<M::PointerTag, M::AllocExtra>> {
        let tcx = self.tcx;
        let a = self.alloc_map.get_mut_or(id, || {
            // Need to make a copy, even if `get_static_alloc` is able
            // to give us a cheap reference.
            let alloc = Self::get_static_alloc(tcx, id)?;
            if alloc.mutability == Mutability::Immutable {
                return err!(ModifiedConstantMemory);
            }
            let kind = M::STATIC_KIND.expect(
                "An allocation is being mutated but the machine does not expect that to happen"
            );
            Ok((MemoryKind::Machine(kind), alloc.into_owned()))
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

    pub fn get_size_and_align(&self, id: AllocId) -> (Size, Align) {
        if let Ok(alloc) = self.get(id) {
            return (Size::from_bytes(alloc.bytes.len() as u64), alloc.align);
        }
        // Could also be a fn ptr or extern static
        match self.tcx.alloc_map.lock().get(id) {
            Some(AllocType::Function(..)) => (Size::ZERO, Align::from_bytes(1).unwrap()),
            Some(AllocType::Static(did)) => {
                // The only way `get` couldn't have worked here is if this is an extern static
                assert!(self.tcx.is_foreign_item(did));
                // Use size and align of the type
                let ty = self.tcx.type_of(did);
                let layout = self.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();
                (layout.size, layout.align.abi)
            }
            _ => {
                // Must be a deallocated pointer
                *self.dead_alloc_map.get(&id).expect(
                    "allocation missing in dead_alloc_map"
                )
            }
        }
    }

    pub fn get_fn(&self, ptr: Pointer<M::PointerTag>) -> EvalResult<'tcx, Instance<'tcx>> {
        if ptr.offset.bytes() != 0 {
            return err!(InvalidFunctionPointer);
        }
        trace!("reading fn ptr: {}", ptr.alloc_id);
        match self.tcx.alloc_map.lock().get(ptr.alloc_id) {
            Some(AllocType::Function(instance)) => Ok(instance),
            _ => Err(EvalErrorKind::ExecuteMemory.into()),
        }
    }

    pub fn mark_immutable(&mut self, id: AllocId) -> EvalResult<'tcx> {
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
                        Some(AllocType::Memory(alloc)) => {
                            self.dump_alloc_helper(
                                &mut allocs_seen, &mut allocs_to_print,
                                msg, alloc, " (immutable)".to_owned()
                            );
                        }
                        Some(AllocType::Function(func)) => {
                            trace!("{} {}", msg, func);
                        }
                        Some(AllocType::Static(did)) => {
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

/// Interning (for CTFE)
impl<'a, 'mir, 'tcx, M> Memory<'a, 'mir, 'tcx, M>
where
    M: Machine<'a, 'mir, 'tcx, PointerTag=(), AllocExtra=()>,
    M::MemoryMap: AllocMap<AllocId, (MemoryKind<M::MemoryKinds>, Allocation)>,
{
    /// mark an allocation as static and initialized, either mutable or not
    pub fn intern_static(
        &mut self,
        alloc_id: AllocId,
        mutability: Mutability,
    ) -> EvalResult<'tcx> {
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
        self.tcx.alloc_map.lock().set_id_memory(alloc_id, alloc);
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

/// Reading and writing
impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    pub fn copy(
        &mut self,
        src: Scalar<M::PointerTag>,
        src_align: Align,
        dest: Scalar<M::PointerTag>,
        dest_align: Align,
        size: Size,
        nonoverlapping: bool,
    ) -> EvalResult<'tcx> {
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
    ) -> EvalResult<'tcx> {
        if size.bytes() == 0 {
            // Nothing to do for ZST, other than checking alignment and non-NULLness.
            self.check_align(src, src_align)?;
            self.check_align(dest, dest_align)?;
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
            let relocations = self.relocations(src, size)?;
            let mut new_relocations = Vec::with_capacity(relocations.len() * (length as usize));
            for i in 0..length {
                new_relocations.extend(
                    relocations
                    .iter()
                    .map(|&(offset, reloc)| {
                    (offset + dest.offset - src.offset + (i * size * relocations.len() as u64),
                     reloc)
                    })
                );
            }

            new_relocations
        };

        // This also checks alignment, and relocation edges on the src.
        let src_bytes = self.get_bytes_with_undef_and_ptr(src, size, src_align)?.as_ptr();
        let dest_bytes = self.get_bytes_mut(dest, size * length, dest_align)?.as_mut_ptr();

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

    pub fn read_c_str(&self, ptr: Pointer<M::PointerTag>) -> EvalResult<'tcx, &[u8]> {
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        let offset = ptr.offset.bytes() as usize;
        match alloc.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                let p1 = Size::from_bytes((size + 1) as u64);
                self.check_relocations(ptr, p1)?;
                self.check_defined(ptr, p1)?;
                Ok(&alloc.bytes[offset..offset + size])
            }
            None => err!(UnterminatedCString(ptr.erase_tag())),
        }
    }

    pub fn check_bytes(
        &self,
        ptr: Scalar<M::PointerTag>,
        size: Size,
        allow_ptr_and_undef: bool,
    ) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1).unwrap();
        if size.bytes() == 0 {
            self.check_align(ptr, align)?;
            return Ok(());
        }
        let ptr = ptr.to_ptr()?;
        // Check bounds, align and relocations on the edges
        self.get_bytes_with_undef_and_ptr(ptr, size, align)?;
        // Check undef and ptr
        if !allow_ptr_and_undef {
            self.check_defined(ptr, size)?;
            self.check_relocations(ptr, size)?;
        }
        Ok(())
    }

    pub fn read_bytes(&self, ptr: Scalar<M::PointerTag>, size: Size) -> EvalResult<'tcx, &[u8]> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1).unwrap();
        if size.bytes() == 0 {
            self.check_align(ptr, align)?;
            return Ok(&[]);
        }
        self.get_bytes(ptr.to_ptr()?, size, align)
    }

    pub fn write_bytes(&mut self, ptr: Scalar<M::PointerTag>, src: &[u8]) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1).unwrap();
        if src.is_empty() {
            self.check_align(ptr, align)?;
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, Size::from_bytes(src.len() as u64), align)?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    pub fn write_repeat(
        &mut self,
        ptr: Scalar<M::PointerTag>,
        val: u8,
        count: Size
    ) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1).unwrap();
        if count.bytes() == 0 {
            self.check_align(ptr, align)?;
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, count, align)?;
        for b in bytes {
            *b = val;
        }
        Ok(())
    }

    /// Read a *non-ZST* scalar
    pub fn read_scalar(
        &self,
        ptr: Pointer<M::PointerTag>,
        ptr_align: Align,
        size: Size
    ) -> EvalResult<'tcx, ScalarMaybeUndef<M::PointerTag>> {
        // get_bytes_unchecked tests alignment and relocation edges
        let bytes = self.get_bytes_with_undef_and_ptr(
            ptr, size, ptr_align.min(self.int_align(size))
        )?;
        // Undef check happens *after* we established that the alignment is correct.
        // We must not return Ok() for unaligned pointers!
        if self.check_defined(ptr, size).is_err() {
            // this inflates undefined bytes to the entire scalar, even if only a few
            // bytes are undefined
            return Ok(ScalarMaybeUndef::Undef);
        }
        // Now we do the actual reading
        let bits = read_target_uint(self.tcx.data_layout.endian, bytes).unwrap();
        // See if we got a pointer
        if size != self.pointer_size() {
            // *Now* better make sure that the inside also is free of relocations.
            self.check_relocations(ptr, size)?;
        } else {
            let alloc = self.get(ptr.alloc_id)?;
            match alloc.relocations.get(&ptr.offset) {
                Some(&(tag, alloc_id)) => {
                    let ptr = Pointer::new_with_tag(alloc_id, Size::from_bytes(bits as u64), tag);
                    return Ok(ScalarMaybeUndef::Scalar(ptr.into()))
                }
                None => {},
            }
        }
        // We don't. Just return the bits.
        Ok(ScalarMaybeUndef::Scalar(Scalar::from_uint(bits, size)))
    }

    pub fn read_ptr_sized(
        &self,
        ptr: Pointer<M::PointerTag>,
        ptr_align: Align
    ) -> EvalResult<'tcx, ScalarMaybeUndef<M::PointerTag>> {
        self.read_scalar(ptr, ptr_align, self.pointer_size())
    }

    /// Write a *non-ZST* scalar
    pub fn write_scalar(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        ptr_align: Align,
        val: ScalarMaybeUndef<M::PointerTag>,
        type_size: Size,
    ) -> EvalResult<'tcx> {
        let val = match val {
            ScalarMaybeUndef::Scalar(scalar) => scalar,
            ScalarMaybeUndef::Undef => return self.mark_definedness(ptr, type_size, false),
        };

        let bytes = match val {
            Scalar::Ptr(val) => {
                assert_eq!(type_size, self.pointer_size());
                val.offset.bytes() as u128
            }

            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, type_size.bytes());
                debug_assert_eq!(truncate(bits, Size::from_bytes(size.into())), bits,
                    "Unexpected value of size {} when writing to memory", size);
                bits
            },
        };

        {
            // get_bytes_mut checks alignment
            let endian = self.tcx.data_layout.endian;
            let dst = self.get_bytes_mut(ptr, type_size, ptr_align)?;
            write_target_uint(endian, dst, bytes).unwrap();
        }

        // See if we have to also write a relocation
        match val {
            Scalar::Ptr(val) => {
                self.get_mut(ptr.alloc_id)?.relocations.insert(
                    ptr.offset,
                    (val.tag, val.alloc_id),
                );
            }
            _ => {}
        }

        Ok(())
    }

    pub fn write_ptr_sized(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        ptr_align: Align,
        val: ScalarMaybeUndef<M::PointerTag>
    ) -> EvalResult<'tcx> {
        let ptr_size = self.pointer_size();
        self.write_scalar(ptr.into(), ptr_align, val, ptr_size)
    }

    fn int_align(&self, size: Size) -> Align {
        // We assume pointer-sized integers have the same alignment as pointers.
        // We also assume signed and unsigned integers of the same size have the same alignment.
        let ity = match size.bytes() {
            1 => layout::I8,
            2 => layout::I16,
            4 => layout::I32,
            8 => layout::I64,
            16 => layout::I128,
            _ => bug!("bad integer size: {}", size.bytes()),
        };
        ity.align(self).abi
    }
}

/// Relocations
impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    /// Return all relocations overlapping with the given ptr-offset pair.
    fn relocations(
        &self,
        ptr: Pointer<M::PointerTag>,
        size: Size,
    ) -> EvalResult<'tcx, &[(Size, (M::PointerTag, AllocId))]> {
        // We have to go back `pointer_size - 1` bytes, as that one would still overlap with
        // the beginning of this range.
        let start = ptr.offset.bytes().saturating_sub(self.pointer_size().bytes() - 1);
        let end = ptr.offset + size; // this does overflow checking
        Ok(self.get(ptr.alloc_id)?.relocations.range(Size::from_bytes(start)..end))
    }

    /// Check that there ar eno relocations overlapping with the given range.
    #[inline(always)]
    fn check_relocations(&self, ptr: Pointer<M::PointerTag>, size: Size) -> EvalResult<'tcx> {
        if self.relocations(ptr, size)?.len() != 0 {
            err!(ReadPointerAsBytes)
        } else {
            Ok(())
        }
    }

    /// Remove all relocations inside the given range.
    /// If there are relocations overlapping with the edges, they
    /// are removed as well *and* the bytes they cover are marked as
    /// uninitialized.  This is a somewhat odd "spooky action at a distance",
    /// but it allows strictly more code to run than if we would just error
    /// immediately in that case.
    fn clear_relocations(&mut self, ptr: Pointer<M::PointerTag>, size: Size) -> EvalResult<'tcx> {
        // Find the start and end of the given range and its outermost relocations.
        let (first, last) = {
            // Find all relocations overlapping the given range.
            let relocations = self.relocations(ptr, size)?;
            if relocations.is_empty() {
                return Ok(());
            }

            (relocations.first().unwrap().0,
             relocations.last().unwrap().0 + self.pointer_size())
        };
        let start = ptr.offset;
        let end = start + size;

        let alloc = self.get_mut(ptr.alloc_id)?;

        // Mark parts of the outermost relocations as undefined if they partially fall outside the
        // given range.
        if first < start {
            alloc.undef_mask.set_range(first, start, false);
        }
        if last > end {
            alloc.undef_mask.set_range(end, last, false);
        }

        // Forget all the relocations.
        alloc.relocations.remove_range(first..last);

        Ok(())
    }

    /// Error if there are relocations overlapping with the edges of the
    /// given memory range.
    #[inline]
    fn check_relocation_edges(&self, ptr: Pointer<M::PointerTag>, size: Size) -> EvalResult<'tcx> {
        self.check_relocations(ptr, Size::ZERO)?;
        self.check_relocations(ptr.offset(size, self)?, Size::ZERO)?;
        Ok(())
    }
}

/// Undefined bytes
impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    // FIXME: Add a fast version for the common, nonoverlapping case
    fn copy_undef_mask(
        &mut self,
        src: Pointer<M::PointerTag>,
        dest: Pointer<M::PointerTag>,
        size: Size,
        repeat: u64,
    ) -> EvalResult<'tcx> {
        // The bits have to be saved locally before writing to dest in case src and dest overlap.
        assert_eq!(size.bytes() as usize as u64, size.bytes());

        let undef_mask = self.get(src.alloc_id)?.undef_mask.clone();
        let dest_allocation = self.get_mut(dest.alloc_id)?;

        for i in 0..size.bytes() {
            let defined = undef_mask.get(src.offset + Size::from_bytes(i));

            for j in 0..repeat {
                dest_allocation.undef_mask.set(
                    dest.offset + Size::from_bytes(i + (size.bytes() * j)),
                    defined
                );
            }
        }

        Ok(())
    }

    /// Checks that a range of bytes is defined. If not, returns the `ReadUndefBytes`
    /// error which will report the first byte which is undefined.
    #[inline]
    fn check_defined(&self, ptr: Pointer<M::PointerTag>, size: Size) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        alloc.undef_mask.is_range_defined(
            ptr.offset,
            ptr.offset + size,
        ).or_else(|idx| err!(ReadUndefBytes(idx)))
    }

    pub fn mark_definedness(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        size: Size,
        new_state: bool,
    ) -> EvalResult<'tcx> {
        if size.bytes() == 0 {
            return Ok(());
        }
        let alloc = self.get_mut(ptr.alloc_id)?;
        alloc.undef_mask.set_range(
            ptr.offset,
            ptr.offset + size,
            new_state,
        );
        Ok(())
    }
}
