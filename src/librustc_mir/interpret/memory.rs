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
use std::hash::{Hash, Hasher};
use std::ptr;

use rustc::hir::def_id::DefId;
use rustc::ty::Instance;
use rustc::ty::ParamEnv;
use rustc::ty::query::TyCtxtAt;
use rustc::ty::layout::{self, Align, TargetDataLayout, Size};
use rustc::mir::interpret::{Pointer, AllocId, Allocation, AccessKind, ScalarMaybeUndef,
                            EvalResult, Scalar, EvalErrorKind, GlobalId, AllocType, truncate};
pub use rustc::mir::interpret::{write_target_uint, read_target_uint};
use rustc_data_structures::fx::{FxHashSet, FxHashMap, FxHasher};

use syntax::ast::Mutability;

use super::{EvalContext, Machine};


////////////////////////////////////////////////////////////////////////////////
// Allocations and pointers
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MemoryKind<T> {
    /// Error if deallocated except during a stack pop
    Stack,
    /// Additional memory kinds a machine wishes to distinguish from the builtin ones
    Machine(T),
}

////////////////////////////////////////////////////////////////////////////////
// Top-level interpreter memory
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct Memory<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    /// Additional data required by the Machine
    pub data: M::MemoryData,

    /// Helps guarantee that stack allocations aren't deallocated via `rust_deallocate`
    alloc_kind: FxHashMap<AllocId, MemoryKind<M::MemoryKinds>>,

    /// Actual memory allocations (arbitrary bytes, may contain pointers into other allocations).
    alloc_map: FxHashMap<AllocId, Allocation>,

    pub tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
}

impl<'a, 'mir, 'tcx, M> Eq for Memory<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
          'tcx: 'a + 'mir,
{}

impl<'a, 'mir, 'tcx, M> PartialEq for Memory<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
          'tcx: 'a + 'mir,
{
    fn eq(&self, other: &Self) -> bool {
        let Memory {
            data,
            alloc_kind,
            alloc_map,
            tcx: _,
        } = self;

        *data == other.data
            && *alloc_kind == other.alloc_kind
            && *alloc_map == other.alloc_map
    }
}

impl<'a, 'mir, 'tcx, M> Hash for Memory<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>,
          'tcx: 'a + 'mir,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Memory {
            data,
            alloc_kind: _,
            alloc_map: _,
            tcx: _,
        } = self;

        data.hash(state);

        // We ignore some fields which don't change between evaluation steps.

        // Since HashMaps which contain the same items may have different
        // iteration orders, we use a commutative operation (in this case
        // addition, but XOR would also work), to combine the hash of each
        // `Allocation`.
        self.allocations()
            .map(|allocs| {
                let mut h = FxHasher::default();
                allocs.hash(&mut h);
                h.finish()
            })
            .fold(0u64, |hash, x| hash.wrapping_add(x))
            .hash(state);
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    pub fn new(tcx: TyCtxtAt<'a, 'tcx, 'tcx>, data: M::MemoryData) -> Self {
        Memory {
            data,
            alloc_kind: FxHashMap::default(),
            alloc_map: FxHashMap::default(),
            tcx,
        }
    }

    pub fn allocations<'x>(
        &'x self,
    ) -> impl Iterator<Item = (AllocId, &'x Allocation)> {
        self.alloc_map.iter().map(|(&id, alloc)| (id, alloc))
    }

    pub fn create_fn_alloc(&mut self, instance: Instance<'tcx>) -> Pointer {
        self.tcx.alloc_map.lock().create_fn_alloc(instance).into()
    }

    pub fn allocate_bytes(&mut self, bytes: &[u8]) -> Pointer {
        self.tcx.allocate_bytes(bytes).into()
    }

    /// kind is `None` for statics
    pub fn allocate_value(
        &mut self,
        alloc: Allocation,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, AllocId> {
        let id = self.tcx.alloc_map.lock().reserve();
        M::add_lock(self, id);
        self.alloc_map.insert(id, alloc);
        self.alloc_kind.insert(id, kind);
        Ok(id)
    }

    /// kind is `None` for statics
    pub fn allocate(
        &mut self,
        size: Size,
        align: Align,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, Pointer> {
        self.allocate_value(Allocation::undef(size, align), kind).map(Pointer::from)
    }

    pub fn reallocate(
        &mut self,
        ptr: Pointer,
        old_size: Size,
        old_align: Align,
        new_size: Size,
        new_align: Align,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, Pointer> {
        if ptr.offset.bytes() != 0 {
            return err!(ReallocateNonBasePtr);
        }
        if self.alloc_map.contains_key(&ptr.alloc_id) {
            let alloc_kind = self.alloc_kind[&ptr.alloc_id];
            if alloc_kind != kind {
                return err!(ReallocatedWrongMemoryKind(
                    format!("{:?}", alloc_kind),
                    format!("{:?}", kind),
                ));
            }
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc"
        let new_ptr = self.allocate(new_size, new_align, kind)?;
        self.copy(
            ptr.into(),
            old_align,
            new_ptr.into(),
            new_align,
            old_size.min(new_size),
            /*nonoverlapping*/
            true,
        )?;
        self.deallocate(ptr, Some((old_size, old_align)), kind)?;

        Ok(new_ptr)
    }

    pub fn deallocate_local(&mut self, ptr: Pointer) -> EvalResult<'tcx> {
        match self.alloc_kind.get(&ptr.alloc_id).cloned() {
            Some(MemoryKind::Stack) => self.deallocate(ptr, None, MemoryKind::Stack),
            // Happens if the memory was interned into immutable memory
            None => Ok(()),
            other => bug!("local contained non-stack memory: {:?}", other),
        }
    }

    pub fn deallocate(
        &mut self,
        ptr: Pointer,
        size_and_align: Option<(Size, Align)>,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx> {
        if ptr.offset.bytes() != 0 {
            return err!(DeallocateNonBasePtr);
        }

        let alloc = match self.alloc_map.remove(&ptr.alloc_id) {
            Some(alloc) => alloc,
            None => {
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

        let alloc_kind = self.alloc_kind
                        .remove(&ptr.alloc_id)
                        .expect("alloc_map out of sync with alloc_kind");

        // It is okay for us to still holds locks on deallocation -- for example, we could store
        // data we own in a local, and the local could be deallocated (from StorageDead) before the
        // function returns. However, we should check *something*.  For now, we make sure that there
        // is no conflicting write lock by another frame.  We *have* to permit deallocation if we
        // hold a read lock.
        // FIXME: Figure out the exact rules here.
        M::free_lock(self, ptr.alloc_id, alloc.bytes.len() as u64)?;

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

        debug!("deallocated : {}", ptr.alloc_id);

        Ok(())
    }

    pub fn pointer_size(&self) -> Size {
        self.tcx.data_layout.pointer_size
    }

    pub fn endianness(&self) -> layout::Endian {
        self.tcx.data_layout.endian
    }

    /// Check that the pointer is aligned AND non-NULL. This supports scalars
    /// for the benefit of other parts of miri that need to check alignment even for ZST.
    pub fn check_align(&self, ptr: Scalar, required_align: Align) -> EvalResult<'tcx> {
        // Check non-NULL/Undef, extract offset
        let (offset, alloc_align) = match ptr {
            Scalar::Ptr(ptr) => {
                let alloc = self.get(ptr.alloc_id)?;
                (ptr.offset.bytes(), alloc.align)
            }
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, self.pointer_size().bytes());
                // FIXME: what on earth does this line do? docs or fix needed!
                let v = ((bits as u128) % (1 << self.pointer_size().bytes())) as u64;
                if v == 0 {
                    return err!(InvalidNullPointerUsage);
                }
                // the base address if the "integer allocation" is 0 and hence always aligned
                (v, required_align)
            }
        };
        // Check alignment
        if alloc_align.abi() < required_align.abi() {
            return err!(AlignmentCheckFailed {
                has: alloc_align,
                required: required_align,
            });
        }
        if offset % required_align.abi() == 0 {
            Ok(())
        } else {
            let has = offset % required_align.abi();
            err!(AlignmentCheckFailed {
                has: Align::from_bytes(has, has).unwrap(),
                required: required_align,
            })
        }
    }

    /// Check if the pointer is "in-bounds". Notice that a pointer pointing at the end
    /// of an allocation (i.e., at the first *inaccessible* location) *is* considered
    /// in-bounds!  This follows C's/LLVM's rules.
    pub fn check_bounds(&self, ptr: Pointer, access: bool) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        let allocation_size = alloc.bytes.len() as u64;
        if ptr.offset.bytes() > allocation_size {
            return err!(PointerOutOfBounds {
                ptr,
                access,
                allocation_size: Size::from_bytes(allocation_size),
            });
        }
        Ok(())
    }
}

/// Allocation accessors
impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    fn const_eval_static(&self, def_id: DefId) -> EvalResult<'tcx, &'tcx Allocation> {
        if self.tcx.is_foreign_item(def_id) {
            return err!(ReadForeignStatic);
        }
        let instance = Instance::mono(self.tcx.tcx, def_id);
        let gid = GlobalId {
            instance,
            promoted: None,
        };
        self.tcx.const_eval(ParamEnv::reveal_all().and(gid)).map_err(|err| {
            // no need to report anything, the const_eval call takes care of that for statics
            assert!(self.tcx.is_static(def_id).is_some());
            EvalErrorKind::ReferencedConstant(err).into()
        }).map(|val| {
            self.tcx.const_to_allocation(val)
        })
    }

    pub fn get(&self, id: AllocId) -> EvalResult<'tcx, &Allocation> {
        // normal alloc?
        match self.alloc_map.get(&id) {
            Some(alloc) => Ok(alloc),
            // uninitialized static alloc?
            None => {
                // static alloc?
                let alloc = self.tcx.alloc_map.lock().get(id);
                match alloc {
                    Some(AllocType::Memory(mem)) => Ok(mem),
                    Some(AllocType::Function(..)) => {
                        Err(EvalErrorKind::DerefFunctionPointer.into())
                    }
                    Some(AllocType::Static(did)) => {
                        self.const_eval_static(did)
                    }
                    None => Err(EvalErrorKind::DanglingPointerDeref.into()),
                }
            },
        }
    }

    fn get_mut(
        &mut self,
        id: AllocId,
    ) -> EvalResult<'tcx, &mut Allocation> {
        // normal alloc?
        match self.alloc_map.get_mut(&id) {
            Some(alloc) => Ok(alloc),
            // uninitialized static alloc?
            None => {
                // no alloc or immutable alloc? produce an error
                match self.tcx.alloc_map.lock().get(id) {
                    Some(AllocType::Memory(..)) |
                    Some(AllocType::Static(..)) => err!(ModifiedConstantMemory),
                    Some(AllocType::Function(..)) => err!(DerefFunctionPointer),
                    None => err!(DanglingPointerDeref),
                }
            },
        }
    }

    pub fn get_fn(&self, ptr: Pointer) -> EvalResult<'tcx, Instance<'tcx>> {
        if ptr.offset.bytes() != 0 {
            return err!(InvalidFunctionPointer);
        }
        debug!("reading fn ptr: {}", ptr.alloc_id);
        match self.tcx.alloc_map.lock().get(ptr.alloc_id) {
            Some(AllocType::Function(instance)) => Ok(instance),
            _ => Err(EvalErrorKind::ExecuteMemory.into()),
        }
    }

    pub fn get_alloc_kind(&self, id: AllocId) -> Option<MemoryKind<M::MemoryKinds>> {
        self.alloc_kind.get(&id).cloned()
    }

    /// For debugging, print an allocation and all allocations it points to, recursively.
    pub fn dump_alloc(&self, id: AllocId) {
        if !log_enabled!(::log::Level::Trace) {
            return;
        }
        self.dump_allocs(vec![id]);
    }

    /// For debugging, print a list of allocations and all allocations they point to, recursively.
    pub fn dump_allocs(&self, mut allocs: Vec<AllocId>) {
        if !log_enabled!(::log::Level::Trace) {
            return;
        }
        use std::fmt::Write;
        allocs.sort();
        allocs.dedup();
        let mut allocs_to_print = VecDeque::from(allocs);
        let mut allocs_seen = FxHashSet::default();

        while let Some(id) = allocs_to_print.pop_front() {
            let mut msg = format!("Alloc {:<5} ", format!("{}:", id));
            let prefix_len = msg.len();
            let mut relocations = vec![];

            let (alloc, immutable) =
                // normal alloc?
                match self.alloc_map.get(&id) {
                    Some(a) => (a, match self.alloc_kind[&id] {
                        MemoryKind::Stack => " (stack)".to_owned(),
                        MemoryKind::Machine(m) => format!(" ({:?})", m),
                    }),
                    None => {
                        // static alloc?
                        match self.tcx.alloc_map.lock().get(id) {
                            Some(AllocType::Memory(a)) => (a, "(immutable)".to_owned()),
                            Some(AllocType::Function(func)) => {
                                trace!("{} {}", msg, func);
                                continue;
                            }
                            Some(AllocType::Static(did)) => {
                                trace!("{} {:?}", msg, did);
                                continue;
                            }
                            None => {
                                trace!("{} (deallocated)", msg);
                                continue;
                            }
                        }
                    },
                };

            for i in 0..(alloc.bytes.len() as u64) {
                let i = Size::from_bytes(i);
                if let Some(&target_id) = alloc.relocations.get(&i) {
                    if allocs_seen.insert(target_id) {
                        allocs_to_print.push_back(target_id);
                    }
                    relocations.push((i, target_id));
                }
                if alloc.undef_mask.is_range_defined(i, i + Size::from_bytes(1)) {
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
                alloc.align.abi(),
                immutable
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
    }

    pub fn leak_report(&self) -> usize {
        trace!("### LEAK REPORT ###");
        let leaks: Vec<_> = self.alloc_map
            .keys()
            .cloned()
            .collect();
        let n = leaks.len();
        self.dump_allocs(leaks);
        n
    }
}

/// Byte accessors
impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    /// This checks alignment!
    fn get_bytes_unchecked(
        &self,
        ptr: Pointer,
        size: Size,
        align: Align,
    ) -> EvalResult<'tcx, &[u8]> {
        // Zero-sized accesses can use dangling pointers,
        // but they still have to be aligned and non-NULL
        self.check_align(ptr.into(), align)?;
        if size.bytes() == 0 {
            return Ok(&[]);
        }
        M::check_locks(self, ptr, size, AccessKind::Read)?;
        // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        self.check_bounds(ptr.offset(size, self)?, true)?;
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&alloc.bytes[offset..offset + size.bytes() as usize])
    }

    /// This checks alignment!
    fn get_bytes_unchecked_mut(
        &mut self,
        ptr: Pointer,
        size: Size,
        align: Align,
    ) -> EvalResult<'tcx, &mut [u8]> {
        // Zero-sized accesses can use dangling pointers,
        // but they still have to be aligned and non-NULL
        self.check_align(ptr.into(), align)?;
        if size.bytes() == 0 {
            return Ok(&mut []);
        }
        M::check_locks(self, ptr, size, AccessKind::Write)?;
        // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        self.check_bounds(ptr.offset(size, &*self)?, true)?;
        let alloc = self.get_mut(ptr.alloc_id)?;
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&mut alloc.bytes[offset..offset + size.bytes() as usize])
    }

    fn get_bytes(&self, ptr: Pointer, size: Size, align: Align) -> EvalResult<'tcx, &[u8]> {
        assert_ne!(size.bytes(), 0);
        if self.relocations(ptr, size)?.len() != 0 {
            return err!(ReadPointerAsBytes);
        }
        self.check_defined(ptr, size)?;
        self.get_bytes_unchecked(ptr, size, align)
    }

    fn get_bytes_mut(
        &mut self,
        ptr: Pointer,
        size: Size,
        align: Align,
    ) -> EvalResult<'tcx, &mut [u8]> {
        assert_ne!(size.bytes(), 0);
        self.clear_relocations(ptr, size)?;
        self.mark_definedness(ptr, size, true)?;
        self.get_bytes_unchecked_mut(ptr, size, align)
    }
}

/// Reading and writing
impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    /// mark an allocation pointed to by a static as static and initialized
    fn mark_inner_allocation_initialized(
        &mut self,
        alloc: AllocId,
        mutability: Mutability,
    ) -> EvalResult<'tcx> {
        match self.alloc_kind.get(&alloc) {
            // do not go into statics
            None => Ok(()),
            // just locals and machine allocs
            Some(_) => self.mark_static_initialized(alloc, mutability),
        }
    }

    /// mark an allocation as static and initialized, either mutable or not
    pub fn mark_static_initialized(
        &mut self,
        alloc_id: AllocId,
        mutability: Mutability,
    ) -> EvalResult<'tcx> {
        trace!(
            "mark_static_initialized {:?}, mutability: {:?}",
            alloc_id,
            mutability
        );
        // The machine handled it
        if M::mark_static_initialized(self, alloc_id, mutability)? {
            return Ok(())
        }
        let alloc = self.alloc_map.remove(&alloc_id);
        match self.alloc_kind.remove(&alloc_id) {
            None => {},
            Some(MemoryKind::Machine(_)) => bug!("machine didn't handle machine alloc"),
            Some(MemoryKind::Stack) => {},
        }
        if let Some(mut alloc) = alloc {
            // ensure llvm knows not to put this into immutable memory
            alloc.runtime_mutability = mutability;
            let alloc = self.tcx.intern_const_alloc(alloc);
            self.tcx.alloc_map.lock().set_id_memory(alloc_id, alloc);
            // recurse into inner allocations
            for &alloc in alloc.relocations.values() {
                self.mark_inner_allocation_initialized(alloc, mutability)?;
            }
        } else {
            bug!("no allocation found for {:?}", alloc_id);
        }
        Ok(())
    }

    pub fn copy(
        &mut self,
        src: Scalar,
        src_align: Align,
        dest: Scalar,
        dest_align: Align,
        size: Size,
        nonoverlapping: bool,
    ) -> EvalResult<'tcx> {
        self.copy_repeatedly(src, src_align, dest, dest_align, size, 1, nonoverlapping)
    }

    pub fn copy_repeatedly(
        &mut self,
        src: Scalar,
        src_align: Align,
        dest: Scalar,
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
        self.check_relocation_edges(src, size)?;

        // first copy the relocations to a temporary buffer, because
        // `get_bytes_mut` will clear the relocations, which is correct,
        // since we don't want to keep any relocations at the target.
        let relocations = {
            let relocations = self.relocations(src, size)?;
            let mut new_relocations = Vec::with_capacity(relocations.len() * (length as usize));
            for i in 0..length {
                new_relocations.extend(
                    relocations
                    .iter()
                    .map(|&(offset, alloc_id)| {
                    (offset + dest.offset - src.offset + (i * size * relocations.len() as u64),
                    alloc_id)
                    })
                );
            }

            new_relocations
        };

        // This also checks alignment.
        let src_bytes = self.get_bytes_unchecked(src, size, src_align)?.as_ptr();
        let dest_bytes = self.get_bytes_mut(dest, size * length, dest_align)?.as_mut_ptr();

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
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

        self.copy_undef_mask(src, dest, size, length)?;
        // copy back the relocations
        self.get_mut(dest.alloc_id)?.relocations.insert_presorted(relocations);

        Ok(())
    }

    pub fn read_c_str(&self, ptr: Pointer) -> EvalResult<'tcx, &[u8]> {
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        let offset = ptr.offset.bytes() as usize;
        match alloc.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                let p1 = Size::from_bytes((size + 1) as u64);
                if self.relocations(ptr, p1)?.len() != 0 {
                    return err!(ReadPointerAsBytes);
                }
                self.check_defined(ptr, p1)?;
                M::check_locks(self, ptr, p1, AccessKind::Read)?;
                Ok(&alloc.bytes[offset..offset + size])
            }
            None => err!(UnterminatedCString(ptr)),
        }
    }

    pub fn read_bytes(&self, ptr: Scalar, size: Size) -> EvalResult<'tcx, &[u8]> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        if size.bytes() == 0 {
            self.check_align(ptr, align)?;
            return Ok(&[]);
        }
        self.get_bytes(ptr.to_ptr()?, size, align)
    }

    pub fn write_bytes(&mut self, ptr: Scalar, src: &[u8]) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        if src.is_empty() {
            self.check_align(ptr, align)?;
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, Size::from_bytes(src.len() as u64), align)?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    pub fn write_repeat(&mut self, ptr: Scalar, val: u8, count: Size) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
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
        ptr: Pointer,
        ptr_align: Align,
        size: Size
    ) -> EvalResult<'tcx, ScalarMaybeUndef> {
        // Make sure we don't read part of a pointer as a pointer
        self.check_relocation_edges(ptr, size)?;
        let endianness = self.endianness();
        // get_bytes_unchecked tests alignment
        let bytes = self.get_bytes_unchecked(ptr, size, ptr_align.min(self.int_align(size)))?;
        // Undef check happens *after* we established that the alignment is correct.
        // We must not return Ok() for unaligned pointers!
        if self.check_defined(ptr, size).is_err() {
            // this inflates undefined bytes to the entire scalar,
            // even if only a few bytes are undefined
            return Ok(ScalarMaybeUndef::Undef);
        }
        // Now we do the actual reading
        let bits = read_target_uint(endianness, bytes).unwrap();
        // See if we got a pointer
        if size != self.pointer_size() {
            if self.relocations(ptr, size)?.len() != 0 {
                return err!(ReadPointerAsBytes);
            }
        } else {
            let alloc = self.get(ptr.alloc_id)?;
            match alloc.relocations.get(&ptr.offset) {
                Some(&alloc_id) => {
                    let ptr = Pointer::new(alloc_id, Size::from_bytes(bits as u64));
                    return Ok(ScalarMaybeUndef::Scalar(ptr.into()))
                }
                None => {},
            }
        }
        // We don't. Just return the bits.
        Ok(ScalarMaybeUndef::Scalar(Scalar::Bits {
            bits,
            size: size.bytes() as u8,
        }))
    }

    pub fn read_ptr_sized(&self, ptr: Pointer, ptr_align: Align)
        -> EvalResult<'tcx, ScalarMaybeUndef> {
        self.read_scalar(ptr, ptr_align, self.pointer_size())
    }

    /// Write a *non-ZST* scalar
    pub fn write_scalar(
        &mut self,
        ptr: Pointer,
        ptr_align: Align,
        val: ScalarMaybeUndef,
        type_size: Size,
    ) -> EvalResult<'tcx> {
        let endianness = self.endianness();

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
                assert_eq!(truncate(bits, Size::from_bytes(size.into())), bits,
                    "Unexpected value of size {} when writing to memory", size);
                bits
            },
        };

        {
            // get_bytes_mut checks alignment
            let dst = self.get_bytes_mut(ptr, type_size, ptr_align)?;
            write_target_uint(endianness, dst, bytes).unwrap();
        }

        // See if we have to also write a relocation
        match val {
            Scalar::Ptr(val) => {
                self.get_mut(ptr.alloc_id)?.relocations.insert(
                    ptr.offset,
                    val.alloc_id,
                );
            }
            _ => {}
        }

        Ok(())
    }

    pub fn write_ptr_sized(&mut self, ptr: Pointer, ptr_align: Align, val: ScalarMaybeUndef)
        -> EvalResult<'tcx> {
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
        ity.align(self)
    }
}

/// Relocations
impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    fn relocations(
        &self,
        ptr: Pointer,
        size: Size,
    ) -> EvalResult<'tcx, &[(Size, AllocId)]> {
        let start = ptr.offset.bytes().saturating_sub(self.pointer_size().bytes() - 1);
        let end = ptr.offset + size;
        Ok(self.get(ptr.alloc_id)?.relocations.range(Size::from_bytes(start)..end))
    }

    fn clear_relocations(&mut self, ptr: Pointer, size: Size) -> EvalResult<'tcx> {
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

    fn check_relocation_edges(&self, ptr: Pointer, size: Size) -> EvalResult<'tcx> {
        let overlapping_start = self.relocations(ptr, Size::ZERO)?.len();
        let overlapping_end = self.relocations(ptr.offset(size, self)?, Size::ZERO)?.len();
        if overlapping_start + overlapping_end != 0 {
            return err!(ReadPointerAsBytes);
        }
        Ok(())
    }
}

/// Undefined bytes
impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    // FIXME(solson): This is a very naive, slow version.
    fn copy_undef_mask(
        &mut self,
        src: Pointer,
        dest: Pointer,
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

    fn check_defined(&self, ptr: Pointer, size: Size) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        if !alloc.undef_mask.is_range_defined(
            ptr.offset,
            ptr.offset + size,
        )
        {
            return err!(ReadUndefBytes);
        }
        Ok(())
    }

    pub fn mark_definedness(
        &mut self,
        ptr: Pointer,
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

////////////////////////////////////////////////////////////////////////////////
// Unaligned accesses
////////////////////////////////////////////////////////////////////////////////

pub trait HasMemory<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    fn memory_mut(&mut self) -> &mut Memory<'a, 'mir, 'tcx, M>;
    fn memory(&self) -> &Memory<'a, 'mir, 'tcx, M>;
}

impl<'a, 'mir, 'tcx, M> HasMemory<'a, 'mir, 'tcx, M> for Memory<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>
{
    #[inline]
    fn memory_mut(&mut self) -> &mut Memory<'a, 'mir, 'tcx, M> {
        self
    }

    #[inline]
    fn memory(&self) -> &Memory<'a, 'mir, 'tcx, M> {
        self
    }
}

impl<'a, 'mir, 'tcx, M> HasMemory<'a, 'mir, 'tcx, M> for EvalContext<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>
{
    #[inline]
    fn memory_mut(&mut self) -> &mut Memory<'a, 'mir, 'tcx, M> {
        &mut self.memory
    }

    #[inline]
    fn memory(&self) -> &Memory<'a, 'mir, 'tcx, M> {
        &self.memory
    }
}

impl<'a, 'mir, 'tcx, M> layout::HasDataLayout for &'a Memory<'a, 'mir, 'tcx, M>
    where M: Machine<'mir, 'tcx>
{
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}
