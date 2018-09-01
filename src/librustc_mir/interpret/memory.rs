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

use rustc::ty::{self, Instance, query::TyCtxtAt};
use rustc::ty::layout::{self, Align, TargetDataLayout, Size, HasDataLayout};
use rustc::mir::interpret::{Pointer, AllocId, Allocation, ConstValue, ScalarMaybeUndef, GlobalId,
                            EvalResult, Scalar, EvalErrorKind, AllocType, PointerArithmetic,
                            truncate};
pub use rustc::mir::interpret::{write_target_uint, read_target_uint};
use rustc_data_structures::fx::{FxHashSet, FxHashMap, FxHasher};

use syntax::ast::Mutability;

use super::Machine;

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum MemoryKind<T> {
    /// Error if deallocated except during a stack pop
    Stack,
    /// Additional memory kinds a machine wishes to distinguish from the builtin ones
    Machine(T),
}

#[derive(Clone)]
pub struct Memory<'a, 'mir, 'tcx: 'a + 'mir, M: Machine<'mir, 'tcx>> {
    /// Additional data required by the Machine
    pub data: M::MemoryData,

    /// Allocations local to this instance of the miri engine.  The kind
    /// helps ensure that the same mechanism is used for allocation and
    /// deallocation.  When an allocation is not found here, it is a
    /// static and looked up in the `tcx` for read access.  Writing to
    /// a static creates a copy here, in the machine.
    alloc_map: FxHashMap<AllocId, (MemoryKind<M::MemoryKinds>, Allocation)>,

    pub tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> HasDataLayout for &'a Memory<'a, 'mir, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}
impl<'a, 'b, 'c, 'mir, 'tcx, M: Machine<'mir, 'tcx>> HasDataLayout
    for &'b &'c mut Memory<'a, 'mir, 'tcx, M>
{
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
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
            alloc_map,
            tcx: _,
        } = self;

        *data == other.data
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
            alloc_map: _,
            tcx: _,
        } = self;

        data.hash(state);

        // We ignore some fields which don't change between evaluation steps.

        // Since HashMaps which contain the same items may have different
        // iteration orders, we use a commutative operation (in this case
        // addition, but XOR would also work), to combine the hash of each
        // `Allocation`.
        self.alloc_map.iter()
            .map(|(&id, alloc)| {
                let mut h = FxHasher::default();
                id.hash(&mut h);
                alloc.hash(&mut h);
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
            alloc_map: FxHashMap::default(),
            tcx,
        }
    }

    pub fn create_fn_alloc(&mut self, instance: Instance<'tcx>) -> Pointer {
        self.tcx.alloc_map.lock().create_fn_alloc(instance).into()
    }

    pub fn allocate_static_bytes(&mut self, bytes: &[u8]) -> Pointer {
        self.tcx.allocate_bytes(bytes).into()
    }

    pub fn allocate_with(
        &mut self,
        alloc: Allocation,
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
        self.allocate_with(Allocation::undef(size, align), kind).map(Pointer::from)
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

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc"
        let new_ptr = self.allocate(new_size, new_align, kind)?;
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
    pub fn deallocate_local(&mut self, ptr: Pointer) -> EvalResult<'tcx> {
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
        ptr: Pointer,
        size_and_align: Option<(Size, Align)>,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx> {
        if ptr.offset.bytes() != 0 {
            return err!(DeallocateNonBasePtr);
        }

        let (alloc_kind, alloc) = match self.alloc_map.remove(&ptr.alloc_id) {
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

        debug!("deallocated : {}", ptr.alloc_id);

        Ok(())
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
                assert!(bits < (1u128 << self.pointer_size().bits()));
                if bits == 0 {
                    return err!(InvalidNullPointerUsage);
                }
                // the "base address" is 0 and hence always aligned
                (bits as u64, required_align)
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
    /// in-bounds!  This follows C's/LLVM's rules.  The `access` boolean is just used
    /// for the error message.
    /// If you want to check bounds before doing a memory access, be sure to
    /// check the pointer one past the end of your access, then everything will
    /// work out exactly.
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
    /// Helper function to obtain the global (tcx) allocation for a static
    fn get_static_alloc(
        tcx: TyCtxtAt<'a, 'tcx, 'tcx>,
        id: AllocId,
    ) -> EvalResult<'tcx, &'tcx Allocation> {
        let alloc = tcx.alloc_map.lock().get(id);
        let def_id = match alloc {
            Some(AllocType::Memory(mem)) => {
                return Ok(mem)
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
        tcx.const_eval(ty::ParamEnv::reveal_all().and(gid)).map_err(|err| {
            // no need to report anything, the const_eval call takes care of that for statics
            assert!(tcx.is_static(def_id).is_some());
            EvalErrorKind::ReferencedConstant(err).into()
        }).map(|const_val| {
            if let ConstValue::ByRef(_, allocation, _) = const_val.val {
                allocation
            } else {
                panic!("Matching on non-ByRef static")
            }
        })
    }

    pub fn get(&self, id: AllocId) -> EvalResult<'tcx, &Allocation> {
        match self.alloc_map.get(&id) {
            // Normal alloc?
            Some(alloc) => Ok(&alloc.1),
            // Static. No need to make any copies, just provide read access to the global static
            // memory in tcx.
            None => Self::get_static_alloc(self.tcx, id),
        }
    }

    pub fn get_mut(
        &mut self,
        id: AllocId,
    ) -> EvalResult<'tcx, &mut Allocation> {
        // Static?
        if !self.alloc_map.contains_key(&id) {
            // Ask the machine for what to do
            if let Some(kind) = M::MUT_STATIC_KIND {
                // The machine supports mutating statics.  Make a copy, use that.
                self.deep_copy_static(id, MemoryKind::Machine(kind))?;
            } else {
                return err!(ModifiedConstantMemory)
            }
        }
        // If we come here, we know the allocation is in our map
        let alloc = &mut self.alloc_map.get_mut(&id).unwrap().1;
        // See if we are allowed to mutate this
        if alloc.mutability == Mutability::Immutable {
            err!(ModifiedConstantMemory)
        } else {
            Ok(alloc)
        }
    }

    pub fn get_fn(&self, ptr: Pointer) -> EvalResult<'tcx, Instance<'tcx>> {
        if ptr.offset.bytes() != 0 {
            return err!(InvalidFunctionPointer);
        }
        trace!("reading fn ptr: {}", ptr.alloc_id);
        match self.tcx.alloc_map.lock().get(ptr.alloc_id) {
            Some(AllocType::Function(instance)) => Ok(instance),
            _ => Err(EvalErrorKind::ExecuteMemory.into()),
        }
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
                    Some((kind, alloc)) => (alloc, match kind {
                        MemoryKind::Stack => " (stack)".to_owned(),
                        MemoryKind::Machine(m) => format!(" ({:?})", m),
                    }),
                    None => {
                        // static alloc?
                        match self.tcx.alloc_map.lock().get(id) {
                            Some(AllocType::Memory(a)) => (a, " (immutable)".to_owned()),
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
        let mut_static_kind = M::MUT_STATIC_KIND.map(|k| MemoryKind::Machine(k));
        let leaks: Vec<_> = self.alloc_map
            .iter()
            .filter_map(|(&id, &(kind, _))|
                // exclude mutable statics
                if Some(kind) == mut_static_kind { None } else { Some(id) } )
            .collect();
        let n = leaks.len();
        self.dump_allocs(leaks);
        n
    }
}

/// Byte accessors
impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
    /// The last argument controls whether we error out when there are undefined
    /// or pointer bytes.  You should never call this, call `get_bytes` or
    /// `get_bytes_with_undef_and_ptr` instead,
    fn get_bytes_internal(
        &self,
        ptr: Pointer,
        size: Size,
        align: Align,
        check_defined_and_ptr: bool,
    ) -> EvalResult<'tcx, &[u8]> {
        assert_ne!(size.bytes(), 0, "0-sized accesses should never even get a `Pointer`");
        self.check_align(ptr.into(), align)?;
        // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        self.check_bounds(ptr.offset(size, &*self)?, true)?;

        if check_defined_and_ptr {
            self.check_defined(ptr, size)?;
            self.check_relocations(ptr, size)?;
        } else {
            // We still don't want relocations on the *edges*
            self.check_relocation_edges(ptr, size)?;
        }

        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&alloc.bytes[offset..offset + size.bytes() as usize])
    }

    #[inline]
    fn get_bytes(&self, ptr: Pointer, size: Size, align: Align) -> EvalResult<'tcx, &[u8]> {
        self.get_bytes_internal(ptr, size, align, true)
    }

    /// It is the caller's responsibility to handle undefined and pointer bytes.
    /// However, this still checks that there are no relocations on the egdes.
    #[inline]
    fn get_bytes_with_undef_and_ptr(
        &self,
        ptr: Pointer,
        size: Size,
        align: Align
    ) -> EvalResult<'tcx, &[u8]> {
        self.get_bytes_internal(ptr, size, align, false)
    }

    /// Just calling this already marks everything as defined and removes relocations,
    /// so be sure to actually put data there!
    fn get_bytes_mut(
        &mut self,
        ptr: Pointer,
        size: Size,
        align: Align,
    ) -> EvalResult<'tcx, &mut [u8]> {
        assert_ne!(size.bytes(), 0, "0-sized accesses should never even get a `Pointer`");
        self.check_align(ptr.into(), align)?;
        // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        self.check_bounds(ptr.offset(size, &self)?, true)?;

        self.mark_definedness(ptr, size, true)?;
        self.clear_relocations(ptr, size)?;

        let alloc = self.get_mut(ptr.alloc_id)?;
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&mut alloc.bytes[offset..offset + size.bytes() as usize])
    }
}

/// Reading and writing
impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> Memory<'a, 'mir, 'tcx, M> {
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
            MemoryKind::Stack => {},
        }
        // ensure llvm knows not to put this into immutable memory
        alloc.mutability = mutability;
        let alloc = self.tcx.intern_const_alloc(alloc);
        self.tcx.alloc_map.lock().set_id_memory(alloc_id, alloc);
        // recurse into inner allocations
        for &alloc in alloc.relocations.values() {
            // FIXME: Reusing the mutability here is likely incorrect.  It is originally
            // determined via `is_freeze`, and data is considered frozen if there is no
            // `UnsafeCell` *immediately* in that data -- however, this search stops
            // at references.  So whenever we follow a reference, we should likely
            // assume immutability -- and we should make sure that the compiler
            // does not permit code that would break this!
            if self.alloc_map.contains_key(&alloc) {
                // Not yet interned, so proceed recursively
                self.intern_static(alloc, mutability)?;
            }
        }
        Ok(())
    }

    /// The alloc_id must refer to a (mutable) static; a deep copy of that
    /// static is made into this memory.
    fn deep_copy_static(
        &mut self,
        id: AllocId,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx> {
        let alloc = Self::get_static_alloc(self.tcx, id)?;
        if alloc.mutability == Mutability::Immutable {
            return err!(ModifiedConstantMemory);
        }
        let old = self.alloc_map.insert(id, (kind, alloc.clone()));
        assert!(old.is_none(), "deep_copy_static: must not overwrite existing memory");
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
                    .map(|&(offset, alloc_id)| {
                    (offset + dest.offset - src.offset + (i * size * relocations.len() as u64),
                    alloc_id)
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

    pub fn read_c_str(&self, ptr: Pointer) -> EvalResult<'tcx, &[u8]> {
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
        // get_bytes_unchecked tests alignment and relocation edges
        let bytes = self.get_bytes_with_undef_and_ptr(
            ptr, size, ptr_align.min(self.int_align(size))
        )?;
        // Undef check happens *after* we established that the alignment is correct.
        // We must not return Ok() for unaligned pointers!
        if !self.is_defined(ptr, size)? {
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
                Some(&alloc_id) => {
                    let ptr = Pointer::new(alloc_id, Size::from_bytes(bits as u64));
                    return Ok(ScalarMaybeUndef::Scalar(ptr.into()))
                }
                None => {},
            }
        }
        // We don't. Just return the bits.
        Ok(ScalarMaybeUndef::Scalar(Scalar::from_uint(bits, size)))
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
    /// Return all relocations overlapping with the given ptr-offset pair.
    fn relocations(
        &self,
        ptr: Pointer,
        size: Size,
    ) -> EvalResult<'tcx, &[(Size, AllocId)]> {
        // We have to go back `pointer_size - 1` bytes, as that one would still overlap with
        // the beginning of this range.
        let start = ptr.offset.bytes().saturating_sub(self.pointer_size().bytes() - 1);
        let end = ptr.offset + size; // this does overflow checking
        Ok(self.get(ptr.alloc_id)?.relocations.range(Size::from_bytes(start)..end))
    }

    /// Check that there ar eno relocations overlapping with the given range.
    #[inline(always)]
    fn check_relocations(&self, ptr: Pointer, size: Size) -> EvalResult<'tcx> {
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

    /// Error if there are relocations overlapping with the egdes of the
    /// given memory range.
    #[inline]
    fn check_relocation_edges(&self, ptr: Pointer, size: Size) -> EvalResult<'tcx> {
        self.check_relocations(ptr, Size::ZERO)?;
        self.check_relocations(ptr.offset(size, self)?, Size::ZERO)?;
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

    fn is_defined(&self, ptr: Pointer, size: Size) -> EvalResult<'tcx, bool> {
        let alloc = self.get(ptr.alloc_id)?;
        Ok(alloc.undef_mask.is_range_defined(
            ptr.offset,
            ptr.offset + size,
        ))
    }

    #[inline]
    fn check_defined(&self, ptr: Pointer, size: Size) -> EvalResult<'tcx> {
        if self.is_defined(ptr, size)? {
            Ok(())
        } else {
            err!(ReadUndefBytes)
        }
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
