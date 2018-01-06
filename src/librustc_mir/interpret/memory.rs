use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian, BigEndian};
use std::collections::{btree_map, BTreeMap, HashMap, HashSet, VecDeque};
use std::{ptr, mem, io};

use rustc::ty::{Instance, TyCtxt};
use rustc::ty::layout::{self, Align, TargetDataLayout};
use syntax::ast::Mutability;

use rustc::mir::interpret::{MemoryPointer, AllocId, Allocation, AccessKind, UndefMask, Value, Pointer,
                            EvalResult, PrimVal, EvalErrorKind};

use super::{EvalContext, Machine};

////////////////////////////////////////////////////////////////////////////////
// Allocations and pointers
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum MemoryKind<T> {
    /// Error if deallocated except during a stack pop
    Stack,
    /// A mutable Static. All the others are interned in the tcx
    MutableStatic, // FIXME: move me into the machine, rustc const eval doesn't need them
    /// Additional memory kinds a machine wishes to distinguish from the builtin ones
    Machine(T),
}

////////////////////////////////////////////////////////////////////////////////
// Top-level interpreter memory
////////////////////////////////////////////////////////////////////////////////

pub struct Memory<'a, 'tcx: 'a, M: Machine<'tcx>> {
    /// Additional data required by the Machine
    pub data: M::MemoryData,

    /// Helps guarantee that stack allocations aren't deallocated via `rust_deallocate`
    alloc_kind: HashMap<AllocId, MemoryKind<M::MemoryKinds>>,

    /// Actual memory allocations (arbitrary bytes, may contain pointers into other allocations).
    alloc_map: HashMap<AllocId, Allocation>,

    /// Actual memory allocations (arbitrary bytes, may contain pointers into other allocations).
    ///
    /// Stores statics while they are being processed, before they are interned and thus frozen
    uninitialized_statics: HashMap<AllocId, Allocation>,

    /// Number of virtual bytes allocated.
    memory_usage: u64,

    /// Maximum number of virtual bytes that may be allocated.
    memory_size: u64,

    /// The current stack frame.  Used to check accesses against locks.
    pub cur_frame: usize,

    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, max_memory: u64, data: M::MemoryData) -> Self {
        Memory {
            data,
            alloc_kind: HashMap::new(),
            alloc_map: HashMap::new(),
            uninitialized_statics: HashMap::new(),
            tcx,
            memory_size: max_memory,
            memory_usage: 0,
            cur_frame: usize::max_value(),
        }
    }

    pub fn allocations<'x>(
        &'x self,
    ) -> impl Iterator<Item = (AllocId, &'x Allocation)> {
        self.alloc_map.iter().map(|(&id, alloc)| (id, alloc))
    }

    pub fn create_fn_alloc(&mut self, instance: Instance<'tcx>) -> MemoryPointer {
        let id = self.tcx.interpret_interner.borrow_mut().create_fn_alloc(instance);
        MemoryPointer::new(id, 0)
    }

    pub fn allocate_cached(&mut self, bytes: &[u8]) -> MemoryPointer {
        let id = self.tcx.allocate_cached(bytes);
        MemoryPointer::new(id, 0)
    }

    /// kind is `None` for statics
    pub fn allocate(
        &mut self,
        size: u64,
        align: Align,
        kind: Option<MemoryKind<M::MemoryKinds>>,
    ) -> EvalResult<'tcx, MemoryPointer> {
        if self.memory_size - self.memory_usage < size {
            return err!(OutOfMemory {
                allocation_size: size,
                memory_size: self.memory_size,
                memory_usage: self.memory_usage,
            });
        }
        self.memory_usage += size;
        assert_eq!(size as usize as u64, size);
        let alloc = Allocation {
            bytes: vec![0; size as usize],
            relocations: BTreeMap::new(),
            undef_mask: UndefMask::new(size),
            align,
        };
        let id = self.tcx.interpret_interner.borrow_mut().reserve();
        M::add_lock(self, id);
        match kind {
            Some(kind @ MemoryKind::Stack) |
            Some(kind @ MemoryKind::Machine(_)) => {
                self.alloc_map.insert(id, alloc);
                self.alloc_kind.insert(id, kind);
            },
            None => {
                self.uninitialized_statics.insert(id, alloc);
            },
            Some(MemoryKind::MutableStatic) => bug!("don't allocate mutable statics directly")
        }
        Ok(MemoryPointer::new(id, 0))
    }

    pub fn reallocate(
        &mut self,
        ptr: MemoryPointer,
        old_size: u64,
        old_align: Align,
        new_size: u64,
        new_align: Align,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, MemoryPointer> {
        if ptr.offset != 0 {
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
        let new_ptr = self.allocate(new_size, new_align, Some(kind))?;
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

    pub fn deallocate_local(&mut self, ptr: MemoryPointer) -> EvalResult<'tcx> {
        match self.alloc_kind.get(&ptr.alloc_id).cloned() {
            // for a constant like `const FOO: &i32 = &1;` the local containing
            // the `1` is referred to by the global. We transitively marked everything
            // the global refers to as static itself, so we don't free it here
            Some(MemoryKind::MutableStatic) => Ok(()),
            Some(MemoryKind::Stack) => self.deallocate(ptr, None, MemoryKind::Stack),
            // Happens if the memory was interned into immutable memory
            None => Ok(()),
            other => bug!("local contained non-stack memory: {:?}", other),
        }
    }

    pub fn deallocate(
        &mut self,
        ptr: MemoryPointer,
        size_and_align: Option<(u64, Align)>,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx> {
        if ptr.offset != 0 {
            return err!(DeallocateNonBasePtr);
        }

        let alloc = match self.alloc_map.remove(&ptr.alloc_id) {
            Some(alloc) => alloc,
            None => if self.uninitialized_statics.contains_key(&ptr.alloc_id) {
                return err!(DeallocatedWrongMemoryKind(
                    "uninitializedstatic".to_string(),
                    format!("{:?}", kind),
                ))
            } else if self.tcx.interpret_interner.borrow().get_fn(ptr.alloc_id).is_some() {
                return err!(DeallocatedWrongMemoryKind(
                    "function".to_string(),
                    format!("{:?}", kind),
                ))
            } else if self.tcx.interpret_interner.borrow().get_alloc(ptr.alloc_id).is_some() {
                return err!(DeallocatedWrongMemoryKind(
                    "static".to_string(),
                    format!("{:?}", kind),
                ))
            } else {
                return err!(DoubleFree)
            },
        };

        let alloc_kind = self.alloc_kind.remove(&ptr.alloc_id).expect("alloc_map out of sync with alloc_kind");

        // It is okay for us to still holds locks on deallocation -- for example, we could store data we own
        // in a local, and the local could be deallocated (from StorageDead) before the function returns.
        // However, we should check *something*.  For now, we make sure that there is no conflicting write
        // lock by another frame.  We *have* to permit deallocation if we hold a read lock.
        // TODO: Figure out the exact rules here.
        M::free_lock(self, ptr.alloc_id, alloc.bytes.len() as u64)?;

        if alloc_kind != kind {
            return err!(DeallocatedWrongMemoryKind(
                format!("{:?}", alloc_kind),
                format!("{:?}", kind),
            ));
        }
        if let Some((size, align)) = size_and_align {
            if size != alloc.bytes.len() as u64 || align != alloc.align {
                return err!(IncorrectAllocationInformation(size, alloc.bytes.len(), align.abi(), alloc.align.abi()));
            }
        }

        self.memory_usage -= alloc.bytes.len() as u64;
        debug!("deallocated : {}", ptr.alloc_id);

        Ok(())
    }

    pub fn pointer_size(&self) -> u64 {
        self.tcx.data_layout.pointer_size.bytes()
    }

    pub fn endianess(&self) -> layout::Endian {
        self.tcx.data_layout.endian
    }

    /// Check that the pointer is aligned AND non-NULL.
    pub fn check_align(&self, ptr: Pointer, required_align: Align) -> EvalResult<'tcx> {
        // Check non-NULL/Undef, extract offset
        let (offset, alloc_align) = match ptr.into_inner_primval() {
            PrimVal::Ptr(ptr) => {
                let alloc = self.get(ptr.alloc_id)?;
                (ptr.offset, alloc.align)
            }
            PrimVal::Bytes(bytes) => {
                let v = ((bytes as u128) % (1 << self.pointer_size())) as u64;
                if v == 0 {
                    return err!(InvalidNullPointerUsage);
                }
                // the base address if the "integer allocation" is 0 and hence always aligned
                (v, required_align)
            }
            PrimVal::Undef => return err!(ReadUndefBytes),
        };
        // Check alignment
        if alloc_align.abi() < required_align.abi() {
            return err!(AlignmentCheckFailed {
                has: alloc_align.abi(),
                required: required_align.abi(),
            });
        }
        if offset % required_align.abi() == 0 {
            Ok(())
        } else {
            err!(AlignmentCheckFailed {
                has: offset % required_align.abi(),
                required: required_align.abi(),
            })
        }
    }

    pub fn check_bounds(&self, ptr: MemoryPointer, access: bool) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        let allocation_size = alloc.bytes.len() as u64;
        if ptr.offset > allocation_size {
            return err!(PointerOutOfBounds {
                ptr,
                access,
                allocation_size,
            });
        }
        Ok(())
    }
}

/// Allocation accessors
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    pub fn get(&self, id: AllocId) -> EvalResult<'tcx, &Allocation> {
        // normal alloc?
        match self.alloc_map.get(&id) {
                    Some(alloc) => Ok(alloc),
            // uninitialized static alloc?
            None => match self.uninitialized_statics.get(&id) {
                Some(alloc) => Ok(alloc),
                None => {
                    let int = self.tcx.interpret_interner.borrow();
                    // static alloc?
                    int.get_alloc(id)
                        // no alloc? produce an error
                        .ok_or_else(|| if int.get_fn(id).is_some() {
                            EvalErrorKind::DerefFunctionPointer.into()
                        } else {
                            EvalErrorKind::DanglingPointerDeref.into()
                        })
                },
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
            None => match self.uninitialized_statics.get_mut(&id) {
                Some(alloc) => Ok(alloc),
                None => {
                    let int = self.tcx.interpret_interner.borrow();
                    // no alloc or immutable alloc? produce an error
                    if int.get_alloc(id).is_some() {
                        err!(ModifiedConstantMemory)
                    } else if int.get_fn(id).is_some() {
                        err!(DerefFunctionPointer)
                    } else {
                        err!(DanglingPointerDeref)
                    }
                },
            },
        }
    }

    pub fn get_fn(&self, ptr: MemoryPointer) -> EvalResult<'tcx, Instance<'tcx>> {
        if ptr.offset != 0 {
            return err!(InvalidFunctionPointer);
        }
        debug!("reading fn ptr: {}", ptr.alloc_id);
        self.tcx
            .interpret_interner
            .borrow()
            .get_fn(ptr.alloc_id)
            .ok_or(EvalErrorKind::ExecuteMemory.into())
    }

    /// For debugging, print an allocation and all allocations it points to, recursively.
    pub fn dump_alloc(&self, id: AllocId) {
        self.dump_allocs(vec![id]);
    }

    /// For debugging, print a list of allocations and all allocations they point to, recursively.
    pub fn dump_allocs(&self, mut allocs: Vec<AllocId>) {
        use std::fmt::Write;
        allocs.sort();
        allocs.dedup();
        let mut allocs_to_print = VecDeque::from(allocs);
        let mut allocs_seen = HashSet::new();

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
                        MemoryKind::MutableStatic => " (static mut)".to_owned(),
                    }),
                    // uninitialized static alloc?
                    None => match self.uninitialized_statics.get(&id) {
                        Some(a) => (a, " (static in the process of initialization)".to_owned()),
                        None => {
                            let int = self.tcx.interpret_interner.borrow();
                            // static alloc?
                            match int.get_alloc(id) {
                                Some(a) => (a, "(immutable)".to_owned()),
                                None => if let Some(func) = int.get_fn(id) {
                                    trace!("{} {}", msg, func);
                    continue;
                                } else {
                            trace!("{} (deallocated)", msg);
                            continue;
                                },
                }
                        },
                    },
            };

            for i in 0..(alloc.bytes.len() as u64) {
                if let Some(&target_id) = alloc.relocations.get(&i) {
                    if allocs_seen.insert(target_id) {
                        allocs_to_print.push_back(target_id);
                    }
                    relocations.push((i, target_id));
                }
                if alloc.undef_mask.is_range_defined(i, i + 1) {
                    // this `as usize` is fine, since `i` came from a `usize`
                    write!(msg, "{:02x} ", alloc.bytes[i as usize]).unwrap();
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
                let mut pos = 0;
                let relocation_width = (self.pointer_size() - 1) * 3;
                for (i, target_id) in relocations {
                    // this `as usize` is fine, since we can't print more chars than `usize::MAX`
                    write!(msg, "{:1$}", "", ((i - pos) * 3) as usize).unwrap();
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
        let kinds = &self.alloc_kind;
        let leaks: Vec<_> = self.alloc_map
            .keys()
            .filter_map(|key| if kinds[key] != MemoryKind::MutableStatic {
                Some(*key)
            } else {
                None
            })
            .collect();
        let n = leaks.len();
        self.dump_allocs(leaks);
        n
    }
}

/// Byte accessors
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    fn get_bytes_unchecked(
        &self,
        ptr: MemoryPointer,
        size: u64,
        align: Align,
    ) -> EvalResult<'tcx, &[u8]> {
        // Zero-sized accesses can use dangling pointers, but they still have to be aligned and non-NULL
        self.check_align(ptr.into(), align)?;
        if size == 0 {
            return Ok(&[]);
        }
        M::check_locks(self, ptr, size, AccessKind::Read)?;
        self.check_bounds(ptr.offset(size, self)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        assert_eq!(size as usize as u64, size);
        let offset = ptr.offset as usize;
        Ok(&alloc.bytes[offset..offset + size as usize])
    }

    fn get_bytes_unchecked_mut(
        &mut self,
        ptr: MemoryPointer,
        size: u64,
        align: Align,
    ) -> EvalResult<'tcx, &mut [u8]> {
        // Zero-sized accesses can use dangling pointers, but they still have to be aligned and non-NULL
        self.check_align(ptr.into(), align)?;
        if size == 0 {
            return Ok(&mut []);
        }
        M::check_locks(self, ptr, size, AccessKind::Write)?;
        self.check_bounds(ptr.offset(size, &*self)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        let alloc = self.get_mut(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        assert_eq!(size as usize as u64, size);
        let offset = ptr.offset as usize;
        Ok(&mut alloc.bytes[offset..offset + size as usize])
    }

    fn get_bytes(&self, ptr: MemoryPointer, size: u64, align: Align) -> EvalResult<'tcx, &[u8]> {
        assert_ne!(size, 0);
        if self.relocations(ptr, size)?.count() != 0 {
            return err!(ReadPointerAsBytes);
        }
        self.check_defined(ptr, size)?;
        self.get_bytes_unchecked(ptr, size, align)
    }

    fn get_bytes_mut(
        &mut self,
        ptr: MemoryPointer,
        size: u64,
        align: Align,
    ) -> EvalResult<'tcx, &mut [u8]> {
        assert_ne!(size, 0);
        self.clear_relocations(ptr, size)?;
        self.mark_definedness(ptr.into(), size, true)?;
        self.get_bytes_unchecked_mut(ptr, size, align)
    }
}

/// Reading and writing
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    /// mark an allocation pointed to by a static as static and initialized
    fn mark_inner_allocation_initialized(
        &mut self,
        alloc: AllocId,
        mutability: Mutability,
    ) -> EvalResult<'tcx> {
        match self.alloc_kind.get(&alloc) {
            // do not go into immutable statics
            None |
            // or mutable statics
            Some(&MemoryKind::MutableStatic) => Ok(()),
            // just locals and machine allocs
            Some(_) => self.mark_static_initalized(alloc, mutability),
        }
    }

    /// mark an allocation as static and initialized, either mutable or not
    pub fn mark_static_initalized(
        &mut self,
        alloc_id: AllocId,
        mutability: Mutability,
    ) -> EvalResult<'tcx> {
        trace!(
            "mark_static_initalized {:?}, mutability: {:?}",
            alloc_id,
            mutability
        );
        if mutability == Mutability::Immutable {
            let alloc = self.alloc_map.remove(&alloc_id);
            let kind = self.alloc_kind.remove(&alloc_id);
            assert_ne!(kind, Some(MemoryKind::MutableStatic));
            let uninit = self.uninitialized_statics.remove(&alloc_id);
            if let Some(alloc) = alloc.or(uninit) {
                let alloc = self.tcx.intern_const_alloc(alloc);
                self.tcx.interpret_interner.borrow_mut().intern_at_reserved(alloc_id, alloc);
                // recurse into inner allocations
                for &alloc in alloc.relocations.values() {
                    self.mark_inner_allocation_initialized(alloc, mutability)?;
                }
            }
            return Ok(());
        }
        // We are marking the static as initialized, so move it out of the uninit map
        if let Some(uninit) = self.uninitialized_statics.remove(&alloc_id) {
            self.alloc_map.insert(alloc_id, uninit);
        }
        // do not use `self.get_mut(alloc_id)` here, because we might have already marked a
        // sub-element or have circular pointers (e.g. `Rc`-cycles)
        let relocations = match self.alloc_map.get_mut(&alloc_id) {
            Some(&mut Allocation {
                     ref mut relocations,
                     ..
                 }) => {
                match self.alloc_kind.get(&alloc_id) {
                    // const eval results can refer to "locals".
                    // E.g. `const Foo: &u32 = &1;` refers to the temp local that stores the `1`
                    None |
                    Some(&MemoryKind::Stack) => {},
                    Some(&MemoryKind::Machine(m)) => M::mark_static_initialized(m)?,
                    Some(&MemoryKind::MutableStatic) => {
                        trace!("mark_static_initalized: skipping already initialized static referred to by static currently being initialized");
                        return Ok(());
                    },
                }
                // overwrite or insert
                self.alloc_kind.insert(alloc_id, MemoryKind::MutableStatic);
                // take out the relocations vector to free the borrow on self, so we can call
                // mark recursively
                mem::replace(relocations, Default::default())
            }
            None => return err!(DanglingPointerDeref),
        };
        // recurse into inner allocations
        for &alloc in relocations.values() {
            self.mark_inner_allocation_initialized(alloc, mutability)?;
        }
        // put back the relocations
        self.alloc_map
            .get_mut(&alloc_id)
            .expect("checked above")
            .relocations = relocations;
        Ok(())
    }

    pub fn copy(
        &mut self,
        src: Pointer,
        src_align: Align,
        dest: Pointer,
        dest_align: Align,
        size: u64,
        nonoverlapping: bool,
    ) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be aligned
        self.check_align(src, src_align)?;
        self.check_align(dest, dest_align)?;
        if size == 0 {
            return Ok(());
        }
        let src = src.to_ptr()?;
        let dest = dest.to_ptr()?;
        self.check_relocation_edges(src, size)?;

        // first copy the relocations to a temporary buffer, because
        // `get_bytes_mut` will clear the relocations, which is correct,
        // since we don't want to keep any relocations at the target.

        let relocations: Vec<_> = self.relocations(src, size)?
            .map(|(&offset, &alloc_id)| {
                // Update relocation offsets for the new positions in the destination allocation.
                (offset + dest.offset - src.offset, alloc_id)
            })
            .collect();

        let src_bytes = self.get_bytes_unchecked(src, size, src_align)?.as_ptr();
        let dest_bytes = self.get_bytes_mut(dest, size, dest_align)?.as_mut_ptr();

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
        unsafe {
            assert_eq!(size as usize as u64, size);
            if src.alloc_id == dest.alloc_id {
                if nonoverlapping {
                    if (src.offset <= dest.offset && src.offset + size > dest.offset) ||
                        (dest.offset <= src.offset && dest.offset + size > src.offset)
                    {
                        return err!(Intrinsic(
                            format!("copy_nonoverlapping called on overlapping ranges"),
                        ));
                    }
                }
                ptr::copy(src_bytes, dest_bytes, size as usize);
            } else {
                ptr::copy_nonoverlapping(src_bytes, dest_bytes, size as usize);
            }
        }

        self.copy_undef_mask(src, dest, size)?;
        // copy back the relocations
        self.get_mut(dest.alloc_id)?.relocations.extend(relocations);

        Ok(())
    }

    pub fn read_c_str(&self, ptr: MemoryPointer) -> EvalResult<'tcx, &[u8]> {
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        let offset = ptr.offset as usize;
        match alloc.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                if self.relocations(ptr, (size + 1) as u64)?.count() != 0 {
                    return err!(ReadPointerAsBytes);
                }
                self.check_defined(ptr, (size + 1) as u64)?;
                M::check_locks(self, ptr, (size + 1) as u64, AccessKind::Read)?;
                Ok(&alloc.bytes[offset..offset + size])
            }
            None => err!(UnterminatedCString(ptr)),
        }
    }

    pub fn read_bytes(&self, ptr: Pointer, size: u64) -> EvalResult<'tcx, &[u8]> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        self.check_align(ptr, align)?;
        if size == 0 {
            return Ok(&[]);
        }
        self.get_bytes(ptr.to_ptr()?, size, align)
    }

    pub fn write_bytes(&mut self, ptr: Pointer, src: &[u8]) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        self.check_align(ptr, align)?;
        if src.is_empty() {
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, src.len() as u64, align)?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    pub fn write_repeat(&mut self, ptr: Pointer, val: u8, count: u64) -> EvalResult<'tcx> {
        // Empty accesses don't need to be valid pointers, but they should still be non-NULL
        let align = Align::from_bytes(1, 1).unwrap();
        self.check_align(ptr, align)?;
        if count == 0 {
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, count, align)?;
        for b in bytes {
            *b = val;
        }
        Ok(())
    }

    pub fn read_primval(&self, ptr: MemoryPointer, ptr_align: Align, size: u64, signed: bool) -> EvalResult<'tcx, PrimVal> {
        self.check_relocation_edges(ptr, size)?; // Make sure we don't read part of a pointer as a pointer
        let endianess = self.endianess();
        let bytes = self.get_bytes_unchecked(ptr, size, ptr_align.min(self.int_align(size)))?;
        // Undef check happens *after* we established that the alignment is correct.
        // We must not return Ok() for unaligned pointers!
        if self.check_defined(ptr, size).is_err() {
            return Ok(PrimVal::Undef.into());
        }
        // Now we do the actual reading
        let bytes = if signed {
            read_target_int(endianess, bytes).unwrap() as u128
        } else {
            read_target_uint(endianess, bytes).unwrap()
        };
        // See if we got a pointer
        if size != self.pointer_size() {
            if self.relocations(ptr, size)?.count() != 0 {
                return err!(ReadPointerAsBytes);
            }
        } else {
            let alloc = self.get(ptr.alloc_id)?;
            match alloc.relocations.get(&ptr.offset) {
                Some(&alloc_id) => return Ok(PrimVal::Ptr(MemoryPointer::new(alloc_id, bytes as u64))),
                None => {},
            }
        }
        // We don't. Just return the bytes.
        Ok(PrimVal::Bytes(bytes))
    }

    pub fn read_ptr_sized_unsigned(&self, ptr: MemoryPointer, ptr_align: Align) -> EvalResult<'tcx, PrimVal> {
        self.read_primval(ptr, ptr_align, self.pointer_size(), false)
    }

    pub fn write_primval(&mut self, ptr: MemoryPointer, ptr_align: Align, val: PrimVal, size: u64, signed: bool) -> EvalResult<'tcx> {
        let endianess = self.endianess();

        let bytes = match val {
            PrimVal::Ptr(val) => {
                assert_eq!(size, self.pointer_size());
                val.offset as u128
            }

            PrimVal::Bytes(bytes) => {
                // We need to mask here, or the byteorder crate can die when given a u64 larger
                // than fits in an integer of the requested size.
                let mask = match size {
                    1 => !0u8 as u128,
                    2 => !0u16 as u128,
                    4 => !0u32 as u128,
                    8 => !0u64 as u128,
                    16 => !0,
                    n => bug!("unexpected PrimVal::Bytes size: {}", n),
                };
                bytes & mask
            }

            PrimVal::Undef => {
                self.mark_definedness(PrimVal::Ptr(ptr).into(), size, false)?;
                return Ok(());
            }
        };

        {
            let align = self.int_align(size);
            let dst = self.get_bytes_mut(ptr, size, ptr_align.min(align))?;
            if signed {
                write_target_int(endianess, dst, bytes as i128).unwrap();
            } else {
                write_target_uint(endianess, dst, bytes).unwrap();
            }
        }

        // See if we have to also write a relocation
        match val {
            PrimVal::Ptr(val) => {
                self.get_mut(ptr.alloc_id)?.relocations.insert(
                    ptr.offset,
                    val.alloc_id,
                );
            }
            _ => {}
        }

        Ok(())
    }

    pub fn write_ptr_sized_unsigned(&mut self, ptr: MemoryPointer, ptr_align: Align, val: PrimVal) -> EvalResult<'tcx> {
        let ptr_size = self.pointer_size();
        self.write_primval(ptr, ptr_align, val, ptr_size, false)
    }

    fn int_align(&self, size: u64) -> Align {
        // We assume pointer-sized integers have the same alignment as pointers.
        // We also assume signed and unsigned integers of the same size have the same alignment.
        let ity = match size {
            1 => layout::I8,
            2 => layout::I16,
            4 => layout::I32,
            8 => layout::I64,
            16 => layout::I128,
            _ => bug!("bad integer size: {}", size),
        };
        ity.align(self)
    }
}

/// Relocations
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    fn relocations(
        &self,
        ptr: MemoryPointer,
        size: u64,
    ) -> EvalResult<'tcx, btree_map::Range<u64, AllocId>> {
        let start = ptr.offset.saturating_sub(self.pointer_size() - 1);
        let end = ptr.offset + size;
        Ok(self.get(ptr.alloc_id)?.relocations.range(start..end))
    }

    fn clear_relocations(&mut self, ptr: MemoryPointer, size: u64) -> EvalResult<'tcx> {
        // Find all relocations overlapping the given range.
        let keys: Vec<_> = self.relocations(ptr, size)?.map(|(&k, _)| k).collect();
        if keys.is_empty() {
            return Ok(());
        }

        // Find the start and end of the given range and its outermost relocations.
        let start = ptr.offset;
        let end = start + size;
        let first = *keys.first().unwrap();
        let last = *keys.last().unwrap() + self.pointer_size();

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
        for k in keys {
            alloc.relocations.remove(&k);
        }

        Ok(())
    }

    fn check_relocation_edges(&self, ptr: MemoryPointer, size: u64) -> EvalResult<'tcx> {
        let overlapping_start = self.relocations(ptr, 0)?.count();
        let overlapping_end = self.relocations(ptr.offset(size, self)?, 0)?.count();
        if overlapping_start + overlapping_end != 0 {
            return err!(ReadPointerAsBytes);
        }
        Ok(())
    }
}

/// Undefined bytes
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    // FIXME(solson): This is a very naive, slow version.
    fn copy_undef_mask(
        &mut self,
        src: MemoryPointer,
        dest: MemoryPointer,
        size: u64,
    ) -> EvalResult<'tcx> {
        // The bits have to be saved locally before writing to dest in case src and dest overlap.
        assert_eq!(size as usize as u64, size);
        let mut v = Vec::with_capacity(size as usize);
        for i in 0..size {
            let defined = self.get(src.alloc_id)?.undef_mask.get(src.offset + i);
            v.push(defined);
        }
        for (i, defined) in v.into_iter().enumerate() {
            self.get_mut(dest.alloc_id)?.undef_mask.set(
                dest.offset +
                    i as u64,
                defined,
            );
        }
        Ok(())
    }

    fn check_defined(&self, ptr: MemoryPointer, size: u64) -> EvalResult<'tcx> {
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
        size: u64,
        new_state: bool,
    ) -> EvalResult<'tcx> {
        if size == 0 {
            return Ok(());
        }
        let ptr = ptr.to_ptr()?;
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
// Methods to access integers in the target endianess
////////////////////////////////////////////////////////////////////////////////

fn write_target_uint(
    endianess: layout::Endian,
    mut target: &mut [u8],
    data: u128,
) -> Result<(), io::Error> {
    let len = target.len();
    match endianess {
        layout::Endian::Little => target.write_uint128::<LittleEndian>(data, len),
        layout::Endian::Big => target.write_uint128::<BigEndian>(data, len),
    }
}
fn write_target_int(
    endianess: layout::Endian,
    mut target: &mut [u8],
    data: i128,
) -> Result<(), io::Error> {
    let len = target.len();
    match endianess {
        layout::Endian::Little => target.write_int128::<LittleEndian>(data, len),
        layout::Endian::Big => target.write_int128::<BigEndian>(data, len),
    }
}

fn read_target_uint(endianess: layout::Endian, mut source: &[u8]) -> Result<u128, io::Error> {
    match endianess {
        layout::Endian::Little => source.read_uint128::<LittleEndian>(source.len()),
        layout::Endian::Big => source.read_uint128::<BigEndian>(source.len()),
    }
}

fn read_target_int(endianess: layout::Endian, mut source: &[u8]) -> Result<i128, io::Error> {
    match endianess {
        layout::Endian::Little => source.read_int128::<LittleEndian>(source.len()),
        layout::Endian::Big => source.read_int128::<BigEndian>(source.len()),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unaligned accesses
////////////////////////////////////////////////////////////////////////////////

pub trait HasMemory<'a, 'tcx: 'a, M: Machine<'tcx>> {
    fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx, M>;
    fn memory(&self) -> &Memory<'a, 'tcx, M>;

    /// Convert the value into a pointer (or a pointer-sized integer).  If the value is a ByRef,
    /// this may have to perform a load.
    fn into_ptr(
        &self,
        value: Value,
    ) -> EvalResult<'tcx, Pointer> {
        Ok(match value {
            Value::ByRef(ptr, align) => {
                self.memory().read_ptr_sized_unsigned(ptr.to_ptr()?, align)?
            }
            Value::ByVal(ptr) |
            Value::ByValPair(ptr, _) => ptr,
        }.into())
    }

    fn into_ptr_vtable_pair(
        &self,
        value: Value,
    ) -> EvalResult<'tcx, (Pointer, MemoryPointer)> {
        match value {
            Value::ByRef(ref_ptr, align) => {
                let mem = self.memory();
                let ptr = mem.read_ptr_sized_unsigned(ref_ptr.to_ptr()?, align)?.into();
                let vtable = mem.read_ptr_sized_unsigned(
                    ref_ptr.offset(mem.pointer_size(), &mem.tcx.data_layout)?.to_ptr()?,
                    align
                )?.to_ptr()?;
                Ok((ptr, vtable))
            }

            Value::ByValPair(ptr, vtable) => Ok((ptr.into(), vtable.to_ptr()?)),

            Value::ByVal(PrimVal::Undef) => err!(ReadUndefBytes),
            _ => bug!("expected ptr and vtable, got {:?}", value),
        }
    }

    fn into_slice(
        &self,
        value: Value,
    ) -> EvalResult<'tcx, (Pointer, u64)> {
        match value {
            Value::ByRef(ref_ptr, align) => {
                let mem = self.memory();
                let ptr = mem.read_ptr_sized_unsigned(ref_ptr.to_ptr()?, align)?.into();
                let len = mem.read_ptr_sized_unsigned(
                    ref_ptr.offset(mem.pointer_size(), &mem.tcx.data_layout)?.to_ptr()?,
                    align
                )?.to_bytes()? as u64;
                Ok((ptr, len))
            }
            Value::ByValPair(ptr, val) => {
                let len = val.to_u128()?;
                assert_eq!(len as u64 as u128, len);
                Ok((ptr.into(), len as u64))
            }
            Value::ByVal(PrimVal::Undef) => err!(ReadUndefBytes),
            Value::ByVal(_) => bug!("expected ptr and length, got {:?}", value),
        }
    }
}

impl<'a, 'tcx, M: Machine<'tcx>> HasMemory<'a, 'tcx, M> for Memory<'a, 'tcx, M> {
    #[inline]
    fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx, M> {
        self
    }

    #[inline]
    fn memory(&self) -> &Memory<'a, 'tcx, M> {
        self
    }
}

impl<'a, 'tcx, M: Machine<'tcx>> HasMemory<'a, 'tcx, M> for EvalContext<'a, 'tcx, M> {
    #[inline]
    fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx, M> {
        &mut self.memory
    }

    #[inline]
    fn memory(&self) -> &Memory<'a, 'tcx, M> {
        &self.memory
    }
}

impl<'a, 'tcx, M: Machine<'tcx>> layout::HasDataLayout for &'a Memory<'a, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}
