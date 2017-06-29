use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian, BigEndian};
use std::collections::{btree_map, BTreeMap, HashMap, HashSet, VecDeque, BTreeSet};
use std::{fmt, iter, ptr, mem, io};

use rustc::ty;
use rustc::ty::layout::{self, TargetDataLayout};

use error::{EvalError, EvalResult};
use value::{PrimVal, self};

////////////////////////////////////////////////////////////////////////////////
// Allocations and pointers
////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocId(pub u64);

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct Allocation {
    /// The actual bytes of the allocation.
    /// Note that the bytes of a pointer represent the offset of the pointer
    pub bytes: Vec<u8>,
    /// Maps from byte addresses to allocations.
    /// Only the first byte of a pointer is inserted into the map.
    pub relocations: BTreeMap<u64, AllocId>,
    /// Denotes undefined memory. Reading from undefined memory is forbidden in miri
    pub undef_mask: UndefMask,
    /// The alignment of the allocation to detect unaligned reads.
    pub align: u64,
    /// Whether the allocation may be modified.
    /// Use the `mark_static_initalized` method of `Memory` to ensure that an error occurs, if the memory of this
    /// allocation is modified or deallocated in the future.
    pub static_kind: StaticKind,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum StaticKind {
    /// may be deallocated without breaking miri's invariants
    NotStatic,
    /// may be modified, but never deallocated
    Mutable,
    /// may neither be modified nor deallocated
    Immutable,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Pointer {
    pub alloc_id: AllocId,
    pub offset: u64,
}

impl Pointer {
    pub fn new(alloc_id: AllocId, offset: u64) -> Self {
        Pointer { alloc_id, offset }
    }

    pub fn wrapping_signed_offset<'tcx>(self, i: i64, layout: &TargetDataLayout) -> Self {
        Pointer::new(self.alloc_id, value::wrapping_signed_offset(self.offset, i, layout))
    }

    pub fn overflowing_signed_offset<'tcx>(self, i: i128, layout: &TargetDataLayout) -> (Self, bool) {
        let (res, over) = value::overflowing_signed_offset(self.offset, i, layout);
        (Pointer::new(self.alloc_id, res), over)
    }

    pub fn signed_offset<'tcx>(self, i: i64, layout: &TargetDataLayout) -> EvalResult<'tcx, Self> {
        Ok(Pointer::new(self.alloc_id, value::signed_offset(self.offset, i, layout)?))
    }

    pub fn overflowing_offset<'tcx>(self, i: u64, layout: &TargetDataLayout) -> (Self, bool) {
        let (res, over) = value::overflowing_offset(self.offset, i, layout);
        (Pointer::new(self.alloc_id, res), over)
    }

    pub fn offset<'tcx>(self, i: u64, layout: &TargetDataLayout) -> EvalResult<'tcx, Self> {
        Ok(Pointer::new(self.alloc_id, value::offset(self.offset, i, layout)?))
    }
}

pub type TlsKey = usize;

#[derive(Copy, Clone, Debug)]
pub struct TlsEntry<'tcx> {
    data: PrimVal, // Will eventually become a map from thread IDs to `PrimVal`s, if we ever support more than one thread.
    dtor: Option<ty::Instance<'tcx>>,
}

////////////////////////////////////////////////////////////////////////////////
// Top-level interpreter memory
////////////////////////////////////////////////////////////////////////////////

pub struct Memory<'a, 'tcx> {
    /// Actual memory allocations (arbitrary bytes, may contain pointers into other allocations).
    alloc_map: HashMap<AllocId, Allocation>,

    /// The AllocId to assign to the next new allocation. Always incremented, never gets smaller.
    next_id: AllocId,

    /// Set of statics, constants, promoteds, vtables, ... to prevent `mark_static_initalized` from
    /// stepping out of its own allocations. This set only contains statics backed by an
    /// allocation. If they are ByVal or ByValPair they are not here, but will be inserted once
    /// they become ByRef.
    static_alloc: HashSet<AllocId>,

    /// Number of virtual bytes allocated.
    memory_usage: u64,

    /// Maximum number of virtual bytes that may be allocated.
    memory_size: u64,

    /// Function "allocations". They exist solely so pointers have something to point to, and
    /// we can figure out what they point to.
    functions: HashMap<AllocId, ty::Instance<'tcx>>,

    /// Inverse map of `functions` so we don't allocate a new pointer every time we need one
    function_alloc_cache: HashMap<ty::Instance<'tcx>, AllocId>,

    /// Target machine data layout to emulate.
    pub layout: &'a TargetDataLayout,

    /// List of memory regions containing packed structures.
    ///
    /// We mark memory as "packed" or "unaligned" for a single statement, and clear the marking
    /// afterwards. In the case where no packed structs are present, it's just a single emptyness
    /// check of a set instead of heavily influencing all memory access code as other solutions
    /// would. This is simpler than the alternative of passing a "packed" parameter to every
    /// load/store method.
    ///
    /// One disadvantage of this solution is the fact that you can cast a pointer to a packed
    /// struct to a pointer to a normal struct and if you access a field of both in the same MIR
    /// statement, the normal struct access will succeed even though it shouldn't. But even with
    /// mir optimizations, that situation is hard/impossible to produce.
    packed: BTreeSet<Entry>,

    /// A cache for basic byte allocations keyed by their contents. This is used to deduplicate
    /// allocations for string and bytestring literals.
    literal_alloc_cache: HashMap<Vec<u8>, AllocId>,

    /// pthreads-style thread-local storage.
    thread_local: BTreeMap<TlsKey, TlsEntry<'tcx>>,

    /// The Key to use for the next thread-local allocation.
    next_thread_local: TlsKey,
}

impl<'a, 'tcx> Memory<'a, 'tcx> {
    pub fn new(layout: &'a TargetDataLayout, max_memory: u64) -> Self {
        Memory {
            alloc_map: HashMap::new(),
            functions: HashMap::new(),
            function_alloc_cache: HashMap::new(),
            next_id: AllocId(0),
            layout,
            memory_size: max_memory,
            memory_usage: 0,
            packed: BTreeSet::new(),
            static_alloc: HashSet::new(),
            literal_alloc_cache: HashMap::new(),
            thread_local: BTreeMap::new(),
            next_thread_local: 0,
        }
    }

    pub fn allocations(&self) -> ::std::collections::hash_map::Iter<AllocId, Allocation> {
        self.alloc_map.iter()
    }

    pub fn create_fn_alloc(&mut self, instance: ty::Instance<'tcx>) -> Pointer {
        if let Some(&alloc_id) = self.function_alloc_cache.get(&instance) {
            return Pointer::new(alloc_id, 0);
        }
        let id = self.next_id;
        debug!("creating fn ptr: {}", id);
        self.next_id.0 += 1;
        self.functions.insert(id, instance);
        self.function_alloc_cache.insert(instance, id);
        Pointer::new(id, 0)
    }

    pub fn allocate_cached(&mut self, bytes: &[u8]) -> EvalResult<'tcx, Pointer> {
        if let Some(&alloc_id) = self.literal_alloc_cache.get(bytes) {
            return Ok(Pointer::new(alloc_id, 0));
        }

        let ptr = self.allocate(bytes.len() as u64, 1)?;
        self.write_bytes(ptr, bytes)?;
        self.mark_static_initalized(ptr.alloc_id, false)?;
        self.literal_alloc_cache.insert(bytes.to_vec(), ptr.alloc_id);
        Ok(ptr)
    }

    pub fn allocate(&mut self, size: u64, align: u64) -> EvalResult<'tcx, Pointer> {
        assert_ne!(align, 0);
        assert!(align.is_power_of_two());

        if self.memory_size - self.memory_usage < size {
            return Err(EvalError::OutOfMemory {
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
            static_kind: StaticKind::NotStatic,
        };
        let id = self.next_id;
        self.next_id.0 += 1;
        self.alloc_map.insert(id, alloc);
        Ok(Pointer::new(id, 0))
    }

    // TODO(solson): Track which allocations were returned from __rust_allocate and report an error
    // when reallocating/deallocating any others.
    pub fn reallocate(&mut self, ptr: Pointer, new_size: u64, align: u64) -> EvalResult<'tcx, Pointer> {
        assert!(align.is_power_of_two());
        // TODO(solson): Report error about non-__rust_allocate'd pointer.
        if ptr.offset != 0 {
            return Err(EvalError::Unimplemented(format!("bad pointer offset: {}", ptr.offset)));
        }
        if self.get(ptr.alloc_id).ok().map_or(false, |alloc| alloc.static_kind != StaticKind::NotStatic) {
            return Err(EvalError::ReallocatedStaticMemory);
        }

        let size = self.get(ptr.alloc_id)?.bytes.len() as u64;

        if new_size > size {
            let amount = new_size - size;
            self.memory_usage += amount;
            let alloc = self.get_mut(ptr.alloc_id)?;
            // FIXME: check alignment here
            assert_eq!(amount as usize as u64, amount);
            alloc.bytes.extend(iter::repeat(0).take(amount as usize));
            alloc.undef_mask.grow(amount, false);
        } else if size > new_size {
            self.memory_usage -= size - new_size;
            self.clear_relocations(ptr.offset(new_size, self.layout)?, size - new_size)?;
            let alloc = self.get_mut(ptr.alloc_id)?;
            // FIXME: check alignment here
            // `as usize` is fine here, since it is smaller than `size`, which came from a usize
            alloc.bytes.truncate(new_size as usize);
            alloc.bytes.shrink_to_fit();
            alloc.undef_mask.truncate(new_size);
        }

        Ok(Pointer::new(ptr.alloc_id, 0))
    }

    // TODO(solson): See comment on `reallocate`.
    pub fn deallocate(&mut self, ptr: Pointer) -> EvalResult<'tcx> {
        if ptr.offset != 0 {
            // TODO(solson): Report error about non-__rust_allocate'd pointer.
            return Err(EvalError::Unimplemented(format!("bad pointer offset: {}", ptr.offset)));
        }
        if self.get(ptr.alloc_id).ok().map_or(false, |alloc| alloc.static_kind != StaticKind::NotStatic) {
            return Err(EvalError::DeallocatedStaticMemory);
        }

        if let Some(alloc) = self.alloc_map.remove(&ptr.alloc_id) {
            self.memory_usage -= alloc.bytes.len() as u64;
        } else {
            debug!("deallocated a pointer twice: {}", ptr.alloc_id);
            // TODO(solson): Report error about erroneous free. This is blocked on properly tracking
            // already-dropped state since this if-statement is entered even in safe code without
            // it.
        }
        debug!("deallocated : {}", ptr.alloc_id);

        Ok(())
    }

    pub fn pointer_size(&self) -> u64 {
        self.layout.pointer_size.bytes()
    }

    pub fn endianess(&self) -> layout::Endian {
        self.layout.endian
    }

    pub fn check_align(&self, ptr: Pointer, align: u64, len: u64) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        // check whether the memory was marked as packed
        // we select all elements that have the correct alloc_id and are within
        // the range given by the offset into the allocation and the length
        let start = Entry {
            alloc_id: ptr.alloc_id,
            packed_start: 0,
            packed_end: ptr.offset + len,
        };
        let end = Entry {
            alloc_id: ptr.alloc_id,
            packed_start: ptr.offset + len,
            packed_end: 0,
        };
        for &Entry { packed_start, packed_end, .. } in self.packed.range(start..end) {
            // if the region we are checking is covered by a region in `packed`
            // ignore the actual alignment
            if packed_start <= ptr.offset && (ptr.offset + len) <= packed_end {
                return Ok(());
            }
        }
        if alloc.align < align {
            return Err(EvalError::AlignmentCheckFailed {
                has: alloc.align,
                required: align,
            });
        }
        if ptr.offset % align == 0 {
            Ok(())
        } else {
            Err(EvalError::AlignmentCheckFailed {
                has: ptr.offset % align,
                required: align,
            })
        }
    }

    pub(crate) fn check_bounds(&self, ptr: Pointer, access: bool) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        let allocation_size = alloc.bytes.len() as u64;
        if ptr.offset > allocation_size {
            return Err(EvalError::PointerOutOfBounds { ptr, access, allocation_size });
        }
        Ok(())
    }

    pub(crate) fn mark_packed(&mut self, ptr: Pointer, len: u64) {
        self.packed.insert(Entry {
            alloc_id: ptr.alloc_id,
            packed_start: ptr.offset,
            packed_end: ptr.offset + len,
        });
    }

    pub(crate) fn clear_packed(&mut self) {
        self.packed.clear();
    }

    pub(crate) fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>) -> TlsKey {
        let new_key = self.next_thread_local;
        self.next_thread_local += 1;
        self.thread_local.insert(new_key, TlsEntry { data: PrimVal::Bytes(0), dtor });
        trace!("New TLS key allocated: {} with dtor {:?}", new_key, dtor);
        return new_key;
    }

    pub(crate) fn delete_tls_key(&mut self, key: TlsKey) -> EvalResult<'tcx> {
        return match self.thread_local.remove(&key) {
            Some(_) => {
                trace!("TLS key {} removed", key);
                Ok(())
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }

    pub(crate) fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, PrimVal> {
        return match self.thread_local.get(&key) {
            Some(&TlsEntry { data, .. }) => {
                trace!("TLS key {} loaded: {:?}", key, data);
                Ok(data)
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }

    pub(crate) fn store_tls(&mut self, key: TlsKey, new_data: PrimVal) -> EvalResult<'tcx> {
        return match self.thread_local.get_mut(&key) {
            Some(&mut TlsEntry { ref mut data, .. }) => {
                trace!("TLS key {} stored: {:?}", key, new_data);
                *data = new_data;
                Ok(())
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }
    
    /// Returns a dtor, its argument and its index, if one is supposed to run
    ///
    /// An optional destructor function may be associated with each key value.
    /// At thread exit, if a key value has a non-NULL destructor pointer,
    /// and the thread has a non-NULL value associated with that key,
    /// the value of the key is set to NULL, and then the function pointed
    /// to is called with the previously associated value as its sole argument.
    /// The order of destructor calls is unspecified if more than one destructor
    /// exists for a thread when it exits.
    ///
    /// If, after all the destructors have been called for all non-NULL values
    /// with associated destructors, there are still some non-NULL values with
    /// associated destructors, then the process is repeated.
    /// If, after at least {PTHREAD_DESTRUCTOR_ITERATIONS} iterations of destructor
    /// calls for outstanding non-NULL values, there are still some non-NULL values
    /// with associated destructors, implementations may stop calling destructors,
    /// or they may continue calling destructors until no non-NULL values with
    /// associated destructors exist, even though this might result in an infinite loop.
    pub(crate) fn fetch_tls_dtor(&mut self, key: Option<TlsKey>) -> EvalResult<'tcx, Option<(ty::Instance<'tcx>, PrimVal, TlsKey)>> {
        use std::collections::Bound::*;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        for (&key, &mut TlsEntry { ref mut data, dtor }) in self.thread_local.range_mut((start, Unbounded)) {
            if !data.is_null()? {
                if let Some(dtor) = dtor {
                    let ret = Some((dtor, *data, key));
                    *data = PrimVal::Bytes(0);
                    return Ok(ret);
                }
            }
        }
        return Ok(None);
    }
}

// The derived `Ord` impl sorts first by the first field, then, if the fields are the same
// by the second field, and if those are the same, too, then by the third field.
// This is exactly what we need for our purposes, since a range within an allocation
// will give us all `Entry`s that have that `AllocId`, and whose `packed_start` is <= than
// the one we're looking for, but not > the end of the range we're checking.
// At the same time the `packed_end` is irrelevant for the sorting and range searching, but used for the check.
// This kind of search breaks, if `packed_end < packed_start`, so don't do that!
#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Entry {
    alloc_id: AllocId,
    packed_start: u64,
    packed_end: u64,
}

/// Allocation accessors
impl<'a, 'tcx> Memory<'a, 'tcx> {
    pub fn get(&self, id: AllocId) -> EvalResult<'tcx, &Allocation> {
        match self.alloc_map.get(&id) {
            Some(alloc) => Ok(alloc),
            None => match self.functions.get(&id) {
                Some(_) => Err(EvalError::DerefFunctionPointer),
                None => Err(EvalError::DanglingPointerDeref),
            }
        }
    }

    pub fn get_mut(&mut self, id: AllocId) -> EvalResult<'tcx, &mut Allocation> {
        match self.alloc_map.get_mut(&id) {
            Some(alloc) => match alloc.static_kind {
                StaticKind::Mutable |
                StaticKind::NotStatic => Ok(alloc),
                StaticKind::Immutable => Err(EvalError::ModifiedConstantMemory),
            },
            None => match self.functions.get(&id) {
                Some(_) => Err(EvalError::DerefFunctionPointer),
                None => Err(EvalError::DanglingPointerDeref),
            }
        }
    }

    pub fn get_fn(&self, ptr: Pointer) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        if ptr.offset != 0 {
            return Err(EvalError::InvalidFunctionPointer);
        }
        debug!("reading fn ptr: {}", ptr.alloc_id);
        match self.functions.get(&ptr.alloc_id) {
            Some(&fndef) => Ok(fndef),
            None => match self.alloc_map.get(&ptr.alloc_id) {
                Some(_) => Err(EvalError::ExecuteMemory),
                None => Err(EvalError::InvalidFunctionPointer),
            }
        }
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

            let alloc = match (self.alloc_map.get(&id), self.functions.get(&id)) {
                (Some(a), None) => a,
                (None, Some(instance)) => {
                    trace!("{} {}", msg, instance);
                    continue;
                },
                (None, None) => {
                    trace!("{} (deallocated)", msg);
                    continue;
                },
                (Some(_), Some(_)) => bug!("miri invariant broken: an allocation id exists that points to both a function and a memory location"),
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

            let immutable = match alloc.static_kind {
                StaticKind::Mutable => " (static mut)",
                StaticKind::Immutable => " (immutable)",
                StaticKind::NotStatic => "",
            };
            trace!("{}({} bytes, alignment {}){}", msg, alloc.bytes.len(), alloc.align, immutable);

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
        let leaks: Vec<_> = self.alloc_map
            .iter()
            .filter_map(|(&key, val)| {
                if val.static_kind == StaticKind::NotStatic {
                    Some(key)
                } else {
                    None
                }
            })
            .collect();
        let n = leaks.len();
        self.dump_allocs(leaks);
        n
    }
}

/// Byte accessors
impl<'a, 'tcx> Memory<'a, 'tcx> {
    fn get_bytes_unchecked(&self, ptr: Pointer, size: u64, align: u64) -> EvalResult<'tcx, &[u8]> {
        if size == 0 {
            return Ok(&[]);
        }
        self.check_align(ptr, align, size)?;
        self.check_bounds(ptr.offset(size, self.layout)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        assert_eq!(size as usize as u64, size);
        let offset = ptr.offset as usize;
        Ok(&alloc.bytes[offset..offset + size as usize])
    }

    fn get_bytes_unchecked_mut(&mut self, ptr: Pointer, size: u64, align: u64) -> EvalResult<'tcx, &mut [u8]> {
        if size == 0 {
            return Ok(&mut []);
        }
        self.check_align(ptr, align, size)?;
        self.check_bounds(ptr.offset(size, self.layout)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        let alloc = self.get_mut(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        assert_eq!(size as usize as u64, size);
        let offset = ptr.offset as usize;
        Ok(&mut alloc.bytes[offset..offset + size as usize])
    }

    fn get_bytes(&self, ptr: Pointer, size: u64, align: u64) -> EvalResult<'tcx, &[u8]> {
        assert_ne!(size, 0);
        if self.relocations(ptr, size)?.count() != 0 {
            return Err(EvalError::ReadPointerAsBytes);
        }
        self.check_defined(ptr, size)?;
        self.get_bytes_unchecked(ptr, size, align)
    }

    fn get_bytes_mut(&mut self, ptr: Pointer, size: u64, align: u64) -> EvalResult<'tcx, &mut [u8]> {
        assert_ne!(size, 0);
        self.clear_relocations(ptr, size)?;
        self.mark_definedness(PrimVal::Ptr(ptr), size, true)?;
        self.get_bytes_unchecked_mut(ptr, size, align)
    }
}

/// Reading and writing
impl<'a, 'tcx> Memory<'a, 'tcx> {
    /// mark an allocation as being the entry point to a static (see `static_alloc` field)
    pub fn mark_static(&mut self, alloc_id: AllocId) {
        trace!("mark_static: {:?}", alloc_id);
        if !self.static_alloc.insert(alloc_id) {
            bug!("tried to mark an allocation ({:?}) as static twice", alloc_id);
        }
    }

    /// mark an allocation pointed to by a static as static and initialized
    pub fn mark_inner_allocation(&mut self, alloc: AllocId, mutable: bool) -> EvalResult<'tcx> {
        // relocations into other statics are not "inner allocations"
        if !self.static_alloc.contains(&alloc) {
            self.mark_static_initalized(alloc, mutable)?;
        }
        Ok(())
    }

    /// mark an allocation as static and initialized, either mutable or not
    pub fn mark_static_initalized(&mut self, alloc_id: AllocId, mutable: bool) -> EvalResult<'tcx> {
        trace!("mark_static_initialized {:?}, mutable: {:?}", alloc_id, mutable);
        // do not use `self.get_mut(alloc_id)` here, because we might have already marked a
        // sub-element or have circular pointers (e.g. `Rc`-cycles)
        let relocations = match self.alloc_map.get_mut(&alloc_id) {
            Some(&mut Allocation { ref mut relocations, static_kind: ref mut kind @ StaticKind::NotStatic, .. }) => {
                *kind = if mutable {
                    StaticKind::Mutable
                } else {
                    StaticKind::Immutable
                };
                // take out the relocations vector to free the borrow on self, so we can call
                // mark recursively
                mem::replace(relocations, Default::default())
            },
            None if !self.functions.contains_key(&alloc_id) => return Err(EvalError::DanglingPointerDeref),
            _ => return Ok(()),
        };
        // recurse into inner allocations
        for &alloc in relocations.values() {
            self.mark_inner_allocation(alloc, mutable)?;
        }
        // put back the relocations
        self.alloc_map.get_mut(&alloc_id).expect("checked above").relocations = relocations;
        Ok(())
    }

    pub fn copy(&mut self, src: PrimVal, dest: PrimVal, size: u64, align: u64) -> EvalResult<'tcx> {
        if size == 0 {
            return Ok(());
        }
        let src = src.to_ptr()?;
        let dest = dest.to_ptr()?;
        self.check_relocation_edges(src, size)?;

        let src_bytes = self.get_bytes_unchecked(src, size, align)?.as_ptr();
        let dest_bytes = self.get_bytes_mut(dest, size, align)?.as_mut_ptr();

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
        unsafe {
            assert_eq!(size as usize as u64, size);
            if src.alloc_id == dest.alloc_id {
                ptr::copy(src_bytes, dest_bytes, size as usize);
            } else {
                ptr::copy_nonoverlapping(src_bytes, dest_bytes, size as usize);
            }
        }

        self.copy_undef_mask(src, dest, size)?;
        self.copy_relocations(src, dest, size)?;

        Ok(())
    }

    pub fn read_c_str(&self, ptr: Pointer) -> EvalResult<'tcx, &[u8]> {
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        let offset = ptr.offset as usize;
        match alloc.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                if self.relocations(ptr, (size + 1) as u64)?.count() != 0 {
                    return Err(EvalError::ReadPointerAsBytes);
                }
                self.check_defined(ptr, (size + 1) as u64)?;
                Ok(&alloc.bytes[offset..offset + size])
            },
            None => Err(EvalError::UnterminatedCString(ptr)),
        }
    }

    pub fn read_bytes(&self, ptr: PrimVal, size: u64) -> EvalResult<'tcx, &[u8]> {
        if size == 0 {
            return Ok(&[]);
        }
        self.get_bytes(ptr.to_ptr()?, size, 1)
    }

    pub fn write_bytes(&mut self, ptr: Pointer, src: &[u8]) -> EvalResult<'tcx> {
        if src.is_empty() {
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr, src.len() as u64, 1)?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    pub fn write_repeat(&mut self, ptr: Pointer, val: u8, count: u64) -> EvalResult<'tcx> {
        if count == 0 {
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr, count, 1)?;
        for b in bytes { *b = val; }
        Ok(())
    }

    pub fn read_ptr(&self, ptr: Pointer) -> EvalResult<'tcx, PrimVal> {
        let size = self.pointer_size();
        if self.check_defined(ptr, size).is_err() {
            return Ok(PrimVal::Undef);
        }
        let endianess = self.endianess();
        let bytes = self.get_bytes_unchecked(ptr, size, size)?;
        let offset = read_target_uint(endianess, bytes).unwrap();
        assert_eq!(offset as u64 as u128, offset);
        let offset = offset as u64;
        let alloc = self.get(ptr.alloc_id)?;
        match alloc.relocations.get(&ptr.offset) {
            Some(&alloc_id) => Ok(PrimVal::Ptr(Pointer::new(alloc_id, offset))),
            None => Ok(PrimVal::Bytes(offset as u128)),
        }
    }

    pub fn write_ptr(&mut self, dest: Pointer, ptr: Pointer) -> EvalResult<'tcx> {
        self.write_usize(dest, ptr.offset as u64)?;
        self.get_mut(dest.alloc_id)?.relocations.insert(dest.offset, ptr.alloc_id);
        Ok(())
    }

    pub fn write_primval(
        &mut self,
        dest: PrimVal,
        val: PrimVal,
        size: u64,
    ) -> EvalResult<'tcx> {
        match val {
            PrimVal::Ptr(ptr) => {
                assert_eq!(size, self.pointer_size());
                self.write_ptr(dest.to_ptr()?, ptr)
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
                self.write_uint(dest.to_ptr()?, bytes & mask, size)
            }

            PrimVal::Undef => self.mark_definedness(dest, size, false),
        }
    }

    pub fn read_bool(&self, ptr: Pointer) -> EvalResult<'tcx, bool> {
        let bytes = self.get_bytes(ptr, 1, self.layout.i1_align.abi())?;
        match bytes[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(EvalError::InvalidBool),
        }
    }

    pub fn write_bool(&mut self, ptr: Pointer, b: bool) -> EvalResult<'tcx> {
        let align = self.layout.i1_align.abi();
        self.get_bytes_mut(ptr, 1, align)
            .map(|bytes| bytes[0] = b as u8)
    }

    fn int_align(&self, size: u64) -> EvalResult<'tcx, u64> {
        match size {
            1 => Ok(self.layout.i8_align.abi()),
            2 => Ok(self.layout.i16_align.abi()),
            4 => Ok(self.layout.i32_align.abi()),
            8 => Ok(self.layout.i64_align.abi()),
            16 => Ok(self.layout.i128_align.abi()),
            _ => bug!("bad integer size: {}", size),
        }
    }

    pub fn read_int(&self, ptr: Pointer, size: u64) -> EvalResult<'tcx, i128> {
        let align = self.int_align(size)?;
        self.get_bytes(ptr, size, align).map(|b| read_target_int(self.endianess(), b).unwrap())
    }

    pub fn write_int(&mut self, ptr: Pointer, n: i128, size: u64) -> EvalResult<'tcx> {
        let align = self.int_align(size)?;
        let endianess = self.endianess();
        let b = self.get_bytes_mut(ptr, size, align)?;
        write_target_int(endianess, b, n).unwrap();
        Ok(())
    }

    pub fn read_uint(&self, ptr: Pointer, size: u64) -> EvalResult<'tcx, u128> {
        let align = self.int_align(size)?;
        self.get_bytes(ptr, size, align).map(|b| read_target_uint(self.endianess(), b).unwrap())
    }

    pub fn write_uint(&mut self, ptr: Pointer, n: u128, size: u64) -> EvalResult<'tcx> {
        let align = self.int_align(size)?;
        let endianess = self.endianess();
        let b = self.get_bytes_mut(ptr, size, align)?;
        write_target_uint(endianess, b, n).unwrap();
        Ok(())
    }

    pub fn read_isize(&self, ptr: Pointer) -> EvalResult<'tcx, i64> {
        self.read_int(ptr, self.pointer_size()).map(|i| i as i64)
    }

    pub fn write_isize(&mut self, ptr: Pointer, n: i64) -> EvalResult<'tcx> {
        let size = self.pointer_size();
        self.write_int(ptr, n as i128, size)
    }

    pub fn read_usize(&self, ptr: Pointer) -> EvalResult<'tcx, u64> {
        self.read_uint(ptr, self.pointer_size()).map(|i| i as u64)
    }

    pub fn write_usize(&mut self, ptr: Pointer, n: u64) -> EvalResult<'tcx> {
        let size = self.pointer_size();
        self.write_uint(ptr, n as u128, size)
    }

    pub fn write_f32(&mut self, ptr: Pointer, f: f32) -> EvalResult<'tcx> {
        let endianess = self.endianess();
        let align = self.layout.f32_align.abi();
        let b = self.get_bytes_mut(ptr, 4, align)?;
        write_target_f32(endianess, b, f).unwrap();
        Ok(())
    }

    pub fn write_f64(&mut self, ptr: Pointer, f: f64) -> EvalResult<'tcx> {
        let endianess = self.endianess();
        let align = self.layout.f64_align.abi();
        let b = self.get_bytes_mut(ptr, 8, align)?;
        write_target_f64(endianess, b, f).unwrap();
        Ok(())
    }

    pub fn read_f32(&self, ptr: Pointer) -> EvalResult<'tcx, f32> {
        self.get_bytes(ptr, 4, self.layout.f32_align.abi())
            .map(|b| read_target_f32(self.endianess(), b).unwrap())
    }

    pub fn read_f64(&self, ptr: Pointer) -> EvalResult<'tcx, f64> {
        self.get_bytes(ptr, 8, self.layout.f64_align.abi())
            .map(|b| read_target_f64(self.endianess(), b).unwrap())
    }
}

/// Relocations
impl<'a, 'tcx> Memory<'a, 'tcx> {
    fn relocations(&self, ptr: Pointer, size: u64)
        -> EvalResult<'tcx, btree_map::Range<u64, AllocId>>
    {
        let start = ptr.offset.saturating_sub(self.pointer_size() - 1);
        let end = ptr.offset + size;
        Ok(self.get(ptr.alloc_id)?.relocations.range(start..end))
    }

    fn clear_relocations(&mut self, ptr: Pointer, size: u64) -> EvalResult<'tcx> {
        // Find all relocations overlapping the given range.
        let keys: Vec<_> = self.relocations(ptr, size)?.map(|(&k, _)| k).collect();
        if keys.is_empty() { return Ok(()); }

        // Find the start and end of the given range and its outermost relocations.
        let start = ptr.offset;
        let end = start + size;
        let first = *keys.first().unwrap();
        let last = *keys.last().unwrap() + self.pointer_size();

        let alloc = self.get_mut(ptr.alloc_id)?;

        // Mark parts of the outermost relocations as undefined if they partially fall outside the
        // given range.
        if first < start { alloc.undef_mask.set_range(first, start, false); }
        if last > end { alloc.undef_mask.set_range(end, last, false); }

        // Forget all the relocations.
        for k in keys { alloc.relocations.remove(&k); }

        Ok(())
    }

    fn check_relocation_edges(&self, ptr: Pointer, size: u64) -> EvalResult<'tcx> {
        let overlapping_start = self.relocations(ptr, 0)?.count();
        let overlapping_end = self.relocations(ptr.offset(size, self.layout)?, 0)?.count();
        if overlapping_start + overlapping_end != 0 {
            return Err(EvalError::ReadPointerAsBytes);
        }
        Ok(())
    }

    fn copy_relocations(&mut self, src: Pointer, dest: Pointer, size: u64) -> EvalResult<'tcx> {
        let relocations: Vec<_> = self.relocations(src, size)?
            .map(|(&offset, &alloc_id)| {
                // Update relocation offsets for the new positions in the destination allocation.
                (offset + dest.offset - src.offset, alloc_id)
            })
            .collect();
        self.get_mut(dest.alloc_id)?.relocations.extend(relocations);
        Ok(())
    }
}

/// Undefined bytes
impl<'a, 'tcx> Memory<'a, 'tcx> {
    // FIXME(solson): This is a very naive, slow version.
    fn copy_undef_mask(&mut self, src: Pointer, dest: Pointer, size: u64) -> EvalResult<'tcx> {
        // The bits have to be saved locally before writing to dest in case src and dest overlap.
        assert_eq!(size as usize as u64, size);
        let mut v = Vec::with_capacity(size as usize);
        for i in 0..size {
            let defined = self.get(src.alloc_id)?.undef_mask.get(src.offset + i);
            v.push(defined);
        }
        for (i, defined) in v.into_iter().enumerate() {
            self.get_mut(dest.alloc_id)?.undef_mask.set(dest.offset + i as u64, defined);
        }
        Ok(())
    }

    fn check_defined(&self, ptr: Pointer, size: u64) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        if !alloc.undef_mask.is_range_defined(ptr.offset, ptr.offset + size) {
            return Err(EvalError::ReadUndefBytes);
        }
        Ok(())
    }

    pub fn mark_definedness(
        &mut self,
        ptr: PrimVal,
        size: u64,
        new_state: bool
    ) -> EvalResult<'tcx> {
        if size == 0 {
            return Ok(())
        }
        let ptr = ptr.to_ptr()?;
        let mut alloc = self.get_mut(ptr.alloc_id)?;
        alloc.undef_mask.set_range(ptr.offset, ptr.offset + size, new_state);
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Methods to access integers in the target endianess
////////////////////////////////////////////////////////////////////////////////

fn write_target_uint(endianess: layout::Endian, mut target: &mut [u8], data: u128) -> Result<(), io::Error> {
    let len = target.len();
    match endianess {
        layout::Endian::Little => target.write_uint128::<LittleEndian>(data, len),
        layout::Endian::Big => target.write_uint128::<BigEndian>(data, len),
    }
}
fn write_target_int(endianess: layout::Endian, mut target: &mut [u8], data: i128) -> Result<(), io::Error> {
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
// Methods to access floats in the target endianess
////////////////////////////////////////////////////////////////////////////////

fn write_target_f32(endianess: layout::Endian, mut target: &mut [u8], data: f32) -> Result<(), io::Error> {
    match endianess {
        layout::Endian::Little => target.write_f32::<LittleEndian>(data),
        layout::Endian::Big => target.write_f32::<BigEndian>(data),
    }
}
fn write_target_f64(endianess: layout::Endian, mut target: &mut [u8], data: f64) -> Result<(), io::Error> {
    match endianess {
        layout::Endian::Little => target.write_f64::<LittleEndian>(data),
        layout::Endian::Big => target.write_f64::<BigEndian>(data),
    }
}

fn read_target_f32(endianess: layout::Endian, mut source: &[u8]) -> Result<f32, io::Error> {
    match endianess {
        layout::Endian::Little => source.read_f32::<LittleEndian>(),
        layout::Endian::Big => source.read_f32::<BigEndian>(),
    }
}
fn read_target_f64(endianess: layout::Endian, mut source: &[u8]) -> Result<f64, io::Error> {
    match endianess {
        layout::Endian::Little => source.read_f64::<LittleEndian>(),
        layout::Endian::Big => source.read_f64::<BigEndian>(),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Undefined byte tracking
////////////////////////////////////////////////////////////////////////////////

type Block = u64;
const BLOCK_SIZE: u64 = 64;

#[derive(Clone, Debug)]
pub struct UndefMask {
    blocks: Vec<Block>,
    len: u64,
}

impl UndefMask {
    fn new(size: u64) -> Self {
        let mut m = UndefMask {
            blocks: vec![],
            len: 0,
        };
        m.grow(size, false);
        m
    }

    /// Check whether the range `start..end` (end-exclusive) is entirely defined.
    pub fn is_range_defined(&self, start: u64, end: u64) -> bool {
        if end > self.len { return false; }
        for i in start..end {
            if !self.get(i) { return false; }
        }
        true
    }

    fn set_range(&mut self, start: u64, end: u64, new_state: bool) {
        let len = self.len;
        if end > len { self.grow(end - len, new_state); }
        self.set_range_inbounds(start, end, new_state);
    }

    fn set_range_inbounds(&mut self, start: u64, end: u64, new_state: bool) {
        for i in start..end { self.set(i, new_state); }
    }

    fn get(&self, i: u64) -> bool {
        let (block, bit) = bit_index(i);
        (self.blocks[block] & 1 << bit) != 0
    }

    fn set(&mut self, i: u64, new_state: bool) {
        let (block, bit) = bit_index(i);
        if new_state {
            self.blocks[block] |= 1 << bit;
        } else {
            self.blocks[block] &= !(1 << bit);
        }
    }

    fn grow(&mut self, amount: u64, new_state: bool) {
        let unused_trailing_bits = self.blocks.len() as u64 * BLOCK_SIZE - self.len;
        if amount > unused_trailing_bits {
            let additional_blocks = amount / BLOCK_SIZE + 1;
            assert_eq!(additional_blocks as usize as u64, additional_blocks);
            self.blocks.extend(iter::repeat(0).take(additional_blocks as usize));
        }
        let start = self.len;
        self.len += amount;
        self.set_range_inbounds(start, start + amount, new_state);
    }

    fn truncate(&mut self, length: u64) {
        self.len = length;
        let truncate = self.len / BLOCK_SIZE + 1;
        assert_eq!(truncate as usize as u64, truncate);
        self.blocks.truncate(truncate as usize);
        self.blocks.shrink_to_fit();
    }
}

fn bit_index(bits: u64) -> (usize, usize) {
    let a = bits / BLOCK_SIZE;
    let b = bits % BLOCK_SIZE;
    assert_eq!(a as usize as u64, a);
    assert_eq!(b as usize as u64, b);
    (a as usize, b as usize)
}
