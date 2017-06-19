use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian, BigEndian};
use std::collections::{btree_map, BTreeMap, HashMap, HashSet, VecDeque};
use std::{fmt, iter, ptr, mem, io, ops};

use rustc::ty;
use rustc::ty::layout::{self, TargetDataLayout, HasDataLayout};
use syntax::ast::Mutability;
use rustc::middle::region::CodeExtent;

use error::{EvalError, EvalResult};
use value::{PrimVal, Pointer};
use eval_context::EvalContext;

////////////////////////////////////////////////////////////////////////////////
// Locks
////////////////////////////////////////////////////////////////////////////////

mod range {
    use super::*;

    // The derived `Ord` impl sorts first by the first field, then, if the fields are the same,
    // by the second field.
    // This is exactly what we need for our purposes, since a range query on a BTReeSet/BTreeMap will give us all
    // `MemoryRange`s whose `start` is <= than the one we're looking for, but not > the end of the range we're checking.
    // At the same time the `end` is irrelevant for the sorting and range searching, but used for the check.
    // This kind of search breaks, if `end < start`, so don't do that!
    #[derive(Eq, PartialEq, Ord, PartialOrd, Debug)]
    pub struct MemoryRange {
        start: u64,
        end: u64,
    }

    impl MemoryRange {
        pub fn new(offset: u64, len: u64) -> MemoryRange {
            assert!(len > 0);
            MemoryRange {
                start: offset,
                end: offset + len,
            }
        }

        pub fn range(offset: u64, len: u64) -> ops::Range<MemoryRange> {
            assert!(len > 0);
            // We select all elements that are within
            // the range given by the offset into the allocation and the length.
            // This is sound if "self.contains() || self.overlaps() == true" implies that self is in-range.
            let left = MemoryRange {
                start: 0,
                end: offset,
            };
            let right = MemoryRange {
                start: offset + len + 1,
                end: 0,
            };
            left..right
        }

        pub fn contains(&self, offset: u64, len: u64) -> bool {
            assert!(len > 0);
            self.start <= offset && (offset + len) <= self.end
        }

        pub fn overlaps(&self, offset: u64, len: u64) -> bool {
            assert!(len > 0);
            //let non_overlap = (offset + len) <= self.start || self.end <= offset;
            (offset + len) > self.start && self.end > offset
        }
    }
}
use self::range::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DynamicLifetime {
    frame: usize,
    region: Option<CodeExtent>, // "None" indicates "until the function ends"
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LockStatus {
    Held,
    RecoverAfter(CodeExtent), // the frame is given by the surrounding LockInfo's lifetime.
}

/// Information about a lock that is or will be held.
#[derive(Copy, Clone, Debug)]
pub struct LockInfo {
    kind: AccessKind,
    lifetime: DynamicLifetime,
    status: LockStatus,
}

impl LockInfo {
    fn access_permitted(&self, frame: usize, access: AccessKind) -> bool {
        use self::AccessKind::*;
        match (self.kind, access) {
            (Read, Read) => true, // Read access to read-locked region is okay, no matter who's holding the read lock.
            (Write, _) if self.lifetime.frame == frame => true, // All access is okay when we hold the write lock.
            _ => false, // Somebody else holding the write lock is not okay
        }
    }
}

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
    pub mutable: Mutability,
    /// Use the `mark_static_initalized` method of `Memory` to ensure that an error occurs, if the memory of this
    /// allocation is modified or deallocated in the future.
    /// Helps guarantee that stack allocations aren't deallocated via `rust_deallocate`
    pub kind: Kind,
    /// Memory regions that are locked by some function
    locks: BTreeMap<MemoryRange, Vec<LockInfo>>,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Kind {
    /// Error if deallocated any other way than `rust_deallocate`
    Rust,
    /// Error if deallocated any other way than `free`
    C,
    /// Error if deallocated except during a stack pop
    Stack,
    /// Static in the process of being initialized.
    /// The difference is important: An immutable static referring to a
    /// mutable initialized static will freeze immutably and would not
    /// be able to distinguish already initialized statics from uninitialized ones
    UninitializedStatic,
    /// May never be deallocated
    Static,
    /// Part of env var emulation
    Env,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MemoryPointer {
    pub alloc_id: AllocId,
    pub offset: u64,
}

impl<'tcx> MemoryPointer {
    pub fn new(alloc_id: AllocId, offset: u64) -> Self {
        MemoryPointer { alloc_id, offset }
    }

    pub(crate) fn wrapping_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> Self {
        MemoryPointer::new(self.alloc_id, cx.data_layout().wrapping_signed_offset(self.offset, i))
    }

    pub(crate) fn overflowing_signed_offset<C: HasDataLayout>(self, i: i128, cx: C) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_signed_offset(self.offset, i);
        (MemoryPointer::new(self.alloc_id, res), over)
    }

    pub(crate) fn signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        Ok(MemoryPointer::new(self.alloc_id, cx.data_layout().signed_offset(self.offset, i)?))
    }

    pub(crate) fn overflowing_offset<C: HasDataLayout>(self, i: u64, cx: C) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_offset(self.offset, i);
        (MemoryPointer::new(self.alloc_id, res), over)
    }

    pub(crate) fn offset<C: HasDataLayout>(self, i: u64, cx: C) -> EvalResult<'tcx, Self> {
        Ok(MemoryPointer::new(self.alloc_id, cx.data_layout().offset(self.offset, i)?))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Top-level interpreter memory
////////////////////////////////////////////////////////////////////////////////

pub type TlsKey = usize;

#[derive(Copy, Clone, Debug)]
pub struct TlsEntry<'tcx> {
    data: Pointer, // Will eventually become a map from thread IDs to `Pointer`s, if we ever support more than one thread.
    dtor: Option<ty::Instance<'tcx>>,
}

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

    /// A cache for basic byte allocations keyed by their contents. This is used to deduplicate
    /// allocations for string and bytestring literals.
    literal_alloc_cache: HashMap<Vec<u8>, AllocId>,

    /// pthreads-style thread-local storage.
    thread_local: BTreeMap<TlsKey, TlsEntry<'tcx>>,

    /// The Key to use for the next thread-local allocation.
    next_thread_local: TlsKey,

    /// To avoid having to pass flags to every single memory access, we have some global state saying whether
    /// alignment checking is currently enforced for read and/or write accesses.
    reads_are_aligned: bool,
    writes_are_aligned: bool,

    /// The current stack frame.  Used to check accesses against locks.
    cur_frame: usize,
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
            static_alloc: HashSet::new(),
            literal_alloc_cache: HashMap::new(),
            thread_local: BTreeMap::new(),
            next_thread_local: 0,
            reads_are_aligned: true,
            writes_are_aligned: true,
            cur_frame: usize::max_value(),
        }
    }

    pub fn allocations(&self) -> ::std::collections::hash_map::Iter<AllocId, Allocation> {
        self.alloc_map.iter()
    }

    pub fn create_fn_alloc(&mut self, instance: ty::Instance<'tcx>) -> MemoryPointer {
        if let Some(&alloc_id) = self.function_alloc_cache.get(&instance) {
            return MemoryPointer::new(alloc_id, 0);
        }
        let id = self.next_id;
        debug!("creating fn ptr: {}", id);
        self.next_id.0 += 1;
        self.functions.insert(id, instance);
        self.function_alloc_cache.insert(instance, id);
        MemoryPointer::new(id, 0)
    }

    pub fn allocate_cached(&mut self, bytes: &[u8]) -> EvalResult<'tcx, MemoryPointer> {
        if let Some(&alloc_id) = self.literal_alloc_cache.get(bytes) {
            return Ok(MemoryPointer::new(alloc_id, 0));
        }

        let ptr = self.allocate(bytes.len() as u64, 1, Kind::UninitializedStatic)?;
        self.write_bytes(ptr.into(), bytes)?;
        self.mark_static_initalized(ptr.alloc_id, Mutability::Immutable)?;
        self.literal_alloc_cache.insert(bytes.to_vec(), ptr.alloc_id);
        Ok(ptr)
    }

    pub fn allocate(&mut self, size: u64, align: u64, kind: Kind) -> EvalResult<'tcx, MemoryPointer> {
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
            kind,
            mutable: Mutability::Mutable,
            locks: BTreeMap::new(),
        };
        let id = self.next_id;
        self.next_id.0 += 1;
        self.alloc_map.insert(id, alloc);
        Ok(MemoryPointer::new(id, 0))
    }

    pub fn reallocate(&mut self, ptr: MemoryPointer, old_size: u64, old_align: u64, new_size: u64, new_align: u64, kind: Kind) -> EvalResult<'tcx, MemoryPointer> {
        use std::cmp::min;

        if ptr.offset != 0 {
            return Err(EvalError::ReallocateNonBasePtr);
        }
        if let Ok(alloc) = self.get(ptr.alloc_id) {
            if alloc.kind != kind {
                return Err(EvalError::ReallocatedWrongMemoryKind(alloc.kind, kind));
            }
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc"
        let new_ptr = self.allocate(new_size, new_align, kind)?;
        self.copy(ptr.into(), new_ptr.into(), min(old_size, new_size), min(old_align, new_align), /*nonoverlapping*/true)?;
        self.deallocate(ptr, Some((old_size, old_align)), kind)?;

        Ok(new_ptr)
    }

    pub fn deallocate(&mut self, ptr: MemoryPointer, size_and_align: Option<(u64, u64)>, kind: Kind) -> EvalResult<'tcx> {
        if ptr.offset != 0 {
            return Err(EvalError::DeallocateNonBasePtr);
        }

        let alloc = match self.alloc_map.remove(&ptr.alloc_id) {
            Some(alloc) => alloc,
            None => return Err(EvalError::DoubleFree),
        };

        if alloc.kind != kind {
            return Err(EvalError::DeallocatedWrongMemoryKind(alloc.kind, kind));
        }
        if !alloc.locks.is_empty() {
            return Err(EvalError::DeallocatedLockedMemory);
        }
        if let Some((size, align)) = size_and_align {
            if size != alloc.bytes.len() as u64 || align != alloc.align {
                return Err(EvalError::IncorrectAllocationInformation);
            }
        }

        self.memory_usage -= alloc.bytes.len() as u64;
        debug!("deallocated : {}", ptr.alloc_id);

        Ok(())
    }

    pub fn pointer_size(&self) -> u64 {
        self.layout.pointer_size.bytes()
    }

    pub fn endianess(&self) -> layout::Endian {
        self.layout.endian
    }

    /// Check that the pointer is aligned and non-NULL
    pub fn check_align(&self, ptr: Pointer, align: u64) -> EvalResult<'tcx> {
        let offset = match ptr.into_inner_primval() {
            PrimVal::Ptr(ptr) => {
                let alloc = self.get(ptr.alloc_id)?;
                if alloc.align < align {
                    return Err(EvalError::AlignmentCheckFailed {
                        has: alloc.align,
                        required: align,
                    });
                }
                ptr.offset
            },
            PrimVal::Bytes(bytes) => {
                let v = ((bytes as u128) % (1 << self.pointer_size())) as u64;
                if v == 0 {
                    return Err(EvalError::InvalidNullPointerUsage);
                }
                v
            },
            PrimVal::Undef => return Err(EvalError::ReadUndefBytes),
        };
        if offset % align == 0 {
            Ok(())
        } else {
            Err(EvalError::AlignmentCheckFailed {
                has: offset % align,
                required: align,
            })
        }
    }

    pub(crate) fn check_bounds(&self, ptr: MemoryPointer, access: bool) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        let allocation_size = alloc.bytes.len() as u64;
        if ptr.offset > allocation_size {
            return Err(EvalError::PointerOutOfBounds { ptr, access, allocation_size });
        }
        Ok(())
    }

    pub(crate) fn set_cur_frame(&mut self, cur_frame: usize) {
        self.cur_frame = cur_frame;
    }

    pub(crate) fn create_tls_key(&mut self, dtor: Option<ty::Instance<'tcx>>) -> TlsKey {
        let new_key = self.next_thread_local;
        self.next_thread_local += 1;
        self.thread_local.insert(new_key, TlsEntry { data: Pointer::null(), dtor });
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

    pub(crate) fn load_tls(&mut self, key: TlsKey) -> EvalResult<'tcx, Pointer> {
        return match self.thread_local.get(&key) {
            Some(&TlsEntry { data, .. }) => {
                trace!("TLS key {} loaded: {:?}", key, data);
                Ok(data)
            },
            None => Err(EvalError::TlsOutOfBounds)
        }
    }

    pub(crate) fn store_tls(&mut self, key: TlsKey, new_data: Pointer) -> EvalResult<'tcx> {
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
    pub(crate) fn fetch_tls_dtor(&mut self, key: Option<TlsKey>) -> EvalResult<'tcx, Option<(ty::Instance<'tcx>, Pointer, TlsKey)>> {
        use std::collections::Bound::*;
        let start = match key {
            Some(key) => Excluded(key),
            None => Unbounded,
        };
        for (&key, &mut TlsEntry { ref mut data, dtor }) in self.thread_local.range_mut((start, Unbounded)) {
            if !data.is_null()? {
                if let Some(dtor) = dtor {
                    let ret = Some((dtor, *data, key));
                    *data = Pointer::null();
                    return Ok(ret);
                }
            }
        }
        return Ok(None);
    }
}

/// Locking
impl<'a, 'tcx> Memory<'a, 'tcx> {
    pub(crate) fn check_locks(&self, ptr: MemoryPointer, len: u64, access: AccessKind) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        for (range, locks) in alloc.locks.range(MemoryRange::range(ptr.offset, len)) {
            for lock in locks {
                // Check if the lock is active, overlaps this access, and is in conflict with the access.
                if lock.status == LockStatus::Held  && range.overlaps(ptr.offset, len) && !lock.access_permitted(self.cur_frame, access) {
                    return Err(EvalError::MemoryLockViolation { ptr, len, access, lock: *lock });
                }
            }
        }
        Ok(())
    }

    /// Acquire the lock for the given lifetime
    pub(crate) fn acquire_lock(&mut self, ptr: MemoryPointer, len: u64, region: Option<CodeExtent>, kind: AccessKind) -> EvalResult<'tcx> {
        self.check_bounds(ptr.offset(len, self.layout)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        self.check_locks(ptr, len, kind)?; // make sure we have the access we are acquiring
        let lifetime = DynamicLifetime { frame: self.cur_frame, region };
        let alloc = self.get_mut(ptr.alloc_id)?;
        alloc.locks.entry(MemoryRange::new(ptr.offset, len)).or_insert_with(|| Vec::new()).push(LockInfo { lifetime, kind, status: LockStatus::Held });
        Ok(())
    }

    /// Release a lock prematurely
    pub(crate) fn release_lock_until(&mut self, ptr: MemoryPointer, len: u64, release_until: Option<CodeExtent>) -> EvalResult<'tcx> {
        // Make sure there are no read locks and no *other* write locks here
        if let Err(_) = self.check_locks(ptr, len, AccessKind::Write) {
            return Err(EvalError::InvalidMemoryLockRelease { ptr, len });
        }
        let cur_frame = self.cur_frame;
        let alloc = self.get_mut(ptr.alloc_id)?;
        {
            let lock_infos = alloc.locks.get_mut(&MemoryRange::new(ptr.offset, len)).ok_or(EvalError::InvalidMemoryLockRelease { ptr, len })?;
            let lock_info = match lock_infos.len() {
                0 => return Err(EvalError::InvalidMemoryLockRelease { ptr, len }),
                1 => &mut lock_infos[0],
                _ => bug!("There can not be overlapping locks when write access is possible."),
            };
            assert_eq!(lock_info.lifetime.frame, cur_frame);
            if let Some(ce) = release_until {
                lock_info.status = LockStatus::RecoverAfter(ce);
                return Ok(());
            }
        }
        // Falling through to here means we want to entirely remove the lock.  The control-flow is somewhat weird because of lexical lifetimes.
        alloc.locks.remove(&MemoryRange::new(ptr.offset, len));
        Ok(())
    }

    pub(crate) fn locks_lifetime_ended(&mut self, ending_region: Option<CodeExtent>) {
        let cur_frame = self.cur_frame;
        let has_ended =  |lock: &LockInfo| -> bool {
            if lock.lifetime.frame != cur_frame {
                return false;
            }
            match ending_region {
                None => true, // When a function ends, we end *all* its locks. It's okay for a function to still have lifetime-related locks
                              // when it returns, that can happen e.g. with NLL when a lifetime can, but does not have to, extend beyond the
                              // end of a function.
                Some(ending_region) => lock.lifetime.region == Some(ending_region),
            }
        };

        for alloc in self.alloc_map.values_mut() {
            for (_range, locks) in alloc.locks.iter_mut() {
                // Delete everything that ends now -- i.e., keep only all the other lifeimes.
                locks.retain(|lock| !has_ended(lock));
                // Activate locks that get recovered now
                if let Some(ending_region) = ending_region {
                    for lock in locks.iter_mut() {
                        if lock.lifetime.frame == cur_frame && lock.status == LockStatus::RecoverAfter(ending_region) {
                            lock.status = LockStatus::Held;
                        }
                    }
                }
            }
        }
        // TODO: It may happen now that we leave empty vectors in the map.  Is it worth getting rid of them?
    }
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
            Some(alloc) => if alloc.mutable == Mutability::Mutable {
                Ok(alloc)
            } else {
                Err(EvalError::ModifiedConstantMemory)
            },
            None => match self.functions.get(&id) {
                Some(_) => Err(EvalError::DerefFunctionPointer),
                None => Err(EvalError::DanglingPointerDeref),
            }
        }
    }

    pub fn get_fn(&self, ptr: MemoryPointer) -> EvalResult<'tcx, ty::Instance<'tcx>> {
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

            let immutable = match (alloc.kind, alloc.mutable) {
                (Kind::UninitializedStatic, _) => " (static in the process of initialization)",
                (Kind::Static, Mutability::Mutable) => " (static mut)",
                (Kind::Static, Mutability::Immutable) => " (immutable)",
                (Kind::Env, _) => " (env var)",
                (Kind::C, _) => " (malloc)",
                (Kind::Rust, _) => " (heap)",
                (Kind::Stack, _) => " (stack)",
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
                if val.kind != Kind::Static {
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
    fn get_bytes_unchecked(&self, ptr: MemoryPointer, size: u64, align: u64) -> EvalResult<'tcx, &[u8]> {
        // Zero-sized accesses can use dangling pointers, but they still have to be aligned and non-NULL
        if self.reads_are_aligned {
            self.check_align(ptr.into(), align)?;
        }
        if size == 0 {
            return Ok(&[]);
        }
        self.check_locks(ptr, size, AccessKind::Read)?;
        self.check_bounds(ptr.offset(size, self)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        assert_eq!(size as usize as u64, size);
        let offset = ptr.offset as usize;
        Ok(&alloc.bytes[offset..offset + size as usize])
    }

    fn get_bytes_unchecked_mut(&mut self, ptr: MemoryPointer, size: u64, align: u64) -> EvalResult<'tcx, &mut [u8]> {
        // Zero-sized accesses can use dangling pointers, but they still have to be aligned and non-NULL
        if self.writes_are_aligned {
            self.check_align(ptr.into(), align)?;
        }
        if size == 0 {
            return Ok(&mut []);
        }
        self.check_locks(ptr, size, AccessKind::Write)?;
        self.check_bounds(ptr.offset(size, self.layout)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        let alloc = self.get_mut(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        assert_eq!(size as usize as u64, size);
        let offset = ptr.offset as usize;
        Ok(&mut alloc.bytes[offset..offset + size as usize])
    }

    fn get_bytes(&self, ptr: MemoryPointer, size: u64, align: u64) -> EvalResult<'tcx, &[u8]> {
        assert_ne!(size, 0);
        if self.relocations(ptr, size)?.count() != 0 {
            return Err(EvalError::ReadPointerAsBytes);
        }
        self.check_defined(ptr, size)?;
        self.get_bytes_unchecked(ptr, size, align)
    }

    fn get_bytes_mut(&mut self, ptr: MemoryPointer, size: u64, align: u64) -> EvalResult<'tcx, &mut [u8]> {
        assert_ne!(size, 0);
        self.clear_relocations(ptr, size)?;
        self.mark_definedness(ptr.into(), size, true)?;
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
    pub fn mark_inner_allocation(&mut self, alloc: AllocId, mutability: Mutability) -> EvalResult<'tcx> {
        // relocations into other statics are not "inner allocations"
        if !self.static_alloc.contains(&alloc) {
            self.mark_static_initalized(alloc, mutability)?;
        }
        Ok(())
    }

    /// mark an allocation as static and initialized, either mutable or not
    pub fn mark_static_initalized(&mut self, alloc_id: AllocId, mutability: Mutability) -> EvalResult<'tcx> {
        trace!("mark_static_initalized {:?}, mutability: {:?}", alloc_id, mutability);
        // do not use `self.get_mut(alloc_id)` here, because we might have already marked a
        // sub-element or have circular pointers (e.g. `Rc`-cycles)
        let relocations = match self.alloc_map.get_mut(&alloc_id) {
            Some(&mut Allocation { ref mut relocations, ref mut kind, ref mut mutable, .. }) => {
                match *kind {
                    // const eval results can refer to "locals".
                    // E.g. `const Foo: &u32 = &1;` refers to the temp local that stores the `1`
                    Kind::Stack |
                    // The entire point of this function
                    Kind::UninitializedStatic |
                    // In the future const eval will allow heap allocations so we'll need to protect them
                    // from deallocation, too
                    Kind::Rust |
                    Kind::C => {},
                    Kind::Static => {
                        trace!("mark_static_initalized: skipping already initialized static referred to by static currently being initialized");
                        return Ok(());
                    },
                    // FIXME: This could be allowed, but not for env vars set during miri execution
                    Kind::Env => return Err(EvalError::Unimplemented("statics can't refer to env vars".to_owned())),
                }
                *kind = Kind::Static;
                *mutable = mutability;
                // take out the relocations vector to free the borrow on self, so we can call
                // mark recursively
                mem::replace(relocations, Default::default())
            },
            None if !self.functions.contains_key(&alloc_id) => return Err(EvalError::DanglingPointerDeref),
            _ => return Ok(()),
        };
        // recurse into inner allocations
        for &alloc in relocations.values() {
            self.mark_inner_allocation(alloc, mutability)?;
        }
        // put back the relocations
        self.alloc_map.get_mut(&alloc_id).expect("checked above").relocations = relocations;
        Ok(())
    }

    pub fn copy(&mut self, src: Pointer, dest: Pointer, size: u64, align: u64, nonoverlapping: bool) -> EvalResult<'tcx> {
        if size == 0 {
            // Empty accesses don't need to be valid pointers, but they should still be aligned
            if self.reads_are_aligned {
                self.check_align(src, align)?;
            }
            if self.writes_are_aligned {
                self.check_align(dest, align)?;
            }
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
                if nonoverlapping {
                    if (src.offset <= dest.offset && src.offset + size > dest.offset) ||
                       (dest.offset <= src.offset && dest.offset + size > src.offset) {
                        return Err(EvalError::Intrinsic(format!("copy_nonoverlapping called on overlapping ranges")));
                    }
                }
                ptr::copy(src_bytes, dest_bytes, size as usize);
            } else {
                ptr::copy_nonoverlapping(src_bytes, dest_bytes, size as usize);
            }
        }

        self.copy_undef_mask(src, dest, size)?;
        self.copy_relocations(src, dest, size)?;

        Ok(())
    }

    pub fn read_c_str(&self, ptr: MemoryPointer) -> EvalResult<'tcx, &[u8]> {
        let alloc = self.get(ptr.alloc_id)?;
        assert_eq!(ptr.offset as usize as u64, ptr.offset);
        let offset = ptr.offset as usize;
        match alloc.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                if self.relocations(ptr, (size + 1) as u64)?.count() != 0 {
                    return Err(EvalError::ReadPointerAsBytes);
                }
                self.check_defined(ptr, (size + 1) as u64)?;
                self.check_locks(ptr, (size + 1) as u64, AccessKind::Read)?;
                Ok(&alloc.bytes[offset..offset + size])
            },
            None => Err(EvalError::UnterminatedCString(ptr)),
        }
    }

    pub fn read_bytes(&self, ptr: Pointer, size: u64) -> EvalResult<'tcx, &[u8]> {
        if size == 0 {
            // Empty accesses don't need to be valid pointers, but they should still be non-NULL
            if self.reads_are_aligned {
                self.check_align(ptr, 1)?;
            }
            return Ok(&[]);
        }
        self.get_bytes(ptr.to_ptr()?, size, 1)
    }

    pub fn write_bytes(&mut self, ptr: Pointer, src: &[u8]) -> EvalResult<'tcx> {
        if src.is_empty() {
            // Empty accesses don't need to be valid pointers, but they should still be non-NULL
            if self.writes_are_aligned {
                self.check_align(ptr, 1)?;
            }
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, src.len() as u64, 1)?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    pub fn write_repeat(&mut self, ptr: Pointer, val: u8, count: u64) -> EvalResult<'tcx> {
        if count == 0 {
            // Empty accesses don't need to be valid pointers, but they should still be non-NULL
            if self.writes_are_aligned {
                self.check_align(ptr, 1)?;
            }
            return Ok(());
        }
        let bytes = self.get_bytes_mut(ptr.to_ptr()?, count, 1)?;
        for b in bytes { *b = val; }
        Ok(())
    }

    pub fn read_ptr(&self, ptr: MemoryPointer) -> EvalResult<'tcx, Pointer> {
        let size = self.pointer_size();
        self.check_relocation_edges(ptr, size)?; // Make sure we don't read part of a pointer as a pointer
        let endianess = self.endianess();
        let bytes = self.get_bytes_unchecked(ptr, size, size)?;
        // Undef check happens *after* we established that the alignment is correct.
        // We must not return Ok() for unaligned pointers!
        if self.check_defined(ptr, size).is_err() {
            return Ok(PrimVal::Undef.into());
        }
        let offset = read_target_uint(endianess, bytes).unwrap();
        assert_eq!(offset as u64 as u128, offset);
        let offset = offset as u64;
        let alloc = self.get(ptr.alloc_id)?;
        match alloc.relocations.get(&ptr.offset) {
            Some(&alloc_id) => Ok(PrimVal::Ptr(MemoryPointer::new(alloc_id, offset)).into()),
            None => Ok(PrimVal::Bytes(offset as u128).into()),
        }
    }

    pub fn write_ptr(&mut self, dest: MemoryPointer, ptr: MemoryPointer) -> EvalResult<'tcx> {
        self.write_usize(dest, ptr.offset as u64)?;
        self.get_mut(dest.alloc_id)?.relocations.insert(dest.offset, ptr.alloc_id);
        Ok(())
    }

    pub fn write_primval(
        &mut self,
        dest: Pointer,
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

    pub fn read_bool(&self, ptr: MemoryPointer) -> EvalResult<'tcx, bool> {
        let bytes = self.get_bytes(ptr, 1, self.layout.i1_align.abi())?;
        match bytes[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(EvalError::InvalidBool),
        }
    }

    pub fn write_bool(&mut self, ptr: MemoryPointer, b: bool) -> EvalResult<'tcx> {
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

    pub fn read_int(&self, ptr: MemoryPointer, size: u64) -> EvalResult<'tcx, i128> {
        let align = self.int_align(size)?;
        self.get_bytes(ptr, size, align).map(|b| read_target_int(self.endianess(), b).unwrap())
    }

    pub fn write_int(&mut self, ptr: MemoryPointer, n: i128, size: u64) -> EvalResult<'tcx> {
        let align = self.int_align(size)?;
        let endianess = self.endianess();
        let b = self.get_bytes_mut(ptr, size, align)?;
        write_target_int(endianess, b, n).unwrap();
        Ok(())
    }

    pub fn read_uint(&self, ptr: MemoryPointer, size: u64) -> EvalResult<'tcx, u128> {
        let align = self.int_align(size)?;
        self.get_bytes(ptr, size, align).map(|b| read_target_uint(self.endianess(), b).unwrap())
    }

    pub fn write_uint(&mut self, ptr: MemoryPointer, n: u128, size: u64) -> EvalResult<'tcx> {
        let align = self.int_align(size)?;
        let endianess = self.endianess();
        let b = self.get_bytes_mut(ptr, size, align)?;
        write_target_uint(endianess, b, n).unwrap();
        Ok(())
    }

    pub fn read_isize(&self, ptr: MemoryPointer) -> EvalResult<'tcx, i64> {
        self.read_int(ptr, self.pointer_size()).map(|i| i as i64)
    }

    pub fn write_isize(&mut self, ptr: MemoryPointer, n: i64) -> EvalResult<'tcx> {
        let size = self.pointer_size();
        self.write_int(ptr, n as i128, size)
    }

    pub fn read_usize(&self, ptr: MemoryPointer) -> EvalResult<'tcx, u64> {
        self.read_uint(ptr, self.pointer_size()).map(|i| i as u64)
    }

    pub fn write_usize(&mut self, ptr: MemoryPointer, n: u64) -> EvalResult<'tcx> {
        let size = self.pointer_size();
        self.write_uint(ptr, n as u128, size)
    }

    pub fn write_f32(&mut self, ptr: MemoryPointer, f: f32) -> EvalResult<'tcx> {
        let endianess = self.endianess();
        let align = self.layout.f32_align.abi();
        let b = self.get_bytes_mut(ptr, 4, align)?;
        write_target_f32(endianess, b, f).unwrap();
        Ok(())
    }

    pub fn write_f64(&mut self, ptr: MemoryPointer, f: f64) -> EvalResult<'tcx> {
        let endianess = self.endianess();
        let align = self.layout.f64_align.abi();
        let b = self.get_bytes_mut(ptr, 8, align)?;
        write_target_f64(endianess, b, f).unwrap();
        Ok(())
    }

    pub fn read_f32(&self, ptr: MemoryPointer) -> EvalResult<'tcx, f32> {
        self.get_bytes(ptr, 4, self.layout.f32_align.abi())
            .map(|b| read_target_f32(self.endianess(), b).unwrap())
    }

    pub fn read_f64(&self, ptr: MemoryPointer) -> EvalResult<'tcx, f64> {
        self.get_bytes(ptr, 8, self.layout.f64_align.abi())
            .map(|b| read_target_f64(self.endianess(), b).unwrap())
    }
}

/// Relocations
impl<'a, 'tcx> Memory<'a, 'tcx> {
    fn relocations(&self, ptr: MemoryPointer, size: u64)
        -> EvalResult<'tcx, btree_map::Range<u64, AllocId>>
    {
        let start = ptr.offset.saturating_sub(self.pointer_size() - 1);
        let end = ptr.offset + size;
        Ok(self.get(ptr.alloc_id)?.relocations.range(start..end))
    }

    fn clear_relocations(&mut self, ptr: MemoryPointer, size: u64) -> EvalResult<'tcx> {
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

    fn check_relocation_edges(&self, ptr: MemoryPointer, size: u64) -> EvalResult<'tcx> {
        let overlapping_start = self.relocations(ptr, 0)?.count();
        let overlapping_end = self.relocations(ptr.offset(size, self.layout)?, 0)?.count();
        if overlapping_start + overlapping_end != 0 {
            return Err(EvalError::ReadPointerAsBytes);
        }
        Ok(())
    }

    fn copy_relocations(&mut self, src: MemoryPointer, dest: MemoryPointer, size: u64) -> EvalResult<'tcx> {
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
    fn copy_undef_mask(&mut self, src: MemoryPointer, dest: MemoryPointer, size: u64) -> EvalResult<'tcx> {
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

    fn check_defined(&self, ptr: MemoryPointer, size: u64) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        if !alloc.undef_mask.is_range_defined(ptr.offset, ptr.offset + size) {
            return Err(EvalError::ReadUndefBytes);
        }
        Ok(())
    }

    pub fn mark_definedness(
        &mut self,
        ptr: Pointer,
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
}

fn bit_index(bits: u64) -> (usize, usize) {
    let a = bits / BLOCK_SIZE;
    let b = bits % BLOCK_SIZE;
    assert_eq!(a as usize as u64, a);
    assert_eq!(b as usize as u64, b);
    (a as usize, b as usize)
}

////////////////////////////////////////////////////////////////////////////////
// Unaligned accesses
////////////////////////////////////////////////////////////////////////////////

pub(crate) trait HasMemory<'a, 'tcx> {
    fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx>;
    fn memory(&self) -> &Memory<'a, 'tcx>;

    // These are not supposed to be overriden.
    fn read_maybe_aligned<F, T>(&mut self, aligned: bool, f: F) -> EvalResult<'tcx, T>
        where F: FnOnce(&mut Self) -> EvalResult<'tcx, T>
    {
        assert!(self.memory_mut().reads_are_aligned, "Unaligned reads must not be nested");
        self.memory_mut().reads_are_aligned = aligned;
        let t = f(self);
        self.memory_mut().reads_are_aligned = true;
        t
    }

    fn write_maybe_aligned<F, T>(&mut self, aligned: bool, f: F) -> EvalResult<'tcx, T>
        where F: FnOnce(&mut Self) -> EvalResult<'tcx, T>
    {
        assert!(self.memory_mut().writes_are_aligned, "Unaligned writes must not be nested");
        self.memory_mut().writes_are_aligned = aligned;
        let t = f(self);
        self.memory_mut().writes_are_aligned = true;
        t
    }
}

impl<'a, 'tcx> HasMemory<'a, 'tcx> for Memory<'a, 'tcx> {
    #[inline]
    fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx> {
        self
    }

    #[inline]
    fn memory(&self) -> &Memory<'a, 'tcx> {
        self
    }
}

impl<'a, 'tcx> HasMemory<'a, 'tcx> for EvalContext<'a, 'tcx> {
    #[inline]
    fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx> {
        &mut self.memory
    }

    #[inline]
    fn memory(&self) -> &Memory<'a, 'tcx> {
        &self.memory
    }
}

////////////////////////////////////////////////////////////////////////////////
// Pointer arithmetic
////////////////////////////////////////////////////////////////////////////////

pub trait PointerArithmetic : layout::HasDataLayout {
    // These are not supposed to be overriden.

    //// Trunace the given value to the pointer size; also return whether there was an overflow
    fn truncate_to_ptr(self, val: u128) -> (u64, bool) {
        let max_ptr_plus_1 = 1u128 << self.data_layout().pointer_size.bits();
        ((val % max_ptr_plus_1) as u64, val >= max_ptr_plus_1)
    }

    // Overflow checking only works properly on the range from -u64 to +u64.
    fn overflowing_signed_offset(self, val: u64, i: i128) -> (u64, bool) {
        // FIXME: is it possible to over/underflow here?
        if i < 0 {
            // trickery to ensure that i64::min_value() works fine
            // this formula only works for true negative values, it panics for zero!
            let n = u64::max_value() - (i as u64) + 1;
            val.overflowing_sub(n)
        } else {
            self.overflowing_offset(val, i as u64)
        }
    }

    fn overflowing_offset(self, val: u64, i: u64) -> (u64, bool) {
        let (res, over1) = val.overflowing_add(i);
        let (res, over2) = self.truncate_to_ptr(res as u128);
        (res, over1 || over2)
    }

    fn signed_offset<'tcx>(self, val: u64, i: i64) -> EvalResult<'tcx, u64> {
        let (res, over) = self.overflowing_signed_offset(val, i as i128);
        if over {
            Err(EvalError::OverflowingMath)
        } else {
            Ok(res)
        }
    }

    fn offset<'tcx>(self, val: u64, i: u64) -> EvalResult<'tcx, u64> {
        let (res, over) = self.overflowing_offset(val, i);
        if over {
            Err(EvalError::OverflowingMath)
        } else {
            Ok(res)
        }
    }

    fn wrapping_signed_offset(self, val: u64, i: i64) -> u64 {
        self.overflowing_signed_offset(val, i as i128).0
    }
}

impl<T: layout::HasDataLayout> PointerArithmetic for T {}

impl<'a, 'tcx> layout::HasDataLayout for &'a Memory<'a, 'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        self.layout
    }
}
impl<'a, 'tcx> layout::HasDataLayout for &'a EvalContext<'a, 'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        self.memory().layout
    }
}

impl<'c, 'b, 'a, 'tcx> layout::HasDataLayout for &'c &'b mut EvalContext<'a, 'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        self.memory().layout
    }
}
