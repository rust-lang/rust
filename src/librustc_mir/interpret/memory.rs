use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian, BigEndian};
use std::collections::{btree_map, BTreeMap, HashMap, HashSet, VecDeque};
use std::{fmt, iter, ptr, mem, io};
use std::cell::Cell;

use rustc::ty;
use rustc::ty::layout::{self, TargetDataLayout, HasDataLayout};
use syntax::ast::Mutability;
use rustc::middle::region::CodeExtent;

use super::{
    EvalResult, EvalErrorKind,
    PrimVal, Pointer,
    EvalContext, DynamicLifetime,
    Machine,
    RangeMap,
};

////////////////////////////////////////////////////////////////////////////////
// Locks
////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

/// Information about a lock that is currently held.
#[derive(Clone, Debug)]
struct LockInfo {
    suspended: Vec<SuspendedWriteLock>,
    active: Lock,
}

#[derive(Clone, Debug)]
struct SuspendedWriteLock  {
    /// Original lifetime of the lock that is now suspended
    lft: DynamicLifetime,
    /// Regions that all have to end to reenable this suspension
    suspensions: Vec<CodeExtent>,
}

#[derive(Clone, Debug)]
pub enum Lock {
    NoLock,
    WriteLock(DynamicLifetime),
    ReadLock(Vec<DynamicLifetime>), // This should never be empty -- that would be a read lock held and nobody there to release it...
}
use self::Lock::*;

impl Default for LockInfo {
    fn default() -> Self {
        LockInfo::new(NoLock)
    }
}

impl LockInfo {
    fn new(lock: Lock) -> LockInfo {
        LockInfo { suspended: Vec::new(), active: lock }
    }

    fn access_permitted(&self, frame: Option<usize>, access: AccessKind) -> bool {
        use self::AccessKind::*;
        match (&self.active, access) {
            (&NoLock, _) => true,
            (&ReadLock(ref lfts), Read) => {
                assert!(!lfts.is_empty(), "Someone left an empty read lock behind.");
                // Read access to read-locked region is okay, no matter who's holding the read lock.
                true
            },
            (&WriteLock(ref lft), _) => {
                // All access is okay if we are the ones holding it
                Some(lft.frame) == frame
            },
            _ => false, // Nothing else is okay.
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Allocations and pointers
////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AllocId(u64);

#[derive(Debug)]
enum AllocIdKind {
    /// We can't ever have more than `usize::max_value` functions at the same time
    /// since we never "deallocate" functions
    Function(usize),
    /// Locals and heap allocations (also statics for now, but those will get their
    /// own variant soonish).
    Runtime(u64),
}

impl AllocIdKind {
    fn into_alloc_id(self) -> AllocId {
        match self {
            AllocIdKind::Function(n) => AllocId(n as u64),
            AllocIdKind::Runtime(n) => AllocId((1 << 63) | n),
        }
    }
}

impl AllocId {
    /// Currently yields the top bit to discriminate the `AllocIdKind`s
    fn discriminant(self) -> u64 {
        self.0 >> 63
    }
    /// Yields everything but the discriminant bits
    fn index(self) -> u64 {
        self.0 & ((1 << 63) - 1)
    }
    fn destructure(self) -> AllocIdKind {
        match self.discriminant() {
            0 => AllocIdKind::Function(self.index() as usize),
            1 => AllocIdKind::Runtime(self.index()),
            n => bug!("got discriminant {} for AllocId", n),
        }
    }
}

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.destructure())
    }
}

impl fmt::Debug for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.destructure())
    }
}

#[derive(Debug)]
pub struct Allocation<M> {
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
    pub kind: Kind<M>,
    /// Memory regions that are locked by some function
    locks: RangeMap<LockInfo>,
}

impl<M> Allocation<M> {
    fn check_locks<'tcx>(&self, frame: Option<usize>, offset: u64, len: u64, access: AccessKind) -> Result<(), LockInfo> {
        if len == 0 {
            return Ok(())
        }
        for lock in self.locks.iter(offset, len) {
            // Check if the lock is in conflict with the access.
            if !lock.access_permitted(frame, access) {
                return Err(lock.clone());
            }
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Kind<T> {
    /// Error if deallocated except during a stack pop
    Stack,
    /// Static in the process of being initialized.
    /// The difference is important: An immutable static referring to a
    /// mutable initialized static will freeze immutably and would not
    /// be able to distinguish already initialized statics from uninitialized ones
    UninitializedStatic,
    /// May never be deallocated
    Static,
    /// Additional memory kinds a machine wishes to distinguish from the builtin ones
    Machine(T),
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

    pub fn overflowing_signed_offset<C: HasDataLayout>(self, i: i128, cx: C) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_signed_offset(self.offset, i);
        (MemoryPointer::new(self.alloc_id, res), over)
    }

    pub(crate) fn signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        Ok(MemoryPointer::new(self.alloc_id, cx.data_layout().signed_offset(self.offset, i)?))
    }

    pub fn overflowing_offset<C: HasDataLayout>(self, i: u64, cx: C) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_offset(self.offset, i);
        (MemoryPointer::new(self.alloc_id, res), over)
    }

    pub fn offset<C: HasDataLayout>(self, i: u64, cx: C) -> EvalResult<'tcx, Self> {
        Ok(MemoryPointer::new(self.alloc_id, cx.data_layout().offset(self.offset, i)?))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Top-level interpreter memory
////////////////////////////////////////////////////////////////////////////////

pub struct Memory<'a, 'tcx, M: Machine<'tcx>> {
    /// Additional data required by the Machine
    pub data: M::MemoryData,

    /// Actual memory allocations (arbitrary bytes, may contain pointers into other allocations).
    alloc_map: HashMap<u64, Allocation<M::MemoryKinds>>,

    /// The AllocId to assign to the next new regular allocation. Always incremented, never gets smaller.
    next_alloc_id: u64,

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
    functions: Vec<ty::Instance<'tcx>>,

    /// Inverse map of `functions` so we don't allocate a new pointer every time we need one
    function_alloc_cache: HashMap<ty::Instance<'tcx>, AllocId>,

    /// Target machine data layout to emulate.
    pub layout: &'a TargetDataLayout,

    /// A cache for basic byte allocations keyed by their contents. This is used to deduplicate
    /// allocations for string and bytestring literals.
    literal_alloc_cache: HashMap<Vec<u8>, AllocId>,

    /// To avoid having to pass flags to every single memory access, we have some global state saying whether
    /// alignment checking is currently enforced for read and/or write accesses.
    reads_are_aligned: Cell<bool>,
    writes_are_aligned: Cell<bool>,

    /// The current stack frame.  Used to check accesses against locks.
    cur_frame: usize,
}

impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    pub fn new(layout: &'a TargetDataLayout, max_memory: u64, data: M::MemoryData) -> Self {
        Memory {
            data,
            alloc_map: HashMap::new(),
            functions: Vec::new(),
            function_alloc_cache: HashMap::new(),
            next_alloc_id: 0,
            layout,
            memory_size: max_memory,
            memory_usage: 0,
            static_alloc: HashSet::new(),
            literal_alloc_cache: HashMap::new(),
            reads_are_aligned: Cell::new(true),
            writes_are_aligned: Cell::new(true),
            cur_frame: usize::max_value(),
        }
    }

    pub fn allocations<'x>(&'x self) -> impl Iterator<Item = (AllocId, &'x Allocation<M::MemoryKinds>)> {
        self.alloc_map.iter().map(|(&id, alloc)| (AllocIdKind::Runtime(id).into_alloc_id(), alloc))
    }

    pub fn create_fn_alloc(&mut self, instance: ty::Instance<'tcx>) -> MemoryPointer {
        if let Some(&alloc_id) = self.function_alloc_cache.get(&instance) {
            return MemoryPointer::new(alloc_id, 0);
        }
        let id = self.functions.len();
        debug!("creating fn ptr: {}", id);
        self.functions.push(instance);
        let alloc_id = AllocIdKind::Function(id).into_alloc_id();
        self.function_alloc_cache.insert(instance, alloc_id);
        MemoryPointer::new(alloc_id, 0)
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

    pub fn allocate(
        &mut self,
        size: u64,
        align: u64,
        kind: Kind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, MemoryPointer> {
        assert_ne!(align, 0);
        assert!(align.is_power_of_two());

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
            kind,
            mutable: Mutability::Mutable,
            locks: RangeMap::new(),
        };
        let id = self.next_alloc_id;
        self.next_alloc_id += 1;
        self.alloc_map.insert(id, alloc);
        Ok(MemoryPointer::new(AllocIdKind::Runtime(id).into_alloc_id(), 0))
    }

    pub fn reallocate(
        &mut self,
        ptr: MemoryPointer,
        old_size: u64,
        old_align: u64,
        new_size: u64,
        new_align: u64,
        kind: Kind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, MemoryPointer> {
        use std::cmp::min;

        if ptr.offset != 0 {
            return err!(ReallocateNonBasePtr);
        }
        if let Ok(alloc) = self.get(ptr.alloc_id) {
            if alloc.kind != kind {
                return err!(ReallocatedWrongMemoryKind(format!("{:?}", alloc.kind), format!("{:?}", kind)));
            }
        }

        // For simplicities' sake, we implement reallocate as "alloc, copy, dealloc"
        let new_ptr = self.allocate(new_size, new_align, kind)?;
        self.copy(ptr.into(), new_ptr.into(), min(old_size, new_size), min(old_align, new_align), /*nonoverlapping*/true)?;
        self.deallocate(ptr, Some((old_size, old_align)), kind)?;

        Ok(new_ptr)
    }

    pub fn deallocate(
        &mut self,
        ptr: MemoryPointer,
        size_and_align: Option<(u64, u64)>,
        kind: Kind<M::MemoryKinds>,
    ) -> EvalResult<'tcx> {
        if ptr.offset != 0 {
            return err!(DeallocateNonBasePtr);
        }

        let alloc_id = match ptr.alloc_id.destructure() {
            AllocIdKind::Function(_) =>
                return err!(DeallocatedWrongMemoryKind("function".to_string(), format!("{:?}", kind))),
            AllocIdKind::Runtime(id) => id,
        };

        let alloc = match self.alloc_map.remove(&alloc_id) {
            Some(alloc) => alloc,
            None => return err!(DoubleFree),
        };

        // It is okay for us to still holds locks on deallocation -- for example, we could store data we own
        // in a local, and the local could be deallocated (from StorageDead) before the function returns.
        // However, we should check *something*.  For now, we make sure that there is no conflicting write
        // lock by another frame.  We *have* to permit deallocation if we hold a read lock.
        // TODO: Figure out the exact rules here.
        alloc.check_locks(Some(self.cur_frame), 0, alloc.bytes.len() as u64, AccessKind::Read)
            .map_err(|lock| EvalErrorKind::DeallocatedLockedMemory { ptr, lock: lock.active })?;

        if alloc.kind != kind {
            return err!(DeallocatedWrongMemoryKind(format!("{:?}", alloc.kind), format!("{:?}", kind)));
        }
        if let Some((size, align)) = size_and_align {
            if size != alloc.bytes.len() as u64 || align != alloc.align {
                return err!(IncorrectAllocationInformation);
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

    /// Check that the pointer is aligned AND non-NULL.
    pub fn check_align(&self, ptr: Pointer, align: u64) -> EvalResult<'tcx> {
        let offset = match ptr.into_inner_primval() {
            PrimVal::Ptr(ptr) => {
                let alloc = self.get(ptr.alloc_id)?;
                if alloc.align < align {
                    return err!(AlignmentCheckFailed {
                        has: alloc.align,
                        required: align,
                    });
                }
                ptr.offset
            },
            PrimVal::Bytes(bytes) => {
                let v = ((bytes as u128) % (1 << self.pointer_size())) as u64;
                if v == 0 {
                    return err!(InvalidNullPointerUsage);
                }
                v
            },
            PrimVal::Undef => return err!(ReadUndefBytes),
        };
        if offset % align == 0 {
            Ok(())
        } else {
            err!(AlignmentCheckFailed {
                has: offset % align,
                required: align,
            })
        }
    }

    pub fn check_bounds(&self, ptr: MemoryPointer, access: bool) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        let allocation_size = alloc.bytes.len() as u64;
        if ptr.offset > allocation_size {
            return err!(PointerOutOfBounds { ptr, access, allocation_size });
        }
        Ok(())
    }

    pub(crate) fn set_cur_frame(&mut self, cur_frame: usize) {
        self.cur_frame = cur_frame;
    }
}

/// Locking
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    pub(crate) fn check_locks(&self, ptr: MemoryPointer, len: u64, access: AccessKind) -> EvalResult<'tcx> {
        if len == 0 {
            return Ok(())
        }
        let alloc = self.get(ptr.alloc_id)?;
        let frame = self.cur_frame;
        alloc.check_locks(Some(frame), ptr.offset, len, access)
            .map_err(|lock| EvalErrorKind::MemoryLockViolation { ptr, len, frame, access, lock: lock.active }.into())
    }

    /// Acquire the lock for the given lifetime
    pub(crate) fn acquire_lock(&mut self, ptr: MemoryPointer, len: u64, region: Option<CodeExtent>, kind: AccessKind) -> EvalResult<'tcx> {
        let frame = self.cur_frame;
        assert!(len > 0);
        trace!("Frame {} acquiring {:?} lock at {:?}, size {} for region {:?}", frame, kind, ptr, len, region);
        self.check_bounds(ptr.offset(len, self.layout)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        let alloc = self.get_mut_unchecked(ptr.alloc_id)?;

        // Iterate over our range and acquire the lock.  If the range is already split into pieces,
        // we have to manipulate all of them.
        let lifetime = DynamicLifetime { frame, region };
        for lock in alloc.locks.iter_mut(ptr.offset, len) {
            if !lock.access_permitted(None, kind) {
                return err!(MemoryAcquireConflict { ptr, len, kind, lock: lock.active.clone() });
            }
            // See what we have to do
            match (&mut lock.active, kind) {
                (active @ &mut NoLock, AccessKind::Write) => {
                    *active = WriteLock(lifetime);
                }
                (active @ &mut NoLock, AccessKind::Read) => {
                    *active = ReadLock(vec![lifetime]);
                }
                (&mut ReadLock(ref mut lifetimes), AccessKind::Read) => {
                    lifetimes.push(lifetime);
                }
                _ => bug!("We already checked that there is no conflicting lock"),
            }
        };
        Ok(())
    }

    /// Release or suspend a write lock of the given lifetime prematurely.
    /// When releasing, if there is no write lock or someone else's write lock, that's an error.
    /// When suspending, the same cases are fine; we just register an additional suspension.
    pub(crate) fn release_write_lock(&mut self, ptr: MemoryPointer, len: u64,
                                     lock_region: Option<CodeExtent>, suspend: Option<CodeExtent>) -> EvalResult<'tcx> {
        assert!(len > 0);
        let cur_frame = self.cur_frame;
        let lock_lft = DynamicLifetime { frame: cur_frame, region: lock_region };
        let alloc = self.get_mut_unchecked(ptr.alloc_id)?;

        'locks: for lock in alloc.locks.iter_mut(ptr.offset, len) {
            trace!("Releasing {:?}", lock);
            let is_our_lock = match lock.active {
                WriteLock(lft) => {
                    lft == lock_lft
                }
                ReadLock(_) | NoLock => {
                    false
                }
            };
            if is_our_lock {
                // Disable the lock
                lock.active = NoLock;
            }
            match suspend {
                Some(suspend_region) => {
                    if is_our_lock {
                        // We just released this lock, so add a new suspension
                        lock.suspended.push(SuspendedWriteLock { lft: lock_lft, suspensions: vec![suspend_region] });
                    } else {
                        // Find our lock in the suspended ones
                        for suspended_lock in lock.suspended.iter_mut().rev() {
                            if suspended_lock.lft == lock_lft {
                                // Found it!
                                suspended_lock.suspensions.push(suspend_region);
                                continue 'locks;
                            }
                        }
                        // We did not find it.  Someone else had the lock and we have not suspended it, that's just wrong.
                        return err!(InvalidMemoryLockRelease { ptr, len, frame: cur_frame, lock: lock.active.clone() });
                    }
                }
                None => {
                    // If we do not suspend, make sure we actually released something
                    if !is_our_lock {
                        return err!(InvalidMemoryLockRelease { ptr, len, frame: cur_frame, lock: lock.active.clone() });
                    }
                }
            }
        }

        Ok(())
    }

    /// Release a suspension from the write lock.  If this is the last suspension or if there is no suspension, acquire the lock.
    pub(crate) fn recover_write_lock(&mut self, ptr: MemoryPointer, len: u64,
                                     lock_region: Option<CodeExtent>, suspended_region: CodeExtent, )
        -> EvalResult<'tcx>
    {
        assert!(len > 0);
        let cur_frame = self.cur_frame;
        let lock_lft = DynamicLifetime { frame: cur_frame, region: lock_region };
        let alloc = self.get_mut_unchecked(ptr.alloc_id)?;

        for lock in alloc.locks.iter_mut(ptr.offset, len) {
            // If we have a suspension here, it will be the topmost one
            let (got_the_lock, pop_suspension) = match lock.suspended.last_mut() {
                None => (true, false),
                Some(suspended_lock) => {
                    if suspended_lock.lft == lock_lft {
                        // That's us!  Remove suspension (it should be in there).  The same suspension can
                        // occur multiple times (when there are multiple shared borrows of this that have the same
                        // lifetime); only remove one of them.
                        let idx = match suspended_lock.suspensions.iter().enumerate().find(|&(_, re)| re == &suspended_region) {
                            None => // TODO: Can the user trigger this?
                                bug!("We have this lock suspended, but not for the given region."),
                            Some((idx, _)) => idx
                        };
                        suspended_lock.suspensions.remove(idx);
                        let got_lock = suspended_lock.suspensions.is_empty();
                        (got_lock, got_lock)
                    } else {
                        // Someone else's suspension up top, we should be able to grab the lock
                        (true, false)
                    }
                }
            };
            if pop_suspension { // with NLL; we could do that up in the match above...
                lock.suspended.pop();
            } else {
                // Sanity check: Our lock should not be in the suspension list
                let found = lock.suspended.iter().find(|suspended_lock| suspended_lock.lft == lock_lft);
                assert!(found.is_none());
            }
            if got_the_lock {
                match lock.active {
                    ref mut active @ NoLock => {
                        *active = WriteLock(lock_lft);
                    }
                    _ => {
                        return err!(MemoryAcquireConflict { ptr, len, kind: AccessKind::Write, lock: lock.active.clone() })
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn locks_lifetime_ended(&mut self, ending_region: Option<CodeExtent>) {
        let cur_frame = self.cur_frame;
        trace!("Releasing frame {} locks that expire at {:?}", cur_frame, ending_region);
        let has_ended =  |lifetime: &DynamicLifetime| -> bool {
            if lifetime.frame != cur_frame {
                return false;
            }
            match ending_region {
                None => true, // When a function ends, we end *all* its locks. It's okay for a function to still have lifetime-related locks
                              // when it returns, that can happen e.g. with NLL when a lifetime can, but does not have to, extend beyond the
                              // end of a function.  Same for a function still having recoveries.
                Some(ending_region) => lifetime.region == Some(ending_region),
            }
        };

        for alloc in self.alloc_map.values_mut() {
            for lock in alloc.locks.iter_mut_all() {
                // Delete everything that ends now -- i.e., keep only all the other lifetimes.
                let lock_ended = match lock.active {
                    WriteLock(ref lft) => {
                        has_ended(lft)
                    }
                    ReadLock(ref mut lfts) => {
                        lfts.retain(|lft| !has_ended(lft));
                        lfts.is_empty()
                    }
                    NoLock => false,
                };
                if lock_ended {
                    lock.active = NoLock;
                }
                // Also clean up suspended write locks
                lock.suspended.retain(|suspended_lock| !has_ended(&suspended_lock.lft));
            }
            // Clean up the map
            alloc.locks.retain(|lock| {
                match lock.active { NoLock => lock.suspended.len() > 0, _ => true }
            });
        }
    }
}

/// Allocation accessors
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    pub fn get(&self, id: AllocId) -> EvalResult<'tcx, &Allocation<M::MemoryKinds>> {
        match id.destructure() {
            AllocIdKind::Function(_) => err!(DerefFunctionPointer),
            AllocIdKind::Runtime(id) => match self.alloc_map.get(&id) {
                Some(alloc) => Ok(alloc),
                None => err!(DanglingPointerDeref),
            },
        }
    }
    
    fn get_mut_unchecked(&mut self, id: AllocId) -> EvalResult<'tcx, &mut Allocation<M::MemoryKinds>> {
        match id.destructure() {
            AllocIdKind::Function(_) => err!(DerefFunctionPointer),
            AllocIdKind::Runtime(id) => match self.alloc_map.get_mut(&id) {
                Some(alloc) => Ok(alloc),
                None => err!(DanglingPointerDeref),
            },
        }
    }

    pub fn get_mut(&mut self, id: AllocId) -> EvalResult<'tcx, &mut Allocation<M::MemoryKinds>> {
        let alloc = self.get_mut_unchecked(id)?;
        if alloc.mutable == Mutability::Mutable {
            Ok(alloc)
        } else {
            err!(ModifiedConstantMemory)
        }
    }

    pub fn get_fn(&self, ptr: MemoryPointer) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        if ptr.offset != 0 {
            return err!(InvalidFunctionPointer);
        }
        debug!("reading fn ptr: {}", ptr.alloc_id);
        match ptr.alloc_id.destructure() {
            AllocIdKind::Function(id) => Ok(self.functions[id]),
            AllocIdKind::Runtime(_) => err!(ExecuteMemory),
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

            let alloc = match id.destructure() {
                AllocIdKind::Function(id) => {
                    trace!("{} {}", msg, self.functions[id]);
                    continue;
                },
                AllocIdKind::Runtime(id) => match self.alloc_map.get(&id) {
                    Some(a) => a,
                    None => {
                        trace!("{} (deallocated)", msg);
                        continue;
                    }
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

            let immutable = match (alloc.kind, alloc.mutable) {
                (Kind::UninitializedStatic, _) => " (static in the process of initialization)".to_owned(),
                (Kind::Static, Mutability::Mutable) => " (static mut)".to_owned(),
                (Kind::Static, Mutability::Immutable) => " (immutable)".to_owned(),
                (Kind::Machine(m), _) => format!(" ({:?})", m),
                (Kind::Stack, _) => " (stack)".to_owned(),
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
                    Some(AllocIdKind::Runtime(key).into_alloc_id())
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
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
    fn get_bytes_unchecked(&self, ptr: MemoryPointer, size: u64, align: u64) -> EvalResult<'tcx, &[u8]> {
        // Zero-sized accesses can use dangling pointers, but they still have to be aligned and non-NULL
        if self.reads_are_aligned.get() {
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
        if self.writes_are_aligned.get() {
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
            return err!(ReadPointerAsBytes);
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
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
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
        let alloc_id = match alloc_id.destructure() {
            AllocIdKind::Function(_) => return Ok(()),
            AllocIdKind::Runtime(id) => id,
        };
        let relocations = match self.alloc_map.get_mut(&alloc_id) {
            Some(&mut Allocation { ref mut relocations, ref mut kind, ref mut mutable, .. }) => {
                match *kind {
                    // const eval results can refer to "locals".
                    // E.g. `const Foo: &u32 = &1;` refers to the temp local that stores the `1`
                    Kind::Stack |
                    // The entire point of this function
                    Kind::UninitializedStatic => {},
                    Kind::Machine(m) => M::mark_static_initialized(m)?,
                    Kind::Static => {
                        trace!("mark_static_initalized: skipping already initialized static referred to by static currently being initialized");
                        return Ok(());
                    },
                }
                *kind = Kind::Static;
                *mutable = mutability;
                // take out the relocations vector to free the borrow on self, so we can call
                // mark recursively
                mem::replace(relocations, Default::default())
            },
            None => return err!(DanglingPointerDeref),
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
            if self.reads_are_aligned.get() {
                self.check_align(src, align)?;
            }
            if self.writes_are_aligned.get() {
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
                        return err!(Intrinsic(format!("copy_nonoverlapping called on overlapping ranges")));
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
                    return err!(ReadPointerAsBytes);
                }
                self.check_defined(ptr, (size + 1) as u64)?;
                self.check_locks(ptr, (size + 1) as u64, AccessKind::Read)?;
                Ok(&alloc.bytes[offset..offset + size])
            },
            None => err!(UnterminatedCString(ptr)),
        }
    }

    pub fn read_bytes(&self, ptr: Pointer, size: u64) -> EvalResult<'tcx, &[u8]> {
        if size == 0 {
            // Empty accesses don't need to be valid pointers, but they should still be non-NULL
            if self.reads_are_aligned.get() {
                self.check_align(ptr, 1)?;
            }
            return Ok(&[]);
        }
        self.get_bytes(ptr.to_ptr()?, size, 1)
    }

    pub fn write_bytes(&mut self, ptr: Pointer, src: &[u8]) -> EvalResult<'tcx> {
        if src.is_empty() {
            // Empty accesses don't need to be valid pointers, but they should still be non-NULL
            if self.writes_are_aligned.get() {
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
            if self.writes_are_aligned.get() {
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
            _ => err!(InvalidBool),
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
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
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
            return err!(ReadPointerAsBytes);
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
impl<'a, 'tcx, M: Machine<'tcx>> Memory<'a, 'tcx, M> {
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
            return err!(ReadUndefBytes);
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

pub trait HasMemory<'a, 'tcx, M: Machine<'tcx>> {
    fn memory_mut(&mut self) -> &mut Memory<'a, 'tcx, M>;
    fn memory(&self) -> &Memory<'a, 'tcx, M>;

    // These are not supposed to be overriden.
    fn read_maybe_aligned<F, T>(&self, aligned: bool, f: F) -> EvalResult<'tcx, T>
        where F: FnOnce(&Self) -> EvalResult<'tcx, T>
    {
        let old = self.memory().reads_are_aligned.get();
        // Do alignment checking if *all* nested calls say it has to be aligned.
        self.memory().reads_are_aligned.set(old && aligned);
        let t = f(self);
        self.memory().reads_are_aligned.set(old);
        t
    }

    fn read_maybe_aligned_mut<F, T>(&mut self, aligned: bool, f: F) -> EvalResult<'tcx, T>
        where F: FnOnce(&mut Self) -> EvalResult<'tcx, T>
    {
        let old = self.memory().reads_are_aligned.get();
        // Do alignment checking if *all* nested calls say it has to be aligned.
        self.memory().reads_are_aligned.set(old && aligned);
        let t = f(self);
        self.memory().reads_are_aligned.set(old);
        t
    }

    fn write_maybe_aligned_mut<F, T>(&mut self, aligned: bool, f: F) -> EvalResult<'tcx, T>
        where F: FnOnce(&mut Self) -> EvalResult<'tcx, T>
    {
        let old = self.memory().writes_are_aligned.get();
        // Do alignment checking if *all* nested calls say it has to be aligned.
        self.memory().writes_are_aligned.set(old && aligned);
        let t = f(self);
        self.memory().writes_are_aligned.set(old);
        t
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
            err!(OverflowingMath)
        } else {
            Ok(res)
        }
    }

    fn offset<'tcx>(self, val: u64, i: u64) -> EvalResult<'tcx, u64> {
        let (res, over) = self.overflowing_offset(val, i);
        if over {
            err!(OverflowingMath)
        } else {
            Ok(res)
        }
    }

    fn wrapping_signed_offset(self, val: u64, i: i64) -> u64 {
        self.overflowing_signed_offset(val, i as i128).0
    }
}

impl<T: layout::HasDataLayout> PointerArithmetic for T {}

impl<'a, 'tcx, M: Machine<'tcx>> layout::HasDataLayout for &'a Memory<'a, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        self.layout
    }
}
impl<'a, 'tcx, M: Machine<'tcx>> layout::HasDataLayout for &'a EvalContext<'a, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        self.memory().layout
    }
}

impl<'c, 'b, 'a, 'tcx, M: Machine<'tcx>> layout::HasDataLayout for &'c &'b mut EvalContext<'a, 'tcx, M> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        self.memory().layout
    }
}
