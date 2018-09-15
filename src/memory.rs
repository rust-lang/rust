use std::collections::{HashMap, BTreeMap};

use rustc::ty;

use super::{AllocId, Scalar, LockInfo, RangeMap};

pub type TlsKey = u128;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TlsEntry<'tcx> {
    pub(crate) data: Scalar, // Will eventually become a map from thread IDs to `Scalar`s, if we ever support more than one thread.
    pub(crate) dtor: Option<ty::Instance<'tcx>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryData<'tcx> {
    /// The Key to use for the next thread-local allocation.
    pub(crate) next_thread_local: TlsKey,

    /// pthreads-style thread-local storage.
    pub(crate) thread_local: BTreeMap<TlsKey, TlsEntry<'tcx>>,

    /// Memory regions that are locked by some function
    ///
    /// Only mutable (static mut, heap, stack) allocations have an entry in this map.
    /// The entry is created when allocating the memory and deleted after deallocation.
    pub(crate) locks: HashMap<AllocId, RangeMap<LockInfo<'tcx>>>,
}

impl<'tcx> MemoryData<'tcx> {
    pub(crate) fn new() -> Self {
        MemoryData {
            next_thread_local: 1, // start with 1 as we must not use 0 on Windows
            thread_local: BTreeMap::new(),
            locks: HashMap::new(),
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryKind {
    /// `__rust_alloc` memory
    Rust,
    /// `malloc` memory
    C,
    /// Part of env var emulation
    Env,
    /// mutable statics
    MutStatic,
}

impl Into<::rustc_mir::interpret::MemoryKind<MemoryKind>> for MemoryKind {
    fn into(self) -> ::rustc_mir::interpret::MemoryKind<MemoryKind> {
        ::rustc_mir::interpret::MemoryKind::Machine(self)
    }
}
