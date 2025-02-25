//! Thread Local Storage
//!
//! Currently, we are limited to 1023 TLS entries. The entries
//! live in a page of memory that's unique per-process, and is
//! stored in the `$tp` register. If this register is 0, then
//! TLS has not been initialized and thread cleanup can be skipped.
//!
//! The index into this register is the `key`. This key is identical
//! between all threads, but indexes a different offset within this
//! pointer.
//!
//! # Dtor registration (stolen from Windows)
//!
//! Xous has no native support for running destructors so we manage our own
//! list of destructors to keep track of how to destroy keys. When a thread
//! or the process exits, `run_dtors` is called, which will iterate through
//! the list and run the destructors.
//!
//! Currently unregistration from this list is not supported. A destructor can be
//! registered but cannot be unregistered. There's various simplifying reasons
//! for doing this, the big ones being:
//!
//! 1. Currently we don't even support deallocating TLS keys, so normal operation
//!    doesn't need to deallocate a destructor.
//! 2. There is no point in time where we know we can unregister a destructor
//!    because it could always be getting run by some remote thread.
//!
//! Typically processes have a statically known set of TLS keys which is pretty
//! small, and we'd want to keep this memory alive for the whole process anyway
//! really.
//!
//! Perhaps one day we can fold the `Box` here into a static allocation,
//! expanding the `LazyKey` structure to contain not only a slot for the TLS
//! key but also a slot for the destructor queue on windows. An optimization for
//! another day!

// FIXME(joboet): implement support for native TLS instead.

use core::arch::asm;

use crate::mem::ManuallyDrop;
use crate::os::xous::ffi::{MemoryFlags, map_memory, unmap_memory};
use crate::ptr;
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::sync::atomic::{AtomicPtr, AtomicUsize};

pub type Key = usize;
pub type Dtor = unsafe extern "C" fn(*mut u8);

const TLS_MEMORY_SIZE: usize = 4096;

/// TLS keys start at `1`. Index `0` is unused
#[cfg(not(test))]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys4xous16thread_local_key13TLS_KEY_INDEXE")]
static TLS_KEY_INDEX: AtomicUsize = AtomicUsize::new(1);

#[cfg(not(test))]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys4xous16thread_local_key9DTORSE")]
static DTORS: AtomicPtr<Node> = AtomicPtr::new(ptr::null_mut());

#[cfg(test)]
unsafe extern "Rust" {
    #[link_name = "_ZN16__rust_internals3std3sys4xous16thread_local_key13TLS_KEY_INDEXE"]
    static TLS_KEY_INDEX: AtomicUsize;

    #[link_name = "_ZN16__rust_internals3std3sys4xous16thread_local_key9DTORSE"]
    static DTORS: AtomicPtr<Node>;
}

fn tls_ptr_addr() -> *mut *mut u8 {
    let mut tp: usize;
    unsafe {
        asm!(
            "mv {}, tp",
            out(reg) tp,
        );
    }
    core::ptr::with_exposed_provenance_mut::<*mut u8>(tp)
}

/// Creates an area of memory that's unique per thread. This area will
/// contain all thread local pointers.
fn tls_table() -> &'static mut [*mut u8] {
    let tp = tls_ptr_addr();

    if !tp.is_null() {
        return unsafe {
            core::slice::from_raw_parts_mut(tp, TLS_MEMORY_SIZE / core::mem::size_of::<*mut u8>())
        };
    }
    // If the TP register is `0`, then this thread hasn't initialized
    // its TLS yet. Allocate a new page to store this memory.
    let tp = unsafe {
        map_memory(
            None,
            None,
            TLS_MEMORY_SIZE / core::mem::size_of::<*mut u8>(),
            MemoryFlags::R | MemoryFlags::W,
        )
        .expect("Unable to allocate memory for thread local storage")
    };

    for val in tp.iter() {
        assert!(*val as usize == 0);
    }

    unsafe {
        // Set the thread's `$tp` register
        asm!(
            "mv tp, {}",
            in(reg) tp.as_mut_ptr() as usize,
        );
    }
    tp
}

#[inline]
pub fn create(dtor: Option<Dtor>) -> Key {
    // Allocate a new TLS key. These keys are shared among all threads.
    #[allow(unused_unsafe)]
    let key = unsafe { TLS_KEY_INDEX.fetch_add(1, Relaxed) };
    if let Some(f) = dtor {
        unsafe { register_dtor(key, f) };
    }
    key
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    assert!((key < 1022) && (key >= 1));
    let table = tls_table();
    table[key] = value;
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    assert!((key < 1022) && (key >= 1));
    tls_table()[key]
}

#[inline]
pub unsafe fn destroy(_key: Key) {
    // Just leak the key. Probably not great on long-running systems that create
    // lots of TLS variables, but in practice that's not an issue.
}

struct Node {
    dtor: Dtor,
    key: Key,
    next: *mut Node,
}

unsafe fn register_dtor(key: Key, dtor: Dtor) {
    let mut node = ManuallyDrop::new(Box::new(Node { key, dtor, next: ptr::null_mut() }));

    #[allow(unused_unsafe)]
    let mut head = unsafe { DTORS.load(Acquire) };
    loop {
        node.next = head;
        #[allow(unused_unsafe)]
        match unsafe { DTORS.compare_exchange(head, &mut **node, Release, Acquire) } {
            Ok(_) => return, // nothing to drop, we successfully added the node to the list
            Err(cur) => head = cur,
        }
    }
}

pub unsafe fn destroy_tls() {
    let tp = tls_ptr_addr();

    // If the pointer address is 0, then this thread has no TLS.
    if tp.is_null() {
        return;
    }

    unsafe { run_dtors() };

    // Finally, free the TLS array
    unsafe {
        unmap_memory(core::slice::from_raw_parts_mut(
            tp,
            TLS_MEMORY_SIZE / core::mem::size_of::<usize>(),
        ))
        .unwrap()
    };
}

unsafe fn run_dtors() {
    let mut any_run = true;

    // Run the destructor "some" number of times. This is 5x on Windows,
    // so we copy it here. This allows TLS variables to create new
    // TLS variables upon destruction that will also get destroyed.
    // Keep going until we run out of tries or until we have nothing
    // left to destroy.
    for _ in 0..5 {
        if !any_run {
            break;
        }
        any_run = false;
        #[allow(unused_unsafe)]
        let mut cur = unsafe { DTORS.load(Acquire) };
        while !cur.is_null() {
            let ptr = unsafe { get((*cur).key) };

            if !ptr.is_null() {
                unsafe { set((*cur).key, ptr::null_mut()) };
                unsafe { ((*cur).dtor)(ptr as *mut _) };
                any_run = true;
            }

            unsafe { cur = (*cur).next };
        }
    }

    crate::rt::thread_cleanup();
}
