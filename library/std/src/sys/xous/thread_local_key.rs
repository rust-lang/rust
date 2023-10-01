use crate::mem::ManuallyDrop;
use crate::ptr;
use crate::sync::atomic::AtomicPtr;
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering::SeqCst;
use core::arch::asm;

use crate::os::xous::ffi::{map_memory, unmap_memory, MemoryFlags};

/// Thread Local Storage
///
/// Currently, we are limited to 1023 TLS entries. The entries
/// live in a page of memory that's unique per-process, and is
/// stored in the `$tp` register. If this register is 0, then
/// TLS has not been initialized and thread cleanup can be skipped.
///
/// The index into this register is the `key`. This key is identical
/// between all threads, but indexes a different offset within this
/// pointer.
pub type Key = usize;

pub type Dtor = unsafe extern "C" fn(*mut u8);

const TLS_MEMORY_SIZE: usize = 4096;

/// TLS keys start at `1` to mimic POSIX.
static TLS_KEY_INDEX: AtomicUsize = AtomicUsize::new(1);

fn tls_ptr_addr() -> *mut usize {
    let mut tp: usize;
    unsafe {
        asm!(
            "mv {}, tp",
            out(reg) tp,
        );
    }
    core::ptr::from_exposed_addr_mut::<usize>(tp)
}

/// Create an area of memory that's unique per thread. This area will
/// contain all thread local pointers.
fn tls_ptr() -> *mut usize {
    let mut tp = tls_ptr_addr();

    // If the TP register is `0`, then this thread hasn't initialized
    // its TLS yet. Allocate a new page to store this memory.
    if tp.is_null() {
        tp = unsafe {
            map_memory(
                None,
                None,
                TLS_MEMORY_SIZE / core::mem::size_of::<usize>(),
                MemoryFlags::R | MemoryFlags::W,
            )
        }
        .expect("Unable to allocate memory for thread local storage")
        .as_mut_ptr();

        unsafe {
            // Key #0 is currently unused.
            (tp).write_volatile(0);

            // Set the thread's `$tp` register
            asm!(
                "mv tp, {}",
                in(reg) tp as usize,
            );
        }
    }
    tp
}

/// Allocate a new TLS key. These keys are shared among all threads.
fn tls_alloc() -> usize {
    TLS_KEY_INDEX.fetch_add(1, SeqCst)
}

#[inline]
pub unsafe fn create(dtor: Option<Dtor>) -> Key {
    let key = tls_alloc();
    if let Some(f) = dtor {
        unsafe { register_dtor(key, f) };
    }
    key
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    assert!((key < 1022) && (key >= 1));
    unsafe { tls_ptr().add(key).write_volatile(value as usize) };
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    assert!((key < 1022) && (key >= 1));
    core::ptr::from_exposed_addr_mut::<u8>(unsafe { tls_ptr().add(key).read_volatile() })
}

#[inline]
pub unsafe fn destroy(_key: Key) {
    panic!("can't destroy keys on Xous");
}

// -------------------------------------------------------------------------
// Dtor registration (stolen from Windows)
//
// Xous has no native support for running destructors so we manage our own
// list of destructors to keep track of how to destroy keys. We then install a
// callback later to get invoked whenever a thread exits, running all
// appropriate destructors.
//
// Currently unregistration from this list is not supported. A destructor can be
// registered but cannot be unregistered. There's various simplifying reasons
// for doing this, the big ones being:
//
// 1. Currently we don't even support deallocating TLS keys, so normal operation
//    doesn't need to deallocate a destructor.
// 2. There is no point in time where we know we can unregister a destructor
//    because it could always be getting run by some remote thread.
//
// Typically processes have a statically known set of TLS keys which is pretty
// small, and we'd want to keep this memory alive for the whole process anyway
// really.
//
// Perhaps one day we can fold the `Box` here into a static allocation,
// expanding the `StaticKey` structure to contain not only a slot for the TLS
// key but also a slot for the destructor queue on windows. An optimization for
// another day!

static DTORS: AtomicPtr<Node> = AtomicPtr::new(ptr::null_mut());

struct Node {
    dtor: Dtor,
    key: Key,
    next: *mut Node,
}

unsafe fn register_dtor(key: Key, dtor: Dtor) {
    let mut node = ManuallyDrop::new(Box::new(Node { key, dtor, next: ptr::null_mut() }));

    let mut head = DTORS.load(SeqCst);
    loop {
        node.next = head;
        match DTORS.compare_exchange(head, &mut **node, SeqCst, SeqCst) {
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
    for _ in 0..5 {
        if !any_run {
            break;
        }
        any_run = false;
        let mut cur = DTORS.load(SeqCst);
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
}
