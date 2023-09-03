#![unstable(issue = "none", feature = "std_internals")]

use crate::alloc::{GlobalAlloc, Layout, System};
use crate::os::custom::alloc::IMPL;
use crate::sync::Mutex;
use core::ops::{Deref, DerefMut};

// Simple implementation of a sequential fit allocator
//
// minimal size and alignment: two bytes (less than that => wasted bytes)
// [ NN NN ]: uneven, two-byte pointer to next free slot
// [ SS SS ]: even, two-byte size of current spot
// uneven size => size = 2, current pair is the next ptr
// even value of zero => continue to next byte pair
// even value, non-zero => offset into the byte array to next free slot (right-shifted by one bit)

//   :::::                           ::::::::::::                :::::  :::::        ::::::::::::::::::::::::
// [ XX XX SS SS  00 00 00 00  NN NN XX XX  XX XX SS SS  NN NN   XX XX  XX XX NN NN  XX XX XX XX  XX XX XX XX  SS SS NN NN ]
//         |                       \              \          \______ --> _____/___/_________ --> ______________/
//         first free slot          \              \_________ <-- ___________/___/
//                                   \___________ --> ______________________/
//
// ::::: = occupied slot

// ALLOCATING
//
// a. keep track of the origin of the current spot pointer (first free slot global var / end of the previous spot)
// b. find first free slot offset (from a global variable)
// c. read the size of the current slot
// d. if the size is enough for the required layout (taking alignment into account), use this spot (goto step f)
// e. else, read next pointer (last pair of the spot) and retry at step c with new offset
// f. if there is unusable space at the beginning of the spot (alignment/padding), create a new spot there
// g. if there is leftover space at the end of the spot, create a new spot there
// h. update the current spot pointer origin so that it points to the next free spot

// DEALLOCATING
//
// a. find first free slot offset (from a global variable)
// b. arrange the bytes in the freed slot (size pair + next pointer pair)
//    note: copy the value of the global "first free slot" variable into the next pointer pair
// c. update the global "first free slot" variable

// maximum: 0xffff
// more than 0xffff => will cause infinite loops
const SIZE_BYTES: usize = 4096 * 4;
type HeapArray = [u8; SIZE_BYTES];

// align the heap to a page
#[repr(align(4096))]
struct Heap(HeapArray);

impl Deref for Heap {
    type Target = HeapArray;
    fn deref(&self) -> &HeapArray {
        &self.0
    }
}

impl DerefMut for Heap {
    fn deref_mut(&mut self) -> &mut HeapArray {
        &mut self.0
    }
}

static mut HEAP: Heap = Heap(init_heap());
static FIRST_SLOT: Mutex<usize> = Mutex::new(0);

const fn init_heap() -> [u8; SIZE_BYTES] {
    let mut pages = [0; SIZE_BYTES];

    let len = SIZE_BYTES as u16;
    let [a, b] = len.to_ne_bytes();
    pages[0] = a;
    pages[1] = b;

    let j = SIZE_BYTES - 2;
    let out_of_bounds = 0xffffu16;
    let [a, b] = out_of_bounds.to_ne_bytes();
    pages[j + 0] = a;
    pages[j + 1] = b;

    pages
}

struct DefaultAlloc;

fn read_u16(i: usize) -> usize {
    let bytes = unsafe { [HEAP[i + 0], HEAP[i + 1]] };
    u16::from_ne_bytes(bytes) as usize
}

fn write_u16(i: usize, value: usize) {
    let bytes = (value as u16).to_ne_bytes();
    unsafe { HEAP[i..][..2].copy_from_slice(&bytes) }
}

fn decode_slot(i: usize) -> (usize, usize) {
    let first_pair = read_u16(i);
    let mut len = first_pair;
    let next;

    if (first_pair & 1) != 0 {
        // uneven => this is the next pointer (two-byte spot)
        next = first_pair;
        len = 2;
    } else {
        next = read_u16(i + len - 2);
    }

    (len, next & !1)
}

fn encode_slot(i: usize, len: usize, next: usize) {
    assert_ne!(len, 0);
    assert_eq!(len & 1, 0);

    if len != 2 {
        write_u16(i, len)
    }

    let j = i + len - 2;
    write_u16(j, next | 1)
}

fn free(i: usize, len: usize) {
    assert!(len >= 2);

    let mut first_slot = FIRST_SLOT.lock_no_poison_check();
    encode_slot(i, len, *first_slot);
    *first_slot = i;
}

fn alignment_filler(i: usize, req_align: usize) -> usize {
    let offset = i & (req_align - 1);
    match offset == 0 {
        true => 0,
        false => req_align - offset,
    }
}

fn prepare_layout(layout: Layout) -> (usize, usize) {
    let req_align = layout.align().max(2); // any power of two: 1, 2, 4, 8, 16, 32, 64
    let mut req_size = layout.size();

    assert_ne!(req_size, 0);
    assert_ne!(req_align, 0);

    // make size even
    req_size += req_size & 1;

    (req_align, req_size)
}

fn default_alloc_find(ptr: *mut u8) -> Option<usize> {
    let start = unsafe { HEAP.as_ptr() } as usize;
    let ptr = ptr as usize;

    ptr.checked_sub(start).filter(|offset| *offset < SIZE_BYTES)
}

unsafe impl GlobalAlloc for DefaultAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let (req_align, req_size) = prepare_layout(layout);

        let (mut filler, leftover);
        let mut first_slot = FIRST_SLOT.lock_no_poison_check();
        let mut i = *first_slot;
        let mut prev = None;

        loop {
            if i >= SIZE_BYTES {
                // out of memory
                return core::ptr::null_mut();
            }

            let (len, next_ptr) = decode_slot(i);
            filler = alignment_filler(i, req_align);

            if let Some(surplus) = len.checked_sub(filler + req_size) {
                leftover = surplus;

                if let Some(j) = prev {
                    let (len, _) = decode_slot(j);
                    encode_slot(j, len, next_ptr);
                } else {
                    *first_slot = next_ptr;
                }

                break;
            } else {
                prev = Some(i);
                i = next_ptr;
            }
        }

        drop(first_slot);

        if filler > 0 {
            free(i, filler);
        }

        if leftover > 0 {
            free(i + filler + req_size, leftover);
        }

        let start = HEAP.as_ptr() as usize;
        core::ptr::from_exposed_addr_mut(start + i)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let i = default_alloc_find(ptr).expect("invalid pointer");
        let (_, req_size) = prepare_layout(layout);
        free(i, req_size);
    }
}

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let reader = IMPL.read().expect("poisoned lock");
        if let Some(some_impl) = reader.as_ref() {
            some_impl.alloc(layout)
        } else {
            DefaultAlloc.alloc(layout)
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let reader = IMPL.read().expect("poisoned lock");
        if let Some(some_impl) = reader.as_ref() {
            some_impl.alloc_zeroed(layout)
        } else {
            DefaultAlloc.alloc_zeroed(layout)
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(i) = default_alloc_find(ptr) {
            let (_, req_size) = prepare_layout(layout);
            free(i, req_size);
        } else {
            let reader = IMPL.read().expect("poisoned lock");
            let some_impl = reader.as_ref().expect("invalid pointer");
            some_impl.dealloc(ptr, layout)
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if let Some(i) = default_alloc_find(ptr) {
            // if the ptr was allocated by the default allocator:
            // - the current allocator is used for allocation
            // - the default allocator is used for deallocation

            let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
            let new_ptr = self.alloc(new_layout);
            if !new_ptr.is_null() {
                let num_bytes = core::cmp::min(layout.size(), new_size);
                core::ptr::copy_nonoverlapping(ptr, new_ptr, num_bytes);

                let (_, req_size) = prepare_layout(layout);
                free(i, req_size);
            }
            new_ptr
        } else {
            // if the ptr was allocated by another allocator, forward the call.

            let reader = IMPL.read().expect("poisoned lock");
            let some_impl = reader.as_ref().expect("invalid pointer");
            some_impl.realloc(ptr, layout, new_size)
        }
    }
}
