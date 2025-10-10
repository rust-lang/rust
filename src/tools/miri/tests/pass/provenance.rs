//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::{mem, ptr};

const PTR_SIZE: usize = mem::size_of::<&i32>();

fn main() {
    basic();
    bytewise_ptr_methods();
    bytewise_custom_memcpy();
    bytewise_custom_memcpy_chunked();
    int_load_strip_provenance();
    maybe_uninit_preserves_partial_provenance();
}

/// Some basic smoke tests for provenance.
fn basic() {
    let x = &42;
    let ptr = x as *const i32;
    let addr: usize = unsafe { mem::transmute(ptr) }; // an integer without provenance
    // But we can give provenance back via `with_addr`.
    let ptr_back = ptr.with_addr(addr);
    assert_eq!(unsafe { *ptr_back }, 42);

    // It is preserved by MaybeUninit.
    let addr_mu: mem::MaybeUninit<usize> = unsafe { mem::transmute(ptr) };
    let ptr_back: *const i32 = unsafe { mem::transmute(addr_mu) };
    assert_eq!(unsafe { *ptr_back }, 42);
}

fn bytewise_ptr_methods() {
    let mut ptr1 = &1;
    let mut ptr2 = &2;

    // Swap them, bytewise.
    unsafe {
        ptr::swap_nonoverlapping(
            &mut ptr1 as *mut _ as *mut mem::MaybeUninit<u8>,
            &mut ptr2 as *mut _ as *mut mem::MaybeUninit<u8>,
            mem::size_of::<&i32>(),
        );
    }

    // Make sure they still work.
    assert_eq!(*ptr1, 2);
    assert_eq!(*ptr2, 1);

    // TODO: also test ptr::swap, ptr::copy, ptr::copy_nonoverlapping.
}

fn bytewise_custom_memcpy() {
    unsafe fn memcpy<T>(to: *mut T, from: *const T) {
        let to = to.cast::<mem::MaybeUninit<u8>>();
        let from = from.cast::<mem::MaybeUninit<u8>>();
        for i in 0..mem::size_of::<T>() {
            let b = from.add(i).read();
            to.add(i).write(b);
        }
    }

    let ptr1 = &1;
    let mut ptr2 = &2;

    // Copy, bytewise.
    unsafe { memcpy(&mut ptr2, &ptr1) };

    // Make sure they still work.
    assert_eq!(*ptr1, 1);
    assert_eq!(*ptr2, 1);
}

fn bytewise_custom_memcpy_chunked() {
    unsafe fn memcpy<T>(to: *mut T, from: *const T) {
        assert!(mem::size_of::<T>() % mem::size_of::<usize>() == 0);
        let count = mem::size_of::<T>() / mem::size_of::<usize>();
        let to = to.cast::<mem::MaybeUninit<usize>>();
        let from = from.cast::<mem::MaybeUninit<usize>>();
        for i in 0..count {
            let b = from.add(i).read();
            to.add(i).write(b);
        }
    }

    // Prepare an array where pointers are stored at... interesting... offsets.
    let mut data = [0usize; 2 * PTR_SIZE];
    let mut offsets = vec![];
    for i in 0..mem::size_of::<usize>() {
        // We have 2*PTR_SIZE room for each of these pointers.
        let base = i * 2 * PTR_SIZE;
        // This one is mis-aligned by `i`.
        let offset = base + i;
        offsets.push(offset);
        // Store it there.
        unsafe { data.as_mut_ptr().byte_add(offset).cast::<&i32>().write_unaligned(&42) };
    }

    // Now memcpy that.
    let mut data2 = [0usize; 2 * PTR_SIZE];
    unsafe { memcpy(&mut data2, &data) };

    // And check the result.
    for &offset in &offsets {
        let ptr = unsafe { data2.as_ptr().byte_add(offset).cast::<&i32>().read_unaligned() };
        assert_eq!(*ptr, 42);
    }
}

fn int_load_strip_provenance() {
    let ptrs = [&42];
    let ints: [usize; 1] = unsafe { mem::transmute(ptrs) };
    assert_eq!(ptrs[0] as *const _ as usize, ints[0]);
}

fn maybe_uninit_preserves_partial_provenance() {
    // This is the same test as ptr_copy_loses_partial_provenance.rs, but using MaybeUninit and thus
    // properly preserving partial provenance.
    unsafe {
        let mut bytes = [1u8; 16];
        let bytes = bytes.as_mut_ptr();

        // Put a pointer in the middle.
        bytes.add(4).cast::<&i32>().write_unaligned(&42);
        // Copy the entire thing as two pointers but not perfectly
        // overlapping with the pointer we have in there.
        let copy = bytes.cast::<[mem::MaybeUninit<*const ()>; 2]>().read_unaligned();
        let copy_bytes = copy.as_ptr().cast::<u8>();
        // Now go to the middle of the copy and get the pointer back out.
        let ptr = copy_bytes.add(4).cast::<*const i32>().read_unaligned();
        // And deref this to ensure we get the right value.
        let val = *ptr;
        assert_eq!(val, 42);
    }
}
