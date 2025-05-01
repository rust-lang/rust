//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-strict-provenance
#![feature(slice_partition_dedup)]
#![feature(layout_for_ptr)]

use std::{ptr, slice};

fn slice_of_zst() {
    fn foo<T>(v: &[T]) -> Option<&[T]> {
        let mut it = v.iter();
        for _ in 0..5 {
            it.next();
        }
        Some(it.as_slice())
    }

    fn foo_mut<T>(v: &mut [T]) -> Option<&mut [T]> {
        let mut it = v.iter_mut();
        for _ in 0..5 {
            it.next();
        }
        Some(it.into_slice())
    }

    // In a slice of zero-size elements the pointer is meaningless.
    // Ensure iteration still works even if the pointer is at the end of the address space.
    let slice: &[()] =
        unsafe { slice::from_raw_parts(ptr::without_provenance(-5isize as usize), 10) };
    assert_eq!(slice.len(), 10);
    assert_eq!(slice.iter().count(), 10);

    // .nth() on the iterator should also behave correctly
    let mut it = slice.iter();
    assert!(it.nth(5).is_some());
    assert_eq!(it.count(), 4);

    // Converting Iter to a slice should never have a null pointer
    assert!(foo(slice).is_some());

    // Test mutable iterators as well
    let slice: &mut [()] =
        unsafe { slice::from_raw_parts_mut(ptr::without_provenance_mut(-5isize as usize), 10) };
    assert_eq!(slice.len(), 10);
    assert_eq!(slice.iter_mut().count(), 10);

    {
        let mut it = slice.iter_mut();
        assert!(it.nth(5).is_some());
        assert_eq!(it.count(), 4);
    }

    assert!(foo_mut(slice).is_some())
}

fn test_iter_ref_consistency() {
    use std::fmt::Debug;

    fn test<T: Copy + Debug + PartialEq>(x: T) {
        let v: &[T] = &[x, x, x];
        let v_ptrs: [*const T; 3] = match v {
            [ref v1, ref v2, ref v3] => [v1 as *const _, v2 as *const _, v3 as *const _],
            _ => unreachable!(),
        };
        let len = v.len();

        // nth(i)
        for i in 0..len {
            assert_eq!(&v[i] as *const _, v_ptrs[i]); // check the v_ptrs array, just to be sure
            let nth = v.iter().nth(i).unwrap();
            assert_eq!(nth as *const _, v_ptrs[i]);
        }
        assert_eq!(v.iter().nth(len), None, "nth(len) should return None");

        // stepping through with nth(0)
        {
            let mut it = v.iter();
            for i in 0..len {
                let next = it.nth(0).unwrap();
                assert_eq!(next as *const _, v_ptrs[i]);
            }
            assert_eq!(it.nth(0), None);
        }

        // next()
        {
            let mut it = v.iter();
            for i in 0..len {
                let remaining = len - i;
                assert_eq!(it.size_hint(), (remaining, Some(remaining)));

                let next = it.next().unwrap();
                assert_eq!(next as *const _, v_ptrs[i]);
            }
            assert_eq!(it.size_hint(), (0, Some(0)));
            assert_eq!(it.next(), None, "The final call to next() should return None");
        }

        // next_back()
        {
            let mut it = v.iter();
            for i in 0..len {
                let remaining = len - i;
                assert_eq!(it.size_hint(), (remaining, Some(remaining)));

                let prev = it.next_back().unwrap();
                assert_eq!(prev as *const _, v_ptrs[remaining - 1]);
            }
            assert_eq!(it.size_hint(), (0, Some(0)));
            assert_eq!(it.next_back(), None, "The final call to next_back() should return None");
        }
    }

    fn test_mut<T: Copy + Debug + PartialEq>(x: T) {
        let v: &mut [T] = &mut [x, x, x];
        let v_ptrs: [*mut T; 3] = match v {
            [ref v1, ref v2, ref v3] =>
                [v1 as *const _ as *mut _, v2 as *const _ as *mut _, v3 as *const _ as *mut _],
            _ => unreachable!(),
        };
        let len = v.len();

        // nth(i)
        for i in 0..len {
            assert_eq!(&mut v[i] as *mut _, v_ptrs[i]); // check the v_ptrs array, just to be sure
            let nth = v.iter_mut().nth(i).unwrap();
            assert_eq!(nth as *mut _, v_ptrs[i]);
        }
        assert_eq!(v.iter().nth(len), None, "nth(len) should return None");

        // stepping through with nth(0)
        {
            let mut it = v.iter();
            for i in 0..len {
                let next = it.nth(0).unwrap();
                assert_eq!(next as *const _, v_ptrs[i]);
            }
            assert_eq!(it.nth(0), None);
        }

        // next()
        {
            let mut it = v.iter_mut();
            for i in 0..len {
                let remaining = len - i;
                assert_eq!(it.size_hint(), (remaining, Some(remaining)));

                let next = it.next().unwrap();
                assert_eq!(next as *mut _, v_ptrs[i]);
            }
            assert_eq!(it.size_hint(), (0, Some(0)));
            assert_eq!(it.next(), None, "The final call to next() should return None");
        }

        // next_back()
        {
            let mut it = v.iter_mut();
            for i in 0..len {
                let remaining = len - i;
                assert_eq!(it.size_hint(), (remaining, Some(remaining)));

                let prev = it.next_back().unwrap();
                assert_eq!(prev as *mut _, v_ptrs[remaining - 1]);
            }
            assert_eq!(it.size_hint(), (0, Some(0)));
            assert_eq!(it.next_back(), None, "The final call to next_back() should return None");
        }
    }

    // Make sure iterators and slice patterns yield consistent addresses for various types,
    // including ZSTs.
    test(0u32);
    test(());
    test([0u32; 0]); // ZST with alignment > 0
    test_mut(0u32);
    test_mut(());
    test_mut([0u32; 0]); // ZST with alignment > 0
}

fn uninit_slice() {
    let mut values = Box::<[Box<u32>]>::new_uninit_slice(3);

    let values = unsafe {
        // Deferred initialization:
        values[0].as_mut_ptr().write(Box::new(1));
        values[1].as_mut_ptr().write(Box::new(2));
        values[2].as_mut_ptr().write(Box::new(3));

        values.assume_init()
    };

    assert_eq!(values.iter().map(|x| **x).collect::<Vec<_>>(), vec![1, 2, 3])
}

/// Regression tests for slice methods in the Rust core library where raw pointers are obtained
/// from mutable references.
fn test_for_invalidated_pointers() {
    let mut buffer = [0usize; 64];
    let len = buffer.len();

    // These regression tests (indirectly) call every slice method which contains a `buffer.as_mut_ptr()`.
    // `<[T]>::as_mut_ptr(&mut self)` takes a mutable reference (tagged Unique), which will invalidate all
    // the other pointers that were previously derived from it according to the Stacked Borrows model.
    // An example of where this could go wrong is a prior bug inside `<[T]>::copy_within`:
    //
    //      unsafe {
    //          core::ptr::copy(self.as_ptr().add(src_start), self.as_mut_ptr().add(dest), count);
    //      }
    //
    // The arguments to `core::ptr::copy` are evaluated from left to right. `self.as_ptr()` creates
    // an immutable reference (which is tagged as `SharedReadOnly` by Stacked Borrows) to the array
    // and derives a valid `*const` pointer from it. When jumping to the next argument,
    // `self.as_mut_ptr()` creates a mutable reference (tagged as `Unique`) to the array, which
    // invalidates the existing `SharedReadOnly` reference and any pointers derived from it.
    // The invalidated `*const` pointer (the first argument to `core::ptr::copy`) is then used
    // after the fact when `core::ptr::copy` is called, which triggers undefined behavior.

    unsafe {
        assert_eq!(0, *buffer.as_mut_ptr_range().start);
    }
    // Check that the pointer range is in-bounds, while we're at it
    let range = buffer.as_mut_ptr_range();
    unsafe {
        assert_eq!(*range.start, *range.end.sub(len));
    }

    buffer.reverse();

    // Calls `fn as_chunks_unchecked_mut` internally:
    assert_eq!(2, buffer.as_chunks_mut::<32>().0.len());
    for chunk in buffer.as_chunks_mut::<32>().0 {
        for elem in chunk {
            *elem += 1;
        }
    }

    // Calls `fn split_at_mut_unchecked` internally:
    let split_mut = buffer.split_at_mut(32);
    assert_eq!(split_mut.0, split_mut.1);

    // Calls `fn partition_dedup_by` internally (requires unstable `#![feature(slice_partition_dedup)]`):
    let partition_dedup = buffer.partition_dedup();
    assert_eq!(1, partition_dedup.0.len());
    partition_dedup.0[0] += 1;
    for elem in partition_dedup.1 {
        *elem += 1;
    }

    buffer.rotate_left(8);
    buffer.rotate_right(16);

    buffer.copy_from_slice(&[1usize; 64]);
    buffer.swap_with_slice(&mut [2usize; 64]);

    assert_eq!(0, unsafe { buffer.align_to_mut::<u8>().1[1] });

    buffer.copy_within(1.., 0);
}

fn large_raw_slice() {
    let size = isize::MAX as usize;
    // Creating a raw slice of size isize::MAX and asking for its size is okay.
    let s = std::ptr::slice_from_raw_parts(ptr::without_provenance::<u8>(1), size);
    assert_eq!(size, unsafe { std::mem::size_of_val_raw(s) });
}

fn main() {
    slice_of_zst();
    test_iter_ref_consistency();
    uninit_slice();
    test_for_invalidated_pointers();
    large_raw_slice();
}
