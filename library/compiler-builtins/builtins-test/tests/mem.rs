extern crate compiler_builtins;
use compiler_builtins::mem::{memcmp, memcpy, memmove, memset};

const WORD_SIZE: usize = core::mem::size_of::<usize>();

#[test]
fn memcpy_3() {
    let mut arr: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    unsafe {
        let src = arr.as_ptr().offset(9);
        let dst = arr.as_mut_ptr().offset(1);
        assert_eq!(memcpy(dst, src, 3), dst);
        assert_eq!(arr, [0, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11]);
    }
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    unsafe {
        let src = arr.as_ptr().offset(1);
        let dst = arr.as_mut_ptr().offset(9);
        assert_eq!(memcpy(dst, src, 3), dst);
        assert_eq!(arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]);
    }
}

#[test]
fn memcpy_10() {
    let arr: [u8; 18] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let mut dst: [u8; 12] = [0; 12];
    unsafe {
        let src = arr.as_ptr().offset(1);
        assert_eq!(memcpy(dst.as_mut_ptr(), src, 10), dst.as_mut_ptr());
        assert_eq!(dst, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0]);
    }
    unsafe {
        let src = arr.as_ptr().offset(8);
        assert_eq!(memcpy(dst.as_mut_ptr(), src, 10), dst.as_mut_ptr());
        assert_eq!(dst, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 0]);
    }
}

#[test]
fn memcpy_big() {
    // Make the arrays cross 3 pages
    const SIZE: usize = 8193;
    let src: [u8; SIZE] = [22; SIZE];
    struct Dst {
        start: usize,
        buf: [u8; SIZE],
        end: usize,
    }

    let mut dst = Dst {
        start: 0,
        buf: [0; SIZE],
        end: 0,
    };
    unsafe {
        assert_eq!(
            memcpy(dst.buf.as_mut_ptr(), src.as_ptr(), SIZE),
            dst.buf.as_mut_ptr()
        );
        assert_eq!(dst.start, 0);
        assert_eq!(dst.buf, [22; SIZE]);
        assert_eq!(dst.end, 0);
    }
}

#[test]
fn memmove_forward() {
    let mut arr: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    unsafe {
        let src = arr.as_ptr().offset(6);
        let dst = arr.as_mut_ptr().offset(3);
        assert_eq!(memmove(dst, src, 5), dst);
        assert_eq!(arr, [0, 1, 2, 6, 7, 8, 9, 10, 8, 9, 10, 11]);
    }
}

#[test]
fn memmove_backward() {
    let mut arr: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    unsafe {
        let src = arr.as_ptr().offset(3);
        let dst = arr.as_mut_ptr().offset(6);
        assert_eq!(memmove(dst, src, 5), dst);
        assert_eq!(arr, [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 11]);
    }
}

#[test]
fn memset_zero() {
    let mut arr: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    unsafe {
        let ptr = arr.as_mut_ptr().offset(5);
        assert_eq!(memset(ptr, 0, 2), ptr);
        assert_eq!(arr, [0, 1, 2, 3, 4, 0, 0, 7]);

        // Only the LSB matters for a memset
        assert_eq!(memset(arr.as_mut_ptr(), 0x2000, 8), arr.as_mut_ptr());
        assert_eq!(arr, [0, 0, 0, 0, 0, 0, 0, 0]);
    }
}

#[test]
fn memset_nonzero() {
    let mut arr: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    unsafe {
        let ptr = arr.as_mut_ptr().offset(2);
        assert_eq!(memset(ptr, 22, 3), ptr);
        assert_eq!(arr, [0, 1, 22, 22, 22, 5, 6, 7]);

        // Only the LSB matters for a memset
        assert_eq!(memset(arr.as_mut_ptr(), 0x2009, 8), arr.as_mut_ptr());
        assert_eq!(arr, [9, 9, 9, 9, 9, 9, 9, 9]);
    }
}

#[test]
fn memcmp_eq() {
    let arr1 @ arr2 = gen_arr::<256>();
    for i in 0..256 {
        unsafe {
            assert_eq!(memcmp(arr1.0.as_ptr(), arr2.0.as_ptr(), i), 0);
            assert_eq!(memcmp(arr2.0.as_ptr(), arr1.0.as_ptr(), i), 0);
        }
    }
}

#[test]
fn memcmp_ne() {
    let arr1 @ arr2 = gen_arr::<256>();
    // Reduce iteration count in Miri as it is too slow otherwise.
    let limit = if cfg!(miri) { 64 } else { 256 };
    for i in 0..limit {
        let mut diff_arr = arr1;
        diff_arr.0[i] = 127;
        let expect = diff_arr.0[i].cmp(&arr2.0[i]);
        for k in i + 1..limit {
            let result = unsafe { memcmp(diff_arr.0.as_ptr(), arr2.0.as_ptr(), k) };
            assert_eq!(expect, result.cmp(&0));
        }
    }
}

#[derive(Clone, Copy)]
struct AlignedStorage<const N: usize>([u8; N], [usize; 0]);

fn gen_arr<const N: usize>() -> AlignedStorage<N> {
    let mut ret = AlignedStorage::<N>([0; N], []);
    for i in 0..N {
        ret.0[i] = i as u8;
    }
    ret
}

#[test]
fn memmove_forward_misaligned_nonaligned_start() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let src = arr.0.as_ptr().offset(6);
        let dst = arr.0.as_mut_ptr().offset(3);
        assert_eq!(memmove(dst, src, 17), dst);
        reference.0.copy_within(6..6 + 17, 3);
        assert_eq!(arr.0, reference.0);
    }
}

#[test]
fn memmove_forward_misaligned_aligned_start() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let src = arr.0.as_ptr().offset(6);
        let dst = arr.0.as_mut_ptr().add(0);
        assert_eq!(memmove(dst, src, 17), dst);
        reference.0.copy_within(6..6 + 17, 0);
        assert_eq!(arr.0, reference.0);
    }
}

#[test]
fn memmove_forward_aligned() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let src = arr.0.as_ptr().add(3 + WORD_SIZE);
        let dst = arr.0.as_mut_ptr().add(3);
        assert_eq!(memmove(dst, src, 17), dst);
        reference
            .0
            .copy_within(3 + WORD_SIZE..3 + WORD_SIZE + 17, 3);
        assert_eq!(arr.0, reference.0);
    }
}

#[test]
fn memmove_backward_misaligned_nonaligned_start() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let src = arr.0.as_ptr().offset(3);
        let dst = arr.0.as_mut_ptr().offset(6);
        assert_eq!(memmove(dst, src, 17), dst);
        reference.0.copy_within(3..3 + 17, 6);
        assert_eq!(arr.0, reference.0);
    }
}

#[test]
fn memmove_backward_misaligned_aligned_start() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let src = arr.0.as_ptr().offset(3);
        let dst = arr.0.as_mut_ptr().add(WORD_SIZE);
        assert_eq!(memmove(dst, src, 17), dst);
        reference.0.copy_within(3..3 + 17, WORD_SIZE);
        assert_eq!(arr.0, reference.0);
    }
}

#[test]
fn memmove_backward_aligned() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let src = arr.0.as_ptr().add(3);
        let dst = arr.0.as_mut_ptr().add(3 + WORD_SIZE);
        assert_eq!(memmove(dst, src, 17), dst);
        reference.0.copy_within(3..3 + 17, 3 + WORD_SIZE);
        assert_eq!(arr.0, reference.0);
    }
}

#[test]
fn memmove_misaligned_bounds() {
    // The above test have the downside that the addresses surrounding the range-to-copy are all
    // still in-bounds, so Miri would not actually complain about OOB accesses. So we also test with
    // an array that has just the right size. We test a few times to avoid it being accidentally
    // aligned.
    for _ in 0..8 {
        let mut arr1 = [0u8; 17];
        let mut arr2 = [0u8; 17];
        unsafe {
            // Copy both ways so we hit both the forward and backward cases.
            memmove(arr1.as_mut_ptr(), arr2.as_mut_ptr(), 17);
            memmove(arr2.as_mut_ptr(), arr1.as_mut_ptr(), 17);
        }
    }
}

#[test]
fn memset_backward_misaligned_nonaligned_start() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let ptr = arr.0.as_mut_ptr().offset(6);
        assert_eq!(memset(ptr, 0xCC, 17), ptr);
        core::ptr::write_bytes(reference.0.as_mut_ptr().add(6), 0xCC, 17);
        assert_eq!(arr.0, reference.0);
    }
}

#[test]
fn memset_backward_misaligned_aligned_start() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let ptr = arr.0.as_mut_ptr().add(WORD_SIZE);
        assert_eq!(memset(ptr, 0xCC, 17), ptr);
        core::ptr::write_bytes(reference.0.as_mut_ptr().add(WORD_SIZE), 0xCC, 17);
        assert_eq!(arr.0, reference.0);
    }
}

#[test]
fn memset_backward_aligned() {
    let mut arr = gen_arr::<32>();
    let mut reference = arr;
    unsafe {
        let ptr = arr.0.as_mut_ptr().add(3 + WORD_SIZE);
        assert_eq!(memset(ptr, 0xCC, 17), ptr);
        core::ptr::write_bytes(reference.0.as_mut_ptr().add(3 + WORD_SIZE), 0xCC, 17);
        assert_eq!(arr.0, reference.0);
    }
}
