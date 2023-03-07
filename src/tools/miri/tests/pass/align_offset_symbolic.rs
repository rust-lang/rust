//@compile-flags: -Zmiri-symbolic-alignment-check
#![feature(strict_provenance)]

use std::ptr;

fn test_align_offset() {
    let d = Box::new([0u32; 4]);
    // Get u8 pointer to base
    let raw = d.as_ptr() as *const u8;

    assert_eq!(raw.align_offset(2), 0);
    assert_eq!(raw.align_offset(4), 0);
    assert_eq!(raw.align_offset(8), usize::MAX); // requested alignment higher than allocation alignment

    assert_eq!(raw.wrapping_offset(1).align_offset(2), 1);
    assert_eq!(raw.wrapping_offset(1).align_offset(4), 3);
    assert_eq!(raw.wrapping_offset(1).align_offset(8), usize::MAX); // requested alignment higher than allocation alignment

    assert_eq!(raw.wrapping_offset(2).align_offset(2), 0);
    assert_eq!(raw.wrapping_offset(2).align_offset(4), 2);
    assert_eq!(raw.wrapping_offset(2).align_offset(8), usize::MAX); // requested alignment higher than allocation alignment

    let p = ptr::invalid::<()>(1);
    assert_eq!(p.align_offset(1), 0);
}

fn test_align_to() {
    const N: usize = 4;
    let d = Box::new([0u32; N]);
    // Get u8 slice covering the entire thing
    let s = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, 4 * N) };
    let raw = s.as_ptr();

    {
        let (l, m, r) = unsafe { s.align_to::<u32>() };
        assert_eq!(l.len(), 0);
        assert_eq!(r.len(), 0);
        assert_eq!(m.len(), N);
        assert_eq!(raw, m.as_ptr() as *const u8);
    }

    {
        let (l, m, r) = unsafe { s[1..].align_to::<u32>() };
        assert_eq!(l.len(), 3);
        assert_eq!(m.len(), N - 1);
        assert_eq!(r.len(), 0);
        assert_eq!(raw.wrapping_offset(4), m.as_ptr() as *const u8);
    }

    {
        let (l, m, r) = unsafe { s[..4 * N - 1].align_to::<u32>() };
        assert_eq!(l.len(), 0);
        assert_eq!(m.len(), N - 1);
        assert_eq!(r.len(), 3);
        assert_eq!(raw, m.as_ptr() as *const u8);
    }

    {
        let (l, m, r) = unsafe { s[1..4 * N - 1].align_to::<u32>() };
        assert_eq!(l.len(), 3);
        assert_eq!(m.len(), N - 2);
        assert_eq!(r.len(), 3);
        assert_eq!(raw.wrapping_offset(4), m.as_ptr() as *const u8);
    }

    {
        #[repr(align(8))]
        struct Align8(u64);

        let (l, m, r) = unsafe { s.align_to::<Align8>() }; // requested alignment higher than allocation alignment
        assert_eq!(l.len(), 4 * N);
        assert_eq!(r.len(), 0);
        assert_eq!(m.len(), 0);
    }
}

fn test_from_utf8() {
    const N: usize = 10;
    let vec = vec![0x4141414141414141u64; N];
    let content = unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const u8, 8 * N) };
    println!("{:?}", std::str::from_utf8(content).unwrap());
}

fn test_u64_array() {
    #[derive(Default)]
    #[repr(align(8))]
    struct AlignToU64<T>(T);

    const BYTE_LEN: usize = std::mem::size_of::<[u64; 4]>();
    type Data = AlignToU64<[u8; BYTE_LEN]>;

    fn example(data: &Data) {
        let (head, u64_arrays, tail) = unsafe { data.0.align_to::<[u64; 4]>() };

        assert!(head.is_empty(), "buffer was not aligned for 64-bit numbers");
        assert_eq!(u64_arrays.len(), 1, "buffer was not long enough");
        assert!(tail.is_empty(), "buffer was too long");

        let u64_array = &u64_arrays[0];
        let _val = u64_array[0]; // make sure we can actually load this
    }

    example(&Data::default());
}

fn main() {
    test_align_offset();
    test_align_to();
    test_from_utf8();
    test_u64_array();
}
