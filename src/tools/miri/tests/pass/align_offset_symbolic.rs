//@compile-flags: -Zmiri-symbolic-alignment-check

use std::mem;

fn test_align_to() {
    const N: usize = 4;
    let d = Box::new([0u32; N]);
    // Get u8 slice covering the entire thing
    let s = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, 4 * N) };
    let raw = s.as_ptr();

    // Cases where we get the expected "middle" part without any fuzz, since the allocation is
    // 4-aligned.
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

    // Cases where we request more alignment than the allocation has.
    {
        #[repr(align(8))]
        #[derive(Copy, Clone)]
        struct Align8(#[allow(dead_code)] u64);

        let (_l, m, _r) = unsafe { s.align_to::<Align8>() };
        assert!(m.len() > 0);
        // Ensure the symbolic alignment check has been informed that this is okay now.
        let _val = m[0];
    }
}

fn test_from_utf8() {
    // uses `align_offset` internally
    const N: usize = 10;
    let vec = vec![0x4141414141414141u64; N];
    let content = unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const u8, 8 * N) };
    assert_eq!(
        std::str::from_utf8(content).unwrap(),
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    );
}

fn test_u64_array() {
    #[derive(Default)]
    #[repr(align(8))]
    struct AlignToU64<T>(T);

    const BYTE_LEN: usize = mem::size_of::<[u64; 4]>();
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

fn test_cstr() {
    // uses `align_offset` internally
    std::ffi::CStr::from_bytes_with_nul(b"this is a test that is longer than 16 bytes\0").unwrap();
}

fn huge_align() {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 1 << 47;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 1 << 30;
    #[cfg(target_pointer_width = "16")]
    const SIZE: usize = 1 << 13;
    struct HugeSize(#[allow(dead_code)] [u8; SIZE - 1]);
    let _ = std::ptr::without_provenance::<HugeSize>(SIZE).align_offset(SIZE);
}

// This shows that we cannot store the promised alignment info in `AllocExtra`,
// since vtables do not have an `AllocExtra`.
fn vtable() {
    #[cfg(target_pointer_width = "64")]
    type TWOPTR = u128;
    #[cfg(target_pointer_width = "32")]
    type TWOPTR = u64;

    let ptr: &dyn Send = &0;
    let parts: (*const (), *const u8) = unsafe { mem::transmute(ptr) };
    let vtable = parts.1;
    let offset = vtable.align_offset(mem::align_of::<TWOPTR>());
    let vtable_aligned = vtable.wrapping_add(offset) as *const [TWOPTR; 0];
    // Zero-sized deref, so no in-bounds requirement.
    let _place = unsafe { &*vtable_aligned };
}

fn main() {
    test_align_to();
    test_from_utf8();
    test_u64_array();
    test_cstr();
    huge_align();
    vtable();
}
