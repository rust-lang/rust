fn test_align_offset() {
    let d = Box::new([0u32; 4]);
    // Get u8 pointer to base
    let raw = d.as_ptr() as *const u8;

    assert_eq!(raw.align_offset(2), 0);
    assert_eq!(raw.align_offset(4), 0);
    assert_eq!(raw.align_offset(8), usize::max_value());

    assert_eq!(raw.wrapping_offset(1).align_offset(2), 1);
    assert_eq!(raw.wrapping_offset(1).align_offset(4), 3);
    assert_eq!(raw.wrapping_offset(1).align_offset(8), usize::max_value());

    assert_eq!(raw.wrapping_offset(2).align_offset(2), 0);
    assert_eq!(raw.wrapping_offset(2).align_offset(4), 2);
    assert_eq!(raw.wrapping_offset(2).align_offset(8), usize::max_value());
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
        assert_eq!(m.len(), N-1);
        assert_eq!(r.len(), 0);
        assert_eq!(raw.wrapping_offset(4), m.as_ptr() as *const u8);
    }

    {
        let (l, m, r) = unsafe { s[..4*N - 1].align_to::<u32>() };
        assert_eq!(l.len(), 0);
        assert_eq!(m.len(), N-1);
        assert_eq!(r.len(), 3);
        assert_eq!(raw, m.as_ptr() as *const u8);
    }

    {
        let (l, m, r) = unsafe { s[1..4*N - 1].align_to::<u32>() };
        assert_eq!(l.len(), 3);
        assert_eq!(m.len(), N-2);
        assert_eq!(r.len(), 3);
        assert_eq!(raw.wrapping_offset(4), m.as_ptr() as *const u8);
    }
}

fn test_from_utf8() {
    const N: usize = 10;
    let vec = vec![0x4141414141414141u64; N];
    let content = unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const u8, 8 * N) };
    println!("{:?}", std::str::from_utf8(content).unwrap());
}

fn main() {
    test_align_offset();
    test_align_to();
    test_from_utf8();
}
