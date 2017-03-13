fn main() {
    const LENGTH: usize = 10;
    let mut v: [u64; LENGTH] = [0; LENGTH];

    for idx in 0..LENGTH {
        assert_eq!(v[idx], 0);
    }

    unsafe {
        let p = v.as_mut_ptr();
        ::std::ptr::write_bytes(p, 0xab, LENGTH);
    }

    for idx in 0..LENGTH {
        assert_eq!(v[idx], 0xabababababababab);
    }
}
