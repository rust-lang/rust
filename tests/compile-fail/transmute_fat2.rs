fn main() {
    #[cfg(target_pointer_width="64")]
    let bad = unsafe {
        std::mem::transmute::<u128, &[u8]>(42)
    };
    #[cfg(target_pointer_width="32")]
    let bad = unsafe {
        std::mem::transmute::<u64, &[u8]>(42)
    };
    bad[0]; //~ ERROR index out of bounds: the len is 0 but the index is 0
}
