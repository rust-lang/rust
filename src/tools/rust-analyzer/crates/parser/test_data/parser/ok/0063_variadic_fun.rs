extern "C" {
    fn a(_: *mut u8, ...,);
    fn b(_: *mut u8, _: ...);
    fn c(_: *mut u8, #[cfg(never)] [w, t, f]: ...,);
}
