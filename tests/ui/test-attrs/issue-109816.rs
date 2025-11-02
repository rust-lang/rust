//@ compile-flags: --test

fn align_offset_weird_strides() {
    #[test]
    //~^ ERROR the `#[test]` attribute may only be used on a free function
    struct A5(u32, u8);
}
