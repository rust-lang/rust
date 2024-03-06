//@ compile-flags: --test

fn align_offset_weird_strides() {
    #[test]
    //~^ ERROR the `#[test]` attribute may only be used on a non-associated function
    struct A5(u32, u8);
}
