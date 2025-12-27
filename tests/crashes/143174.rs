//@ known-bug: rust-lang/rust#143174
static FOO: &(u8, ) = &(BAR, );

unsafe extern "C" {
    static BAR: u8;
}
