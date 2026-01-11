//@ known-bug: #149809
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct Qux<'a> {
    x: &'a (),
}
impl<'a> Qux<'a> {
    #[type_const]
    const LEN: usize = 4;
    fn foo(_: [u8; Qux::LEN]) {}
}

fn main() {}
