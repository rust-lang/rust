//@ known-bug: #149809
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct Qux<'a> {
    x: &'a (),
}
impl<'a> Qux<'a> {
    const LEN: usize = core::direct_const_arg!(4);
    fn foo(_: [u8; core::direct_const_arg!(Qux::LEN)]) {}
}

fn main() {}
