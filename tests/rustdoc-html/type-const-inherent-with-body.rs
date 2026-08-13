#![crate_name = "foo"]
#![feature(min_generic_const_args, macroless_generic_const_args, inherent_associated_types)]
#![expect(incomplete_features)]

pub struct Foo;

impl Foo {
    type const LEN: usize = 4;
}

//@ has 'foo/fn.mk_array.html'
//@ has - '//pre[@class="rust item-decl"]/code' '[u8; Foo::LEN]'
pub fn mk_array() -> [u8; Foo::LEN] {
    [0u8; Foo::LEN]
}
