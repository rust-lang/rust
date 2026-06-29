//! Regression test for <https://github.com/rust-lang/rust/issues/149287>
//! and <https://github.com/rust-lang/rust/issues/158155>

#![crate_name = "foo"]
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

pub trait Tr {
    type const SIZE: usize;
}

//@ has 'foo/fn.mk_array.html'
//@ has - '//pre[@class="rust item-decl"]/code' '[(); <T as Tr>::SIZE]'
pub fn mk_array<T: Tr>() -> [(); <T as Tr>::SIZE] {
    [(); T::SIZE]
}

//@ has 'foo/type.Arr.html'
//@ has - '//pre[@class="rust item-decl"]/code' '[(); <T as Tr>::SIZE]'
pub type Arr<T> = [(); <T as Tr>::SIZE];
