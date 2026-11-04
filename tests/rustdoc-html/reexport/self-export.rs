// Because of the `use val::Val::{self}`, the path build always kept the last element
// of the path instead of `Val`, making it displayed as `self` instead of `Val`.
// Regression test for <https://github.com/rust-lang/rust/issues/165317>.

#![crate_name = "foo"]

use val::Val::{self};

pub mod val {
    pub enum Val {}
}

//@ has 'foo/type.Result.html'
//@ has - '//*[@class="rust item-decl"]' 'pub type Result<T = Val, E = ()> = Result<T, E>;'
// Not the same "item-decl". This one is the "Aliased type".
//@ has - '//*[@class="rust item-decl"]' 'pub enum Result<T = Val, E = ()> {'
pub type Result<T = Val, E = ()> = std::result::Result<T, E>;

//@ has 'foo/type.X.html' '//*[@class="rust item-decl"]' 'pub type X = Val;'
pub type X = Val;

//@ has 'foo/enum.Y.html' '//*[@class="rust item-decl"]' 'Y(Val)'
pub enum Y {
    Y(Val),
}

//@ has 'foo/fn.z.html' '//*[@class="rust item-decl"]' 'pub fn z(v: Val)'
pub fn z(v: Val) {}
