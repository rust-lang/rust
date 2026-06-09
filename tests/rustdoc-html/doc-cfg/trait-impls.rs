// This test ensures that `doc_cfg` feature is working as expected on trait impls.
// Regression test for <https://github.com/rust-lang/rust/issues/153655>.

#![feature(doc_cfg)]
#![doc(auto_cfg(hide(
    target_pointer_width = "64",
)))]

#![crate_name = "foo"]

pub trait Trait {
    fn f(&self) {}
}

pub trait Bob {
    fn bob(&self) {}
}

pub trait Foo {
    fn foo(&self) {}
}

pub struct X;

//@has 'foo/struct.X.html'
//@count - '//*[@id="impl-Bob-for-X"]' 1
//@count - '//*[@id="impl-Bob-for-X"]/*[@class="item-info"]' 0
//@count - '//*[@id="impl-Trait-for-X"]' 1
//@count - '//*[@id="impl-Trait-for-X"]/*[@class="item-info"]' 0

// If you need to update this XPath, in particular `item-info`, update all
// the others in this file.
//@count - '//*[@id="impl-Foo-for-X"]/*[@class="item-info"]' 1

//@has 'foo/trait.Trait.html'
//@count - '//*[@id="impl-Trait-for-X"]' 1
//@count - '//*[@id="impl-Trait-for-X"]/*[@class="item-info"]' 0
#[cfg(any(target_pointer_width = "64", target_arch = "wasm32"))]
#[doc(auto_cfg(hide(target_arch = "wasm32")))]
mod imp {
    impl super::Trait for super::X { fn f(&self) {} }
}

//@has 'foo/trait.Bob.html'
//@count - '//*[@id="impl-Bob-for-X"]' 1
//@count - '//*[@id="impl-Bob-for-X"]/*[@class="item-info"]' 0
#[cfg(any(target_pointer_width = "64", target_arch = "wasm32"))]
#[doc(auto_cfg = false)]
mod imp2 {
    impl super::Bob for super::X { fn bob(&self) {} }
}

//@has 'foo/trait.Foo.html'
//@count - '//*[@id="impl-Foo-for-X"]/*[@class="item-info"]' 1
// We use this to force xpath tests to be updated if `item-info` class is changed.
#[cfg(any(target_pointer_width = "64", target_arch = "wasm32"))]
mod imp3 {
    impl super::Foo for super::X { fn foo(&self) {} }
}

pub struct Y;

//@has 'foo/struct.Y.html'
//@count - '//*[@id="implementations-list"]/*[@class="impl-items"]' 1
//@count - '//*[@id="implementations-list"]/*[@class="impl-items"]/*[@class="item-info"]' 0
#[cfg(any(target_pointer_width = "64", target_arch = "wasm32"))]
#[doc(auto_cfg(hide(target_arch = "wasm32")))]
mod imp4 {
    impl super::Y { pub fn plain_auto() {} }
}

pub struct Z;

//@has 'foo/struct.Z.html'
//@count - '//*[@id="implementations-list"]/*[@class="impl-items"]' 1
//@count - '//*[@id="implementations-list"]/*[@class="impl-items"]/*[@class="item-info"]' 0
#[cfg(any(target_pointer_width = "64", target_arch = "wasm32"))]
#[doc(auto_cfg = false)]
mod imp5 {
    impl super::Z { pub fn plain_auto() {} }
}

// The "witness" which has the item info.
pub struct W;

//@has 'foo/struct.W.html'
//@count - '//*[@id="implementations-list"]/*[@class="impl-items"]' 1
//@count - '//*[@id="implementations-list"]/*[@class="impl-items"]/*[@class="item-info"]' 1
#[cfg(any(target_pointer_width = "64", target_arch = "wasm32"))]
mod imp6 {
    impl super::W { pub fn plain_auto() {} }
}
