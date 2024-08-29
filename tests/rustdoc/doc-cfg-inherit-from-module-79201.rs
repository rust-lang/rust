// https://github.com/rust-lang/rust/issues/79201
#![crate_name="foo"]

#![feature(doc_cfg)]

//@ has 'foo/trait.Foo.html'
//@ count   - '//*[@class="stab portability"]' 6
//@ matches - '//*[@class="stab portability"]' 'crate feature foo-root'
//@ matches - '//*[@class="stab portability"]' 'crate feature foo-public-mod'
//@ matches - '//*[@class="stab portability"]' 'crate feature foo-private-mod'
//@ matches - '//*[@class="stab portability"]' 'crate feature foo-fn'
//@ matches - '//*[@class="stab portability"]' 'crate feature foo-method'

pub trait Foo {}

#[doc(cfg(feature = "foo-root"))]
impl crate::Foo for usize {}

#[doc(cfg(feature = "foo-public-mod"))]
pub mod public {
    impl crate::Foo for u8 {}
}

#[doc(cfg(feature = "foo-private-mod"))]
mod private {
    impl crate::Foo for u16 {}
}

#[doc(cfg(feature = "foo-const"))]
const _: () = {
    impl crate::Foo for u32 {}
};

#[doc(cfg(feature = "foo-fn"))]
fn __() {
    impl crate::Foo for u64 {}
}

#[doc(cfg(feature = "foo-method"))]
impl dyn Foo {
    fn __() {
        impl crate::Foo for u128 {}
    }
}
