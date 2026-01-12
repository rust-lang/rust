// Regression test for <https://github.com/rust-lang/rust/issues/111064>.
#![feature(no_core, lang_items)]
#![no_core]
#![crate_name = "foo"]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

//@ files "foo" "['sidebar-items.js', 'all.html', 'hidden', 'index.html', 'struct.Bar.html', \
//        'visible', 'trait.Sized.html', 'trait.MetaSized.html', 'trait.PointeeSized.html']"
//@ files "foo/hidden" "['inner']"
//@ files "foo/hidden/inner" "['trait.Foo.html']"
//@ files "foo/visible" "['index.html', 'sidebar-items.js', 'trait.Foo.html']"

//@ !has 'foo/hidden/index.html'
//@ !has 'foo/hidden/inner/index.html'
// FIXME: Should be `@!has`: https://github.com/rust-lang/rust/issues/111249
//@ has 'foo/hidden/inner/trait.Foo.html'
//@ matchesraw - '<meta http-equiv="refresh" content="0;URL=../../../foo/visible/trait.Foo.html">'
#[doc(hidden)]
pub mod hidden {
    pub mod inner {
        pub trait Foo {
            /// Hello, world!
            fn test();
        }
    }
}

//@ has 'foo/visible/index.html'
//@ has 'foo/visible/trait.Foo.html'
#[doc(inline)]
pub use hidden::inner as visible;

//@ has 'foo/struct.Bar.html'
//@ count - '//*[@id="impl-Foo-for-Bar"]' 1
pub struct Bar;

impl visible::Foo for Bar {
    fn test() {}
}
