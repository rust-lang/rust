#![crate_name = "foo"]
#![feature(no_core)]
#![no_core]

// @files "foo" "['all.html', 'visible', 'index.html', 'sidebar-items.js', 'hidden', \
//        'struct.Bar.html']"
// @files "foo/visible" "['trait.Foo.html', 'index.html', 'sidebar-items.js']"
// @files "foo/hidden" "['inner']"
// @files "foo/hidden/inner" "['trait.Foo.html']"

// The following five should not fail!
// @!has 'foo/hidden/index.html'
// @!has 'foo/hidden/inner/index.html'
// FIXME: Should be `@!has`: https://github.com/rust-lang/rust/issues/111249
// @has 'foo/hidden/inner/trait.Foo.html'
// @matchesraw - '<meta http-equiv="refresh" content="0;URL=../../../foo/visible/trait.Foo.html">'
// @!has 'foo/hidden/inner/inner_hidden/index.html'
// @!has 'foo/hidden/inner/inner_hidden/trait.HiddenFoo.html'
#[doc(hidden)]
pub mod hidden {
    pub mod inner {
        pub trait Foo {}

        #[doc(hidden)]
        pub mod inner_hidden {
            pub trait HiddenFoo {}
        }
    }
}

// @has 'foo/visible/index.html'
// @has 'foo/visible/trait.Foo.html'
#[doc(inline)]
pub use hidden::inner as visible;

// @has 'foo/struct.Bar.html'
// @count - '//*[@id="impl-Foo-for-Bar"]' 1
pub struct Bar;

impl visible::Foo for Bar {}
