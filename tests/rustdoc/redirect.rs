// aux-build:reexp-stripped.rs
// build-aux-docs
// ignore-cross-compile

extern crate reexp_stripped;

pub trait Foo {}

// @has redirect/index.html
// @has - '//code' 'pub use reexp_stripped::Bar'
// @has - '//code/a' 'Bar'
// @has - '//a[@href="../reexp_stripped/hidden/struct.Bar.html"]' 'Bar'
// @has reexp_stripped/hidden/struct.Bar.html
// @has 'reexp_stripped/struct.Bar.html'
// @has - '//a[@href="struct.Bar.html"]' 'Bar'
#[doc(no_inline)]
pub use reexp_stripped::Bar;
impl Foo for Bar {}

// @has redirect/index.html
// @has - '//code' 'pub use reexp_stripped::Quz'
// @has - '//code/a' 'Quz'
// @has reexp_stripped/private/struct.Quz.html
// @has - '//p/a' '../../reexp_stripped/struct.Quz.html'
// @has 'reexp_stripped/struct.Quz.html'
#[doc(no_inline)]
pub use reexp_stripped::Quz;
impl Foo for Quz {}

mod private_no_inline {
    pub struct Qux;
    impl ::Foo for Qux {}
}

// @has redirect/index.html
// @has - '//code' 'pub use private_no_inline::Qux'
// @!has - '//a' 'Qux'
// @!has redirect/struct.Qux.html
#[doc(no_inline)]
pub use private_no_inline::Qux;
