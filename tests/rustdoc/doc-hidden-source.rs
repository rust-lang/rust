// Test for <https://github.com/rust-lang/rust/issues/137342>.

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ !has - '//*[@id="main-content"]//*[@class="struct"]' 'Bar'
#[doc(hidden)]
pub struct Bar;

//@ !has - '//*' 'pub use crate::Bar as A;'
pub use crate::Bar as A;
//@ !has - '//*' 'pub use crate::A as B;'
pub use crate::A as B;
//@ has - '//dt/a[@class="struct"]' 'C'
#[doc(inline)]
pub use crate::Bar as C;
