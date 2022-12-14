#![crate_name = "foo"]

// This test ensures we don't display anonymous (non-inline) re-exports of public items.

// @has 'foo/index.html'
// @has - '//*[@id="main-content"]' ''
// We check that the only "h2" present is for "Bla".
// @count - '//*[@id="main-content"]/h2' 1
// @has - '//*[@id="main-content"]/h2' 'Structs'
// @count - '//*[@id="main-content"]//a[@class="struct"]' 1

mod ext {
    pub trait Foo {}
    pub trait Bar {}
    pub struct S;
}

pub use crate::ext::Foo as _;
pub use crate::ext::Bar as _;
pub use crate::ext::S as _;

pub struct Bla;
