#![feature(doc_notable_trait)]
#![crate_name = "foo"]

// Doc-hidden traits don't get badges.
#[doc(notable_trait)]
#[doc(hidden)]
pub trait Hidden {}

// Private traits don't get badges.
#[doc(notable_trait)]
trait Private {}

//@ count 'foo/struct.Foo.html' '//div[@class="notable-trait-badge-container"]' 0
pub struct Foo;
impl Hidden for Foo {}
impl Private for Foo {}
