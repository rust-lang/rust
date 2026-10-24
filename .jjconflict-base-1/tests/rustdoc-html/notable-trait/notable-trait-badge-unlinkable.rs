#![feature(doc_notable_trait)]
#![crate_name = "foo"]

// Doc-hidden traits don't get badges.
#[doc(notable_trait)]
#[doc(hidden)]
pub trait Hidden {}

// Private traits don't get badges.
#[doc(notable_trait)]
trait Private {}
#[doc(notable_trait)]
pub trait Public {}

//@ has 'foo/struct.Foo.html'
//@ count - '//div[@class="notable-trait-badge-container"]' 1
//@ has - '//div[@class="notable-trait-badge-container"]/a[@href="trait.Public.html"]' 'Public'
pub struct Foo;
impl Hidden for Foo {}
impl Private for Foo {}
impl Public for Foo {}
