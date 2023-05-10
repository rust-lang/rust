#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// @has 'inherent_projections/fn.create.html'
// @has - '//pre[@class="rust item-decl"]' "create() -> Owner::Metadata"
// @has - '//pre[@class="rust item-decl"]//a[@class="associatedtype"]/@href' 'struct.Owner.html#associatedtype.Metadata'
pub fn create() -> Owner::Metadata {}

pub struct Owner;

impl Owner {
    pub type Metadata = ();
}

// Make sure we handle bound vars correctly.
// @has 'inherent_projections/type.User.html' '//pre[@class="rust item-decl"]' "for<'a> fn(_: Carrier<'a>::Focus)"
pub type User = for<'a> fn(Carrier<'a>::Focus);

pub struct Carrier<'a>(&'a ());

impl<'a> Carrier<'a> {
    pub type Focus = &'a mut i32;
}

////////////////////////////////////////

// FIXME(inherent_associated_types): Below we link to `Proj` but we should link to `Proj-1`.
// The current test checks for the buggy behavior for demonstration purposes.

// @has 'inherent_projections/type.Test.html'
// @has - '//pre[@class="rust item-decl"]' "Parametrized<i32>"
// @has - '//pre[@class="rust item-decl"]//a[@class="associatedtype"]/@href' 'struct.Parametrized.html#associatedtype.Proj'
// @!has - '//pre[@class="rust item-decl"]//a[@class="associatedtype"]/@href' 'struct.Parametrized.html#associatedtype.Proj-1'
pub type Test = Parametrized<i32>::Proj;

pub struct Parametrized<T>(T);

impl Parametrized<bool> {
    pub type Proj = ();
}

impl Parametrized<i32> {
    pub type Proj = String;
}
