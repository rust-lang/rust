#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// FIXME(fmease): Properly render inherent projections.

// @has inherent_projections/fn.create.html
// @has - '//pre[@class="rust item-decl"]' "create() -> _"
pub fn create() -> Owner::Metadata {}

pub struct Owner;

impl Owner {
    pub type Metadata = ();
}
