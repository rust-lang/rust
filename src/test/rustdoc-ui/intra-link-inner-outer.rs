// check-pass

pub enum A {}

/// Links to [outer][A] and [outer][B]
//~^ WARNING: unresolved link to `B`
pub mod M {
    //! Links to [inner][A] and [inner][B]
    //~^ WARNING: unresolved link to `A`

    pub struct B;
}
