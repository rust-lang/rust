//! This test used to ICE during the normalization of
//! `I`'s type, because of the mismatch of generic parameters
//! on the impl with the generic parameters the projection can
//! fulfill.

struct Thing;

pub trait Every {
    type Assoc;
}
impl<T: ?Sized> Every for Thing {
    //~^ ERROR: `T` is not constrained
    type Assoc = T;
}

static I: <Thing as Every>::Assoc = 3;

fn main() {}
