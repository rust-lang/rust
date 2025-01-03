// Make sure we don't ICE in `normalize_erasing_regions` when normalizing
// an associated type in an impl with unconstrained non-lifetime params.
// (This time in a function signature)

struct Thing;

pub trait Every {
    type Assoc;
}
impl<T: ?Sized> Every for Thing {
//~^ ERROR the type parameter `T` is not constrained
    type Assoc = T;
}

fn foo(_: <Thing as Every>::Assoc) {}

fn main() {}
