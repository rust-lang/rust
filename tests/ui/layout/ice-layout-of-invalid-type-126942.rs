// Regression test for #126942 and #127804

// Tests that we do not ICE when a projection
// type cannot be fully normalized

struct Thing;

pub trait Every {
    type Assoc;
}
impl<T: ?Sized> Every for Thing {
//~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
    type Assoc = T;
}

 // The type of this static normalizes to `Infer`
 // thanks to the `?Sized` constraint in the impl above
static I: <Thing as Every>::Assoc = 3;
//~^ ERROR type annotations needed

fn foo(_: <Thing as Every>::Assoc) {}
//~^ ERROR type annotations needed
//~| ERROR type annotations needed

fn main() {}
