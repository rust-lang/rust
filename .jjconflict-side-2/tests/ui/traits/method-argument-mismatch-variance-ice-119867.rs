trait Deserialize {
    fn deserialize(&self);
}

struct ArchivedVec<T>(T);

impl<T> Deserialize for ArchivedVec<T> {
    fn deserialize(s: _) {}
    //~^ ERROR: `_` is not allowed within types on item signatures
    //~| ERROR: has a `&self` declaration in the trait, but not in the impl
}

fn main() {}
