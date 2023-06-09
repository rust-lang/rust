trait Identity {
    type Identity;
}
impl<T> Identity for T {
    type Identity = T;
}

trait Trait {
    type Assoc: Identity;
    fn tokenize(&self) -> <Self::Assoc as Identity>::Identity;
}

impl Trait for () {
    type Assoc = DoesNotExist;
    //~^ ERROR cannot find type `DoesNotExist` in this scope

    fn tokenize(&self) -> <Self::Assoc as Identity>::Identity {
        unimplemented!()
    }
}

fn main() {}
