pub trait Foo<RHS=Self> {
    type Assoc;
}

pub trait Bar: Foo<Assoc=()> {
    //~^ ERROR: the size for values of type `Self` cannot be known
    //~| ERROR: the size for values of type `Self` cannot be known
    fn new(&self, b: &
           dyn Bar //~ ERROR the trait `Bar` is not dyn compatible
              <Assoc=()>
    );
}

fn main() {}
