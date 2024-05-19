pub trait Foo<RHS=Self> {
    type Assoc;
}

pub trait Bar: Foo<Assoc=()> {
    //~^ ERROR: the size for values of type `Self` cannot be known
    //~| ERROR: the size for values of type `Self` cannot be known
    fn new(&self, b: &
           dyn Bar //~ ERROR the trait `Bar` cannot be made into an object
              <Assoc=()>
    );
}

fn main() {}
