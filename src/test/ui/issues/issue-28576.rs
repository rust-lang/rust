pub trait Foo<RHS=Self> {
    type Assoc;
}

pub trait Bar: Foo<Assoc=()> {
    fn new(&self, b: &
           dyn Bar //~ ERROR the trait `Bar` cannot be made into an object
              <Assoc=()>
    );
}

fn main() {}
