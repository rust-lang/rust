trait Trait {}

struct Foo<T:Trait> {
    x: T,
}

enum Bar<T:Trait> {
    ABar(isize),
    BBar(T),
    CBar(usize),
}

trait PolyTrait<T>
{
    fn whatever(&self, t: T) {}
}

struct Struct;

impl PolyTrait<Foo<u16>> for Struct {
//~^ ERROR E0277
}

fn main() {
}
