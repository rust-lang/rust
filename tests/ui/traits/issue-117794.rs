trait Foo {}

trait T {
    fn a(&self) -> impl Foo {
        self.b(|| 0)
        //~^ ERROR no method named `b` found for reference `&Self` in the current scope
    }
}

fn main() {}
