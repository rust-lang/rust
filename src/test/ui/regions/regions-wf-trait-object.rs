// Check that the explicit lifetime bound (`'b`, in this example) must
// outlive all the superbound from the trait (`'a`, in this example).

trait TheTrait<'t>: 't { }

struct Foo<'a,'b> {
    x: Box<dyn TheTrait<'a>+'b> //~ ERROR E0478
}

fn main() { }
