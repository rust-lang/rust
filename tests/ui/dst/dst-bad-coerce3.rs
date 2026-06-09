// Attempt to extend the lifetime as well as unsizing.

struct Fat<T: ?Sized> {
    ptr: T
}

struct Foo;
trait Bar { fn bar(&self) {} }
impl Bar for Foo {}

fn baz<'a>() {
    // With a vec of ints.
    let f1 = Fat { ptr: [1, 2, 3] };
    let f2: &Fat<[isize; 3]> = &f1; //~ ERROR `f1` does not live long enough
    let f3: &'a Fat<[isize]> = f2;

    // With a trait.
    let f1 = Fat { ptr: Foo };
    let f2: &Fat<Foo> = &f1; //~ ERROR `f1` does not live long enough
    let f3: &'a Fat<dyn Bar> = f2;
}

pub fn main() {
    baz();
}
