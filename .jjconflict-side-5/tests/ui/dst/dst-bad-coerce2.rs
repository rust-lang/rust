// Attempt to change the mutability as well as unsizing.

struct Fat<T: ?Sized> {
    ptr: T
}

struct Foo;
trait Bar {}
impl Bar for Foo {}

pub fn main() {
    // With a vec of ints.
    let f1 = Fat { ptr: [1, 2, 3] };
    let f2: &Fat<[isize; 3]> = &f1;
    let f3: &mut Fat<[isize]> = f2; //~ ERROR mismatched types

    // With a trait.
    let f1 = Fat { ptr: Foo };
    let f2: &Fat<Foo> = &f1;
    let f3: &mut Fat<dyn Bar> = f2; //~ ERROR mismatched types

    // Tuple with a vec of ints.
    let f1 = ([1, 2, 3],);
    let f2: &([isize; 3],) = &f1;
    let f3: &mut ([isize],) = f2; //~ ERROR mismatched types

    // Tuple with a trait.
    let f1 = (Foo,);
    let f2: &(Foo,) = &f1;
    let f3: &mut (dyn Bar,) = f2; //~ ERROR mismatched types
}
