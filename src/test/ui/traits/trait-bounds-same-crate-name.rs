// aux-build:crate_a1.rs
// aux-build:crate_a2.rs

// Issue 22750
// This tests the extra help message reported when a trait bound
// is not met but the struct implements a trait with the same path.

fn main() {
    let foo2 = {
        extern crate crate_a2 as a;
        a::Foo
    };

    {
        extern crate crate_a1 as a;
        a::try_foo(foo2);
        //~^ ERROR E0277
        //~| Trait impl with same name found
        //~| Perhaps two different versions of crate `crate_a2`
    }
}
