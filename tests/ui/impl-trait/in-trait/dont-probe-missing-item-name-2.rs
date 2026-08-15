trait Foo {
    fn rpitit() -> impl Sized;
}

// Ensure that we don't try to probe the name of the RPITIT when looking for
// fixes to suggest for the redundant generic below.

fn test<T: Foo<i32, Assoc = i32>>() {}
//~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied
//~| ERROR associated type `Assoc` not found for `Foo`

fn main() {}
