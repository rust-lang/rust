// This test checks that calling `.clone()` on a type that does not implement the `Clone` trait
// results in a compilation error. The `Foo` struct does not derive or implement `Clone`,
// so attempting to clone it should fail.

struct Foo {
  i: isize,
}

fn foo(i:isize) -> Foo {
    Foo {
        i: i
    }
}

fn main() {
    let x = foo(10);
    let _y = x.clone();
    //~^ ERROR no method named `clone` found
}
