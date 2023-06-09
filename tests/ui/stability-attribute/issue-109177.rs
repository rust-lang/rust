// aux-build: similar-unstable-method.rs

extern crate similar_unstable_method;

fn main() {
    // FIXME: this function should not suggest the `foo` function.
    similar_unstable_method::foo1();
    //~^ ERROR cannot find function `foo1` in crate `similar_unstable_method` [E0425]

    let foo = similar_unstable_method::Foo;
    foo.foo1();
    //~^ ERROR no method named `foo1` found for struct `Foo` in the current scope [E0599]
}
