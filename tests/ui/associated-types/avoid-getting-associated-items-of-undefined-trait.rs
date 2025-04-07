// Fix for <https://github.com/rust-lang/rust/issues/137508>.

trait Tr {
    type Item;
}

fn main() {
    let _: dyn Tr + ?Foo<Assoc = ()>;
    //~^ ERROR: `?Trait` is not permitted in trait object types
    //~| ERROR: cannot find trait `Foo` in this scope
    //~| ERROR: the value of the associated type `Item` in `Tr` must be specified
}
