// Fix for <https://github.com/rust-lang/rust/issues/137508>.

trait Tr {
    type Item;
}

fn main() {
    let _: dyn Tr + ?Foo<Assoc = ()>;
    //~^ ERROR: cannot find trait `Foo` in this scope
    //~| ERROR: `?Trait` is not permitted in trait object types
}
