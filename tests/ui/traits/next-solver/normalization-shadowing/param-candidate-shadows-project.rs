//@ compile-flags: -Znext-solver

trait Foo {
    type Assoc;
}

trait Bar {}

impl<T> Foo for T {
    type Assoc = i32;
}

impl<T> Bar for T where T: Foo<Assoc = i32> {}

fn require_bar<T: Bar>() {}

fn foo<T: Foo>() {
    // Unlike the classic solver, the new solver previously projected
    // `<T as Foo>::Assoc = _` down to `i32` even though there's a param-env
    // candidate here, since we don't assemble any param-env projection
    // candidates for `T: Foo` alone.
    //
    // However, allowing impl candidates shadowed by env candidates results
    // in multiple issues, so we explicitly hide them, e.g.
    //
    //     https://github.com/rust-lang/trait-system-refactor-initiative/issues/76
    require_bar::<T>();
    //~^ ERROR type mismatch resolving `<T as Foo>::Assoc == i32`
}

fn main() {}
