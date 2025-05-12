//@ compile-flags: -Znext-solver

// When we're solving `<T as Foo>::Assoc = i32`, we actually first solve
// `<T as Foo>::Assoc = ?1t`, then unify `?1t` with `i32`. That goal
// with the inference variable is ambiguous when there are >1 param-env
// candidates.

// We don't unify the RHS of a projection goal eagerly when solving, both
// for caching reasons and partly to make sure that we don't make the new
// trait solver smarter than it should be.

// This is (as far as I can tell) a forwards-compatible decision, but if you
// make this test go from fail to pass, be sure you understand the implications!

trait Foo {
    type Assoc;
}

trait Bar {}

impl<T> Bar for T where T: Foo<Assoc = i32> {}

fn needs_bar<T: Bar>() {}

fn foo<T: Foo<Assoc = i32> + Foo<Assoc = u32>>() {
    needs_bar::<T>();
    //~^ ERROR type annotations needed: cannot normalize
}

fn main() {}
