// compile-flags: -Ztrait-solver=next
// check-pass

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
    // Unlike the classic solver, `<T as Foo>::Assoc = _` will still project
    // down to `i32` even though there's a param-env candidate here, since we
    // don't assemble any param-env projection candidates for `T: Foo` alone.
    require_bar::<T>();
}

fn main() {}
