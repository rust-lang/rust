// compile-flags: -Ztrait-solver=next
// check-pass

// Checks that we don't explode when we assemble >1 candidate for a goal.

struct Wrapper<T>(T);

trait Foo {}

impl Foo for Wrapper<i32> {}

impl Foo for Wrapper<()> {}

fn needs_foo(_: impl Foo) {}

fn main() {
    let mut x = Default::default();
    let w = Wrapper(x);
    needs_foo(w);
    x = 1;
    drop(x);
}
