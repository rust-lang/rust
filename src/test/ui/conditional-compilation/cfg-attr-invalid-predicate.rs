#[cfg(foo(bar))] //~ ERROR invalid predicate `foo`
fn check() {}

fn main() {}
