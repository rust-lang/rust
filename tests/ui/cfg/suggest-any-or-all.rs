#[cfg(foo, bar)]
//~^ ERROR malformed `cfg` attribute input
fn f() {}

#[cfg()]
//~^ ERROR malformed `cfg` attribute input
struct Foo;

fn main() {}
