#[cfg(target(os = "linux"))] //~ ERROR compact `cfg(target(..))` is experimental
struct Foo(u64, u64);

#[cfg_attr(target(os = "linux"), non_exhaustive)] //~ ERROR compact `cfg(target(..))` is experimental
struct Bar(u64, u64);

#[cfg(not(any(all(target(os = "linux")))))] //~ ERROR compact `cfg(target(..))` is experimental
fn foo() {}

fn main() {
    cfg!(target(os = "linux"));
    //~^ ERROR compact `cfg(target(..))` is experimental and subject to change
}
