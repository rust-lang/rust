#[cfg(target = "x")] //~ ERROR `cfg(target)` is experimental
struct Foo(u64, u64);

#[cfg_attr(target = "x", x)] //~ ERROR `cfg(target)` is experimental
struct Bar(u64, u64);

#[cfg(not(any(all(target = "x"))))] //~ ERROR `cfg(target)` is experimental
fn foo() {}

fn main() {
    cfg!(target = "x");
    //~^ ERROR `cfg(target)` is experimental and subject to change
}
