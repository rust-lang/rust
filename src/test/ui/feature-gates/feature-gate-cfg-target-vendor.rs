#[cfg(target_vendor = "x")] //~ ERROR `cfg(target_vendor)` is experimental
#[cfg_attr(target_vendor = "x", x)] //~ ERROR `cfg(target_vendor)` is experimental
struct Foo(u64, u64);

#[cfg(not(any(all(target_vendor = "x"))))] //~ ERROR `cfg(target_vendor)` is experimental
fn foo() {}

fn main() {
    cfg!(target_vendor = "x");
    //~^ ERROR `cfg(target_vendor)` is experimental and subject to change
}
