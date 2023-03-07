#[cfg(target_abi = "x")] //~ ERROR `cfg(target_abi)` is experimental
struct Foo(u64, u64);

#[cfg_attr(target_abi = "x", x)] //~ ERROR `cfg(target_abi)` is experimental
struct Bar(u64, u64);

#[cfg(not(any(all(target_abi = "x"))))] //~ ERROR `cfg(target_abi)` is experimental
fn foo() {}

fn main() {
    cfg!(target_abi = "x");
    //~^ ERROR `cfg(target_abi)` is experimental and subject to change
}
