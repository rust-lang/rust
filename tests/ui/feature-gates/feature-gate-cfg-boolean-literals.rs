#[cfg(true)] //~ ERROR `cfg(true)` is experimental
fn foo() {}

#[cfg_attr(true, cfg(false))] //~ ERROR `cfg(true)` is experimental
//~^ ERROR `cfg(false)` is experimental
fn foo() {}

fn main() {
    cfg!(false); //~ ERROR `cfg(false)` is experimental
}
