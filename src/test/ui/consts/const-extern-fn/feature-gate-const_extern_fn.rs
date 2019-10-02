// Check that `const extern fn` and `const unsafe extern fn` are feature-gated.

#[cfg(FALSE)] const extern fn foo1() {} //~ ERROR `const extern fn` definitions are unstable
#[cfg(FALSE)] const extern "C" fn foo2() {} //~ ERROR `const extern fn` definitions are unstable
#[cfg(FALSE)] const extern "Rust" fn foo3() {} //~ ERROR `const extern fn` definitions are unstable
#[cfg(FALSE)] const unsafe extern fn bar1() {} //~ ERROR `const extern fn` definitions are unstable
#[cfg(FALSE)] const unsafe extern "C" fn bar2() {}
//~^ ERROR `const extern fn` definitions are unstable
#[cfg(FALSE)] const unsafe extern "Rust" fn bar3() {}
//~^ ERROR `const extern fn` definitions are unstable

fn main() {}
