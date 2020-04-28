#[cfg(version("1.44"))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn foo() -> bool { true }
#[cfg(not(version("1.44")))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn foo() -> bool { false }

#[cfg(version("1.43", "1.44", "1.45"))] //~ ERROR: expected single version literal
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version(false))] //~ ERROR: expected a version literal
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("foo"))] //~ ERROR: invalid version literal
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("999"))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("999"))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("0"))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool { true }

fn main() {
    assert!(foo());
    assert!(bar());
    assert!(cfg!(version("1.42"))); //~ ERROR `cfg(version)` is experimental and subject to change
}
