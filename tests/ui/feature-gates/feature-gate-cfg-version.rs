#[cfg(version(42))] //~ ERROR: expected a version literal
//~^ ERROR `cfg(version)` is experimental and subject to change
fn foo() {}
#[cfg(version(1.20))] //~ ERROR: expected a version literal
//~^ ERROR `cfg(version)` is experimental and subject to change
fn foo() -> bool { true }
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
#[cfg(version("foo"))] //~ WARNING: unknown version literal format
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("999"))] //~ WARNING: unknown version literal format
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("-1"))] //~ WARNING: unknown version literal format
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("65536"))] //~ WARNING: unknown version literal format
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("0"))] //~ WARNING: unknown version literal format
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool { true }
#[cfg(version("1.0"))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool { true }
#[cfg(version("1.65536.2"))] //~ WARNING: unknown version literal format
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("1.20.0-stable"))] //~ WARNING: unknown version literal format
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() {}

fn main() {
    assert!(foo());
    assert!(bar());
    assert!(cfg!(version("1.42"))); //~ ERROR `cfg(version)` is experimental and subject to change
}
