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
#[cfg(version("-1"))] //~ ERROR: invalid version literal
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("65536"))] //~ ERROR: invalid version literal
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool  { false }
#[cfg(version("0"))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn bar() -> bool { true }

#[cfg(version("1.65536.2"))]
//~^ ERROR `cfg(version)` is experimental and subject to change
fn version_check_bug() {}

fn main() {
    // This should fail but due to a bug in version_check `1.65536.2` is interpreted as `1.2`.
    // See https://github.com/SergioBenitez/version_check/issues/11
    version_check_bug();
    assert!(foo());
    assert!(bar());
    assert!(cfg!(version("1.42"))); //~ ERROR `cfg(version)` is experimental and subject to change
}
