#[cfg(version(42))] //~ ERROR: expected a version literal
fn foo() {}
#[cfg(version(1.20))] //~ ERROR: expected a version literal
fn foo() -> bool { true }
#[cfg(version = "1.20")] //~ WARN: unexpected `cfg` condition name: `version`
fn foo() -> bool { true }
#[cfg(version("1.44"))]
fn foo() -> bool { true }
#[cfg(not(version("1.44")))]
fn foo() -> bool { false }

#[cfg(version("1.43", "1.44", "1.45"))] //~ ERROR: expected single version literal
fn bar() -> bool  { false }
#[cfg(version(false))] //~ ERROR: expected a version literal
fn bar() -> bool  { false }
#[cfg(version("foo"))] //~ WARNING: unknown version literal format
fn bar() -> bool  { false }
#[cfg(version("999"))] //~ WARNING: unknown version literal format
fn bar() -> bool  { false }
#[cfg(version("-1"))] //~ WARNING: unknown version literal format
fn bar() -> bool  { false }
#[cfg(version("65536"))] //~ WARNING: unknown version literal format
fn bar() -> bool  { false }
#[cfg(version("0"))] //~ WARNING: unknown version literal format
fn bar() -> bool { true }
#[cfg(version("1.0"))]
fn bar() -> bool { true }
#[cfg(version("1.65536.2"))] //~ WARNING: unknown version literal format
fn bar() -> bool  { false }
#[cfg(version("1.20.0-stable"))] //~ WARNING: unknown version literal format
fn bar() {}

fn main() {
    assert!(foo());
    assert!(bar());
    assert!(cfg!(version("1.42")));
}
