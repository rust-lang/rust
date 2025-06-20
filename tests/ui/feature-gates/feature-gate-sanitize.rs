//@ normalize-stderr: "you are using [0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?( \([^)]*\))?" -> "you are using $$RUSTC_VERSION"
#![feature(no_sanitize)] //~ ERROR feature has been removed

#[sanitize(address = "on")]
//~^ ERROR the `#[sanitize]` attribute is an experimental feature
fn main() {
}
