#![feature(no_sanitize)] //~ ERROR feature has been removed

#[sanitize(address = "on")]
//~^ ERROR the `#[sanitize]` attribute is an experimental feature
fn main() {
}
