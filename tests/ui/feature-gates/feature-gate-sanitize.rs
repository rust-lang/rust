#![feature(no_sanitize)] //~ ERROR feature was renamed

#[sanitize(address = "on")]
//~^ ERROR the `#[sanitize]` attribute is an experimental feature
fn main() {
}
