#![feature(no_sanitize)] //~ ERROR feature has been removed [E0557]

#[sanitize(address = "on")]
//~^ ERROR the `#[sanitize]` attribute is an experimental feature
fn main() {
}
