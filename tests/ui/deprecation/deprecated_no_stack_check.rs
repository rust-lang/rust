//@ normalize-stderr: "you are using [0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?( \([^)]*\))?" -> "you are using $$RUSTC_VERSION"

#![deny(warnings)]
#![feature(no_stack_check)]
//~^ ERROR: feature has been removed [E0557]
fn main() {

}
