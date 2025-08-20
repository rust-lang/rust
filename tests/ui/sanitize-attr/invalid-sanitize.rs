#![feature(sanitize)]

#[sanitize(brontosaurus = "off")] //~ ERROR invalid argument
fn main() {
}

#[sanitize(address = "off")] //~ ERROR multiple `sanitize` attributes
#[sanitize(address = "off")]
fn multiple_consistent() {}

#[sanitize(address = "on")] //~ ERROR multiple `sanitize` attributes
#[sanitize(address = "off")]
fn multiple_inconsistent() {}

#[sanitize(address = "bogus")] //~ ERROR invalid argument for `sanitize`
fn wrong_value() {}

#[sanitize = "off"] //~ ERROR malformed `sanitize` attribute input
fn name_value () {}

#[sanitize] //~ ERROR malformed `sanitize` attribute input
fn just_word() {}
