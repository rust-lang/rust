#![feature(sanitize)]

#[sanitize(brontosaurus = "off")] //~ ERROR malformed `sanitize` attribute input
fn main() {}

#[sanitize(address = "off")] //~ ERROR multiple `sanitize` attributes
#[sanitize(address = "off")]
fn multiple_consistent() {}

#[sanitize(address = "on")] //~ ERROR multiple `sanitize` attributes
#[sanitize(address = "off")]
fn multiple_inconsistent() {}

#[sanitize(address = "bogus")] //~ ERROR malformed `sanitize` attribute input
fn wrong_value() {}

#[sanitize = "off"] //~ ERROR malformed `sanitize` attribute input
fn name_value() {}

#[sanitize] //~ ERROR malformed `sanitize` attribute input
fn just_word() {}
