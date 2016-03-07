#![feature(plugin)]
#![plugin(clippy, regex_macros)]

#[macro_use]
extern crate regex;

#[deny(mut_mut)]
#[allow(regex_macro)]
fn main() {
    let pattern = regex!(r"^(?P<level>[#]+)\s(?P<title>.+)$");
    assert!(pattern.is_match("# headline"));
}
