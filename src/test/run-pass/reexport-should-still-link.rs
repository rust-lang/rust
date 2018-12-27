// aux-build:reexport-should-still-link.rs

// pretty-expanded FIXME #23616

extern crate reexport_should_still_link as foo;

pub fn main() {
    foo::bar();
}
