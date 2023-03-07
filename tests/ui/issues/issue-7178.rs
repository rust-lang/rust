// run-pass
// aux-build:issue-7178.rs

// pretty-expanded FIXME #23616

extern crate issue_7178 as cross_crate_self;

pub fn main() {
    let _ = cross_crate_self::Foo::new(&1);
}
