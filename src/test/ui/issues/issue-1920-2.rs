//! Test that when a crate is linked under another name that name is used in global paths

// aux-build:issue-1920.rs

extern crate issue_1920 as bar;

fn assert_clone<T>() where T : Clone { }

fn main() {
    assert_clone::<bar::S>();
    //~^ ERROR `bar::S: std::clone::Clone` is not satisfied
}
