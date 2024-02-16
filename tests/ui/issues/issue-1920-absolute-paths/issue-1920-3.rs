//! Test that when a crate is linked multiple times that the shortest absolute path name is used

//@ aux-build:issue-1920.rs

mod foo {
    pub extern crate issue_1920;
}

extern crate issue_1920;

fn assert_clone<T>() where T : Clone { }

fn main() {
    assert_clone::<foo::issue_1920::S>();
    //~^ ERROR `S: Clone` is not satisfied
}
