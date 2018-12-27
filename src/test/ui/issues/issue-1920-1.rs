//! Test that absolute path names are correct when a crate is not linked into the root namespace

// aux-build:issue-1920.rs

mod foo {
    pub extern crate issue_1920;
}

fn assert_clone<T>() where T : Clone { }

fn main() {
    assert_clone::<foo::issue_1920::S>();
    //~^ ERROR `foo::issue_1920::S: std::clone::Clone` is not satisfied
}
