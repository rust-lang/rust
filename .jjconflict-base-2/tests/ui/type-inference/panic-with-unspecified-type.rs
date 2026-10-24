//! regression test for <https://github.com/rust-lang/rust/issues/16966>
//@ edition:2015..2021
fn main() {
    panic!(std::default::Default::default());
    //~^ ERROR type annotations needed
}
