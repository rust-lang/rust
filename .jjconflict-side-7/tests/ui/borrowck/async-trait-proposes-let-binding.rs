// Regression test for <https://github.com/rust-lang/rust/issues/119686>.
//@ edition: 2024
struct A;
pub trait Trait1 {
    async fn func() -> ();
}

impl Trait1 for A {
    async fn func() -> () {
        let p = std::convert::identity(&("".to_string()));
        //~^ ERROR temporary value dropped while borrowed
        let _q = p;
    }
}

fn main() {}
