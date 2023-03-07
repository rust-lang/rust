// compile-flags -Wrust-2021-incompatible-closure-captures

fn main() {}

pub(crate) struct Numberer {}

impl Numberer {
    pub(crate) async fn new(
    //~^ ERROR `async fn` is not permitted in Rust 2015
        interval: Duration,
        //~^ ERROR cannot find type `Duration` in this scope
    ) -> Numberer {
        Numberer {}
    }
}
