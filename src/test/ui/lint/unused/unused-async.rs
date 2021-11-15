// edition:2018
// run-pass
#![allow(dead_code)]

#[must_use]
//~^ WARNING `must_use`
async fn test() -> i32 {
    1
}


struct Wowee {}

impl Wowee {
    #[must_use]
    //~^ WARNING `must_use`
    async fn test_method() -> i32 {
        1
    }
}

/* FIXME(guswynn) update this test when async-fn-in-traits works
trait Doer {
    #[must_use]
    async fn test_trait_method() -> i32;
    WARNING must_use
    async fn test_other_trait() -> i32;
}

impl Doer for Wowee {
    async fn test_trait_method() -> i32 {
        1
    }
    #[must_use]
    async fn test_other_trait() -> i32 {
        WARNING must_use
        1
    }
}
*/

fn main() {
}
