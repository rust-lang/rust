//@ edition:2018
#![deny(unused_must_use)]


#[must_use]
async fn foo() -> i32 {
    1
}

#[must_use]
fn bar() -> impl std::future::Future<Output=i32> {
    async {
        42
    }
}

async fn baz() -> i32 {
    0
}

struct Wowee {}

impl Wowee {
    #[must_use]
    async fn test_method() -> i32 {
        1
    }
}

async fn test() {
    foo(); //~ ERROR unused return value of `foo` that must be used
    //~^ ERROR unused implementer of `Future` that must be used
    foo().await; //~ ERROR unused output of future returned by `foo` that must be used
    bar(); //~ ERROR unused return value of `bar` that must be used
    //~^ ERROR unused implementer of `Future` that must be used
    bar().await; // ok, it's not an async fn
    baz(); //~ ERROR unused implementer of `Future` that must be used
    baz().await; // ok
}

trait Doer {
    #[must_use]
    async fn test_trait_method() -> i32;
    async fn test_other_trait() -> i32;
}

impl Doer for Wowee {
    async fn test_trait_method() -> i32 {
        1
    }
    #[must_use]
    //~^  WARN: the `must_use` attribute cannot be used on trait methods in impl blocks [unused_attributes]
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    async fn test_other_trait() -> i32 {
        1
    }
}

fn main() {
}
