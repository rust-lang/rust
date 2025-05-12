//@ edition:2018

trait From {
    fn from();
}

impl From for () {
    fn from() {}
}

impl From for () {
//~^ ERROR conflicting implementations of trait
    fn from() {}
}

fn bar() -> impl core::future::Future<Output = ()> {
    async move { From::from() }
}

fn main() {}
