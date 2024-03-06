//@ edition: 2021
//@ compile-flags: -Zunpretty=expanded

trait Trait {
    async fn method() {}
}

fn foo<T: Trait<method(i32): Send>>() {}
//~^ ERROR associated type bounds are unstable

fn main() {}
