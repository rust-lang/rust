//@ edition: 2021
#![feature(const_trait_impl)]

const trait Tr {
    async fn ft1() {}
//~^ ERROR async functions are not allowed in `const` traits
}

const trait Tr2 {
    fn f() -> impl std::future::Future<Output = ()>;
}

impl const Tr2 for () {
    async fn f() {}
//~^ ERROR async functions are not allowed in `const` trait impls
}

fn main() {}
