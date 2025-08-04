//@ edition: 2021
#![feature(const_trait_impl)]

const trait Tr {
    async fn ft1() {}
//~^ ERROR async functions are not allowed in `const` traits
}

fn main() {}
