//@ known-bug: #117629
//@ edition:2021

#![feature(const_trait_impl)]

const trait Tr {
    async fn ft1() {}
}

fn main() {}
