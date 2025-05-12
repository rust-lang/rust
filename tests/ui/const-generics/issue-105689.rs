//@ check-pass
//@ edition:2021
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#[allow(unused)]
async fn foo<'a>() {
    let _data = &mut [0u8; { 1 + 4 }];
    bar().await
}

async fn bar() {}

fn main() {}
