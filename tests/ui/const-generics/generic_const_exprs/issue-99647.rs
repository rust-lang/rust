// edition:2018
// build-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

#[allow(unused)]
async fn foo<'a>() {
    let _data = &mut [0u8; { 1 + 4 }];
    bar().await
}

async fn bar() {}

fn main() {}
