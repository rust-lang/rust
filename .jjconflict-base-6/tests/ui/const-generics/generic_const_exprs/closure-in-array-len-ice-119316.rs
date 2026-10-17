//@ edition: 2018
// regression test for #119316
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

async fn foo<'a>() {
    let _data = &mut [0u8; { N + (|| 42)() }];
    //~^ ERROR cannot find value `N` in this scope
}

fn main() {}
