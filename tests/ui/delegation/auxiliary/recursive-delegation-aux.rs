#![feature(fn_delegation)]
#![allow(incomplete_features)]

fn foo() {}

reuse foo as bar;
pub reuse bar as goo;
