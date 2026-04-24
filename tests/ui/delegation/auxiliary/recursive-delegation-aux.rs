#![feature(fn_delegation)]

fn foo() {}

reuse foo as bar;
pub reuse bar as goo;
