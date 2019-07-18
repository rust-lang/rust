// compile-flags: --edition 2018
#![feature(async_await)]
#![allow(dead_code)]

async fn sink1<'a>(_: &'a str) {} // lint
async fn sink1_elided(_: &str) {} // ok

async fn one_to_one<'a>(s: &'a str) -> &'a str { s } // lint
async fn one_to_one_elided(s: &str) -> &str { s } // ok
async fn all_to_one<'a>(a: &'a str, _b: &'a str) -> &'a str { a } // ok
// async fn unrelated(_: &str, _: &str) {} // Not allowed in async fn

// #3988
struct Foo;
impl Foo {
    pub async fn foo(&mut self) {} // ok
}

// rust-lang/rust#61115
async fn print(s: &str) { // ok
    println!("{}", s);
}

fn main() {}
