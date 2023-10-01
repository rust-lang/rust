#![allow(dead_code)]
#![allow(clippy::uninlined_format_args)]

async fn sink1<'a>(_: &'a str) {} // lint
//~^ ERROR: the following explicit lifetimes could be elided: 'a
//~| NOTE: `-D clippy::needless-lifetimes` implied by `-D warnings`
async fn sink1_elided(_: &str) {} // ok

// lint
async fn one_to_one<'a>(s: &'a str) -> &'a str {
    //~^ ERROR: the following explicit lifetimes could be elided: 'a
    s
}

// ok
async fn one_to_one_elided(s: &str) -> &str {
    s
}

// ok
async fn all_to_one<'a>(a: &'a str, _b: &'a str) -> &'a str {
    a
}

// async fn unrelated(_: &str, _: &str) {} // Not allowed in async fn

// #3988
struct Foo;
impl Foo {
    // ok
    pub async fn new(&mut self) -> Self {
        //~^ ERROR: methods called `new` usually take no `self`
        Foo {}
    }
}

// rust-lang/rust#61115
// ok
async fn print(s: &str) {
    println!("{}", s);
}

fn main() {}
