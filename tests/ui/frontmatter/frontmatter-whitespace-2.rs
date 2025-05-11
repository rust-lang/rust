---cargo

//@ compile-flags: --crate-type lib

#![feature(frontmatter)]

fn foo(x: i32) -> i32 {
    ---x
    //~^ ERROR: invalid preceding whitespace for frontmatter close
    //~| ERROR: extra characters after frontmatter close are not allowed
}
//~^ ERROR: unexpected closing delimiter: `}`

// this test is for the weird case that valid Rust code can have three dashes
// within them and get treated as a frontmatter close.
