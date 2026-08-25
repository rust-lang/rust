---cargo
//~^ ERROR: unclosed frontmatter

//@ compile-flags: --crate-type lib

#![feature(frontmatter)]

fn foo(x: i32) -> i32 {
    ---x
     //~^ WARNING: use of a double negation [double_negations]
}

// this test is for the weird case that valid Rust code can have three dashes
// within them and get treated as a frontmatter close.
