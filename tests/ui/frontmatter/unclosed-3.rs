----cargo
//~^ ERROR: frontmatter close does not match the opening

//@ compile-flags: --crate-type lib

// Unfortunate recovery situation. Not really preventable with improving the
// recovery strategy, but this type of code is rare enough already.

 #![feature(frontmatter)]

fn foo(x: i32) -> i32 {
    ---x
    //~^ ERROR: invalid preceding whitespace for frontmatter close
    //~| ERROR: extra characters after frontmatter close are not allowed
}
//~^ ERROR: unexpected closing delimiter: `}`
