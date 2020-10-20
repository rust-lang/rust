// run-rustfix

// See issue #78168.

#![allow(incomplete_features)]
#![feature(inline_const)]

// FIXME(#78171): the lint has to be allowed because of a bug
#[allow(dead_code)]
const fn one() -> i32 {
    1
}

fn foo() -> i32 {
    let x = 2;

    match x {
        const 2 => {}
        //~^ ERROR expected `{`, found `2`
        //~| HELP try placing this code inside a block
        _ => {}
    }

    match x {
        const 1 + 2 * 3 / 4 => {}
        //~^ ERROR expected `{`, found `1`
        //~| HELP try placing this code inside a block
        _ => {}
    }

    match x {
        const one() => {}
        //~^ ERROR expected `{`, found `one`
        //~| HELP try placing this code inside a block
        _ => {}
    }

    x
}

fn bar() -> i32 {
    let x = const 2;
    //~^ ERROR expected `{`, found `2`
    //~| HELP try placing this code inside a block

    x
}

fn baz() -> i32 {
    let y = const 1 + 2 * 3 / 4;
    //~^ ERROR expected `{`, found `1`
    //~| HELP try placing this code inside a block

    y
}

fn main() {
    foo();
    bar();
    baz();
}
