//@ run-rustfix

// See issue #78168.

#![feature(inline_const_pat)]

const fn one() -> i32 {
    1
}

fn foo() -> i32 {
    let x = 2;

    match x {
        const 2 => {}
        //~^ ERROR expected `{`, found `2`
        //~| HELP you might have meant to write this as part of a block
        _ => {}
    }

    match x {
        const 1 + 2 * 3 / 4 => {}
        //~^ ERROR expected `{`, found `1`
        //~| HELP you might have meant to write this as part of a block
        _ => {}
    }

    match x {
        const one() => {}
        //~^ ERROR expected `{`, found `one`
        //~| HELP you might have meant to write this as part of a block
        _ => {}
    }

    x
}

fn bar() -> i32 {
    let x = const 2;
    //~^ ERROR expected `{`, found `2`
    //~| HELP you might have meant to write this as part of a block

    x
}

fn baz() -> i32 {
    let y = const 1 + 2 * 3 / 4;
    //~^ ERROR expected `{`, found `1`
    //~| HELP you might have meant to write this as part of a block

    y
}

fn main() {
    foo();
    bar();
    baz();
}
