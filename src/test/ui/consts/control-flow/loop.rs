// Ensure that loops are forbidden in a const context unless `#![feature(const_loop)]` is enabled.
// `while` loops require `#![feature(const_if_match)]` to be enabled as well.

// gate-test-const_loop
// revisions: stock if_match loop_ both

#![cfg_attr(any(both, if_match), feature(const_if_match))]
#![cfg_attr(any(both, loop_), feature(const_loop))]

const _: () = loop {}; //[stock,if_match]~ ERROR `loop` is not allowed in a `const`

static FOO: i32 = loop { break 4; }; //[stock,if_match]~ ERROR `loop` is not allowed in a `static`

const fn foo() {
    loop {} //[stock,if_match]~ ERROR `loop` is not allowed in a `const fn`
}

pub trait Foo {
    const BAR: i32 = loop { break 4; }; //[stock,if_match]~ ERROR `loop` is not allowed in a `const`
}

impl Foo for () {
    const BAR: i32 = loop { break 4; }; //[stock,if_match]~ ERROR `loop` is not allowed in a `const`
}

fn non_const_outside() {
    const fn const_inside() {
        loop {} //[stock,if_match]~ ERROR `loop` is not allowed in a `const fn`
    }
}

const fn const_outside() {
    fn non_const_inside() {
        loop {}
    }
}

fn main() {
    let x = [0; {
        while false {}
        //[stock,if_match,loop_]~^ ERROR `while` is not allowed in a `const`
        4
    }];
}

const _: i32 = {
    let mut x = 0;

    while x < 4 { //[stock,if_match,loop_]~ ERROR `while` is not allowed in a `const`
        x += 1;
    }

    while x < 8 { //[stock,if_match,loop_]~ ERROR `while` is not allowed in a `const`
        x += 1;
    }

    x
};

const _: i32 = {
    let mut x = 0;

    for i in 0..4 { //[stock,if_match,loop_,both]~ ERROR `for` is not allowed in a `const`
        x += i;
    }

    for i in 0..4 { //[stock,if_match,loop_,both]~ ERROR `for` is not allowed in a `const`
        x += i;
    }

    x
};

const _: i32 = {
    let mut x = 0;

    loop { //[stock,if_match]~ ERROR `loop` is not allowed in a `const`
        x += 1;
        if x == 4 { //[stock,loop_]~ ERROR `if` is not allowed in a `const`
            break;
        }
    }

    loop { //[stock,if_match]~ ERROR `loop` is not allowed in a `const`
        x += 1;
        if x == 8 { //[stock,loop_]~ ERROR `if` is not allowed in a `const`
            break;
        }
    }

    x
};

const _: i32 = {
    let mut x = 0;
    while let None = Some(x) { } //[stock,if_match,loop_]~ ERROR `while` is not allowed in a `const`
    while let None = Some(x) { } //[stock,if_match,loop_]~ ERROR `while` is not allowed in a `const`
    x
};
