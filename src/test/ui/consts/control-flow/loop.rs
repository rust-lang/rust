// Ensure that loops are forbidden in a const context unless `#![feature(const_loop)]` is enabled.

// gate-test-const_loop
// revisions: stock loop_

#![cfg_attr(loop_, feature(const_loop))]

const _: () = loop {}; //[stock]~ ERROR `loop` is not allowed in a `const`

static FOO: i32 = loop { break 4; }; //[stock]~ ERROR `loop` is not allowed in a `static`

const fn foo() {
    loop {} //[stock]~ ERROR `loop` is not allowed in a `const fn`
}

pub trait Foo {
    const BAR: i32 = loop { break 4; }; //[stock]~ ERROR `loop` is not allowed in a `const`
}

impl Foo for () {
    const BAR: i32 = loop { break 4; }; //[stock]~ ERROR `loop` is not allowed in a `const`
}

fn non_const_outside() {
    const fn const_inside() {
        loop {} //[stock]~ ERROR `loop` is not allowed in a `const fn`
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
        //[stock]~^ ERROR `while` is not allowed in a `const`
        4
    }];
}

const _: i32 = {
    let mut x = 0;

    while x < 4 { //[stock]~ ERROR `while` is not allowed in a `const`
        x += 1;
    }

    while x < 8 { //[stock]~ ERROR `while` is not allowed in a `const`
        x += 1;
    }

    x
};

const _: i32 = {
    let mut x = 0;

    for i in 0..4 { //[stock,loop_]~ ERROR `for` is not allowed in a `const`
        x += i;
    }

    for i in 0..4 { //[stock,loop_]~ ERROR `for` is not allowed in a `const`
        x += i;
    }

    x
};

const _: i32 = {
    let mut x = 0;

    loop { //[stock]~ ERROR `loop` is not allowed in a `const`
        x += 1;
        if x == 4 {
            break;
        }
    }

    loop { //[stock]~ ERROR `loop` is not allowed in a `const`
        x += 1;
        if x == 8 {
            break;
        }
    }

    x
};

const _: i32 = {
    let mut x = 0;
    while let None = Some(x) { } //[stock]~ ERROR `while` is not allowed in a `const`
    while let None = Some(x) { } //[stock]~ ERROR `while` is not allowed in a `const`
    x
};
