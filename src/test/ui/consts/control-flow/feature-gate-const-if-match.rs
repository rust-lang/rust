// Ensure that `if`, `if let` and `match` are only allowed in the various const contexts when
// `#![feature(const_if_match)]` is enabled. When the feature gate is removed, the `#[rustc_error]`
// on `main` should be removed and this test converted to `check-pass`.

// revisions: stock if_match

#![feature(rustc_attrs)]
#![cfg_attr(if_match, feature(const_if_match))]

const _: i32 = if true { //[stock]~ ERROR `if` is not allowed in a `const`
    5
} else {
    6
};

const _: i32 = if let Some(true) = Some(false) { //[stock]~ ERROR `if` is not allowed in a `const`
    0
} else {
    1
};

const _: i32 = match 1 { //[stock]~ ERROR `match` is not allowed in a `const`
    2 => 3,
    4 => 5,
    _ => 0,
};

static FOO: i32 = {
    let x = if true { 0 } else { 1 };
    //[stock]~^ ERROR `if` is not allowed in a `static`
    let x = match x { 0 => 1, _ => 0 };
    //[stock]~^ ERROR `match` is not allowed in a `static`
    if let Some(x) = Some(x) { x } else { 1 }
    //[stock]~^ ERROR `if` is not allowed in a `static`
};

static mut BAR: i32 = {
    let x = if true { 0 } else { 1 };
    //[stock]~^ ERROR `if` is not allowed in a `static mut`
    let x = match x { 0 => 1, _ => 0 };
    //[stock]~^ ERROR `match` is not allowed in a `static mut`
    if let Some(x) = Some(x) { x } else { 1 }
    //[stock]~^ ERROR `if` is not allowed in a `static mut`
};

const fn if_() -> i32 {
    if true { 5 } else { 6 } //[stock]~ ERROR `if` is not allowed in a `const fn`
}

const fn if_let(a: Option<bool>) -> i32 {
    if let Some(true) = a { //[stock]~ ERROR `if` is not allowed in a `const fn`
        0
    } else {
        1
    }
}

const fn match_(i: i32) -> i32 {
    match i { //[stock]~ ERROR `match` is not allowed in a `const fn`
        i if i > 10 => i,
        1 => 2,
        _ => 0
    }
}

pub trait Foo {
    const IF: i32 = if true { 5 } else { 6 };
    //[stock]~^ ERROR `if` is not allowed in a `const`

    const IF_LET: i32 = if let Some(true) = None { 5 } else { 6 };
    //[stock]~^ ERROR `if` is not allowed in a `const`

    const MATCH: i32 = match 0 { 1 => 2, _ => 0 };
    //[stock]~^ ERROR `match` is not allowed in a `const`
}

impl Foo for () {
    const IF: i32 = if true { 5 } else { 6 };
    //[stock]~^ ERROR `if` is not allowed in a `const`

    const IF_LET: i32 = if let Some(true) = None { 5 } else { 6 };
    //[stock]~^ ERROR `if` is not allowed in a `const`

    const MATCH: i32 = match 0 { 1 => 2, _ => 0 };
    //[stock]~^ ERROR `match` is not allowed in a `const`
}

fn non_const_outside() {
    const fn const_inside(y: bool) -> i32 {
        let x = if y { 0 } else { 1 };
        //[stock]~^ ERROR `if` is not allowed in a `const fn`
        let x = match x { 0 => 1, _ => 0 };
        //[stock]~^ ERROR `match` is not allowed in a `const fn`
        if let Some(x) = Some(x) { x } else { 1 }
        //[stock]~^ ERROR `if` is not allowed in a `const fn`
    }
}

const fn const_outside() {
    fn non_const_inside(y: bool) -> i32 {
        let x = if y { 0 } else { 1 };
        let x = match x { 0 => 1, _ => 0 };
        if let Some(x) = Some(x) { x } else { 1 }
    }
}

#[rustc_error]
fn main() { //[if_match]~ ERROR fatal error triggered by #[rustc_error]
    let _ = [0; {
        let x = if false { 0 } else { 1 };
        //[stock]~^ ERROR `if` is not allowed in a `const`
        let x = match x { 0 => 1, _ => 0 };
        //[stock]~^ ERROR `match` is not allowed in a `const`
        if let Some(x) = Some(x) { x } else { 1 }
        //[stock]~^ ERROR `if` is not allowed in a `const`
    }];
}
