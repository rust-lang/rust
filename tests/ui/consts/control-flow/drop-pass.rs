//@ run-pass
//@ revisions: stock precise

#![allow(unused)]
#![cfg_attr(precise, feature(const_precise_live_drops))]

// `x` is always moved into the final value and is not dropped inside the initializer.
const _: Option<Vec<i32>> = {
    let y: Option<Vec<i32>> = None;
    let x = Some(Vec::new());

    if true {
        x
    } else {
        x
    }
};

const _: Option<Vec<i32>> = {
    let x = Some(Vec::new());
    match () {
        () => x,
    }
};

const _: Option<Vec<i32>> = {
    let mut some = Some(Vec::new());
    let mut tmp = None;

    let mut i = 0;
    while i < 10 {
        tmp = some;
        some = None;

        // We can never exit the loop with `Some` in `tmp`.

        some = tmp;
        tmp = None;

        i += 1;
    }

    some
};

fn main() {}
