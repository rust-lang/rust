// run-pass

#![feature(const_if_match)]

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

fn main() {}
