#![feature(const_if_match)]

// `x` is *not* always moved into the final value may be dropped inside the initializer.
const _: Option<Vec<i32>> = {
    let y: Option<Vec<i32>> = None;
    let x = Some(Vec::new());
    //~^ ERROR destructors cannot be evaluated at compile-time

    if true {
        x
    } else {
        y
    }
};

// We only clear `NeedsDrop` if a local is moved from in entirely. This is a shortcoming of the
// existing analysis.
const _: Vec<i32> = {
    let vec_tuple = (Vec::new(),);
    //~^ ERROR destructors cannot be evaluated at compile-time

    vec_tuple.0
};

// This applies to single-field enum variants as well.
const _: Vec<i32> = {
    let x: Result<_, Vec<i32>> = Ok(Vec::new());
    //~^ ERROR destructors cannot be evaluated at compile-time

    match x {
        Ok(x) | Err(x) => x,
    }
};

fn main() {}
