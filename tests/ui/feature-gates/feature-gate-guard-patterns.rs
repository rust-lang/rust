#![allow(irrefutable_let_patterns)]

fn match_guards_still_work() {
    match 0 {
        0 if guard(0) => {},
        _ => {},
    }
}

fn other_guards_dont() {
    match 0 {
        (0 if guard(0)) => {},
        //~^ ERROR unexpected parentheses surrounding `match` arm pattern
        _ => {},
    }

    match 0 {
        (0 if guard(0)) | 1 => {},
        //~^ ERROR: guard patterns are experimental
        _ => {},
    }

    let ((x if guard(x)) | x) = 0;
    //~^ ERROR: guard patterns are experimental
    //~| ERROR: cannot find value `x`

    if let (x if guard(x)) = 0 {}
    //~^ ERROR: guard patterns are experimental

    while let (x if guard(x)) = 0 {}
    //~^ ERROR: guard patterns are experimental

    #[cfg(false)]
    while let (x if guard(x)) = 0 {}
    //~^ ERROR: guard patterns are experimental
}

fn even_as_function_parameters(((x if guard(x), _) | (_, x)): (i32, i32)) {}
//~^ ERROR: guard patterns are experimental
//~| ERROR: cannot find value `x`

fn guard<T>(x: T) -> bool {
    unimplemented!()
}

fn main() {}
