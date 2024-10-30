// FIXME(guard_patterns): this lint should not be triggered
// once exhaustiveness is implemented correctly
#![allow(irrefutable_let_patterns)]

fn match_guards_still_work() {
    match 0 {
        0 if guard(0) => {},
        _ => {},
    }
}

fn other_guards_dont() {
    match 0 {
        (0 if guard(0)) | 1 => {},
        //~^ ERROR: guard patterns are experimental
        _ => {},
    }

    let ((x if guard(x)) | x) = 0;
    //~^ ERROR: guard patterns are experimental

    if let (x if guard(x)) = 0 {}
    //~^ ERROR: guard patterns are experimental
    while let (x if guard(x)) = 0 {}
    //~^ ERROR: guard patterns are experimental
}

fn even_as_function_parameters(((x if guard(x), _) | (_, x)): (i32, i32)) {}
//~^ ERROR: guard patterns are experimental

fn guard<T>(x: T) -> bool {
    unimplemented!()
}

fn main() {
    match_guards_still_work();
    other_guards_dont();
    even_as_function_parameters((0, 0));
}
