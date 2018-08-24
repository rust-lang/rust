#![feature(slice_patterns)]

fn main() {
    match "foo".to_string() {
        ['f', 'o', ..] => {}
        //~^ ERROR expected an array or slice, found `std::string::String`
        _ => { }
    };

    // Note that this one works with default binding modes.
    match &[0, 1, 2] {
        [..] => {}
    };

    match &[0, 1, 2] {
        &[..] => {} // ok
    };

    match [0, 1, 2] {
        [0] => {}, //~ ERROR pattern requires

        [0, 1, x..] => {
            let a: [_; 1] = x;
        }
        [0, 1, 2, 3, x..] => {} //~ ERROR pattern requires
    };

    match does_not_exist { //~ ERROR cannot find value `does_not_exist` in this scope
        [] => {}
    };
}

fn another_fn_to_avoid_suppression() {
    match Default::default()
    {
        [] => {}  //~ ERROR type annotations needed
    };
}
