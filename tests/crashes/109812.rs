//@ known-bug: #109812

#![warn(rust_2021_incompatible_closure_captures)]

enum Either {
    One(X),
    Two(X),
}

struct X(Y);

struct Y;

fn move_into_fnmut() {
    let x = X(Y);

    consume_fnmut(|| {
        let Either::Two(ref mut _t) = x;

        let X(mut _t) = x;
    });
}
