#![feature(plugin)]
#![plugin(clippy)]
#![warn(unit_expr)]
#[allow(unused_variables)]

fn main() {
    // lint should note removing the semicolon from "baz"
    let x = {
        "foo";
        "baz";
    };


    // lint should ignore false positive.
    let y = if true {
        "foo"
    } else {
        return;
    };

    // lint should note removing semicolon from "bar"
    let z = if true {
        "foo";
    } else {
        "bar";
    };


    let a1 = Some(5);

    // lint should ignore false positive
    let a2 = match a1 {
        Some(x) => x,
        _ => {
            return;
        },
    };

    // lint should note removing the semicolon after `x;`
    let a3 = match a1 {
        Some(x) => {
            x;
        },
        _ => {
            0;
        },
    };
}
