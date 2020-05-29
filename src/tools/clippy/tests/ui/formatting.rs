#![warn(clippy::all)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::deref_addrof)]

fn foo() -> bool {
    true
}

#[rustfmt::skip]
fn main() {
    // weird `else` formatting:
    if foo() {
    } {
    }

    if foo() {
    } if foo() {
    }

    let _ = { // if as the last expression
        let _ = 0;

        if foo() {
        } if foo() {
        }
        else {
        }
    };

    let _ = { // if in the middle of a block
        if foo() {
        } if foo() {
        }
        else {
        }

        let _ = 0;
    };

    if foo() {
    } else
    {
    }

    if foo() {
    }
    else
    {
    }

    if foo() {
    } else
    if foo() { // the span of the above error should continue here
    }

    if foo() {
    }
    else
    if foo() { // the span of the above error should continue here
    }

    // those are ok:
    if foo() {
    }
    {
    }

    if foo() {
    } else {
    }

    if foo() {
    }
    else {
    }

    if foo() {
    }
    if foo() {
    }

    if foo() {
    } else if foo() {
    }

    if foo() {
    }
    else if foo() {
    }

    if foo() {
    }
    else if
    foo() {}

    // weird op_eq formatting:
    let mut a = 42;
    a =- 35;
    a =* &191;

    let mut b = true;
    b =! false;

    // those are ok:
    a = -35;
    a = *&191;
    b = !false;

    // possible missing comma in an array
    let _ = &[
        -1, -2, -3 // <= no comma here
        -4, -5, -6
    ];
    let _ = &[
        -1, -2, -3 // <= no comma here
        *4, -5, -6
    ];

    // those are ok:
    let _ = &[
        -1, -2, -3,
        -4, -5, -6
    ];
    let _ = &[
        -1, -2, -3,
        -4, -5, -6,
    ];
    let _ = &[
        1 + 2, 3 +
        4, 5 + 6,
    ];

    // don't lint for bin op without unary equiv
    // issue 3244
    vec![
        1
        / 2,
    ];
    // issue 3396
    vec![
        true
        | false,
    ];

    // don't lint if the indentation suggests not to
    let _ = &[
        1 + 2, 3 
                - 4, 5
    ];
    // lint if it doesnt
    let _ = &[
        -1
        -4,
    ];
}
