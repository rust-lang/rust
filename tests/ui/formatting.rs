#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(if_same_then_else)]
#![allow(deref_addrof)]

fn foo() -> bool { true }

fn main() {
    // weird `else if` formatting:
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
}
