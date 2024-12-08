#![warn(clippy::all)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::deref_addrof)]
#![allow(clippy::nonminimal_bool)]

fn foo() -> bool {
    true
}

#[rustfmt::skip]
fn main() {
    // weird op_eq formatting:
    let mut a = 42;
    a =- 35;
    //~^ ERROR: this looks like you are trying to use `.. -= ..`, but you really are doing
    //~| NOTE: to remove this lint, use either `-=` or `= -`
    a =* &191;
    //~^ ERROR: this looks like you are trying to use `.. *= ..`, but you really are doing
    //~| NOTE: to remove this lint, use either `*=` or `= *`

    let mut b = true;
    b =! false;
    //~^ ERROR: this looks like you are trying to use `.. != ..`, but you really are doing
    //~| NOTE: to remove this lint, use either `!=` or `= !`

    // those are ok:
    a = -35;
    a = *&191;
    b = !false;

    // possible missing comma in an array
    let _ = &[
        -1, -2, -3 // <= no comma here
        //~^ ERROR: possibly missing a comma here
        //~| NOTE: to remove this lint, add a comma or write the expr in a single line
        -4, -5, -6
    ];
    let _ = &[
        -1, -2, -3 // <= no comma here
        //~^ ERROR: possibly missing a comma here
        //~| NOTE: to remove this lint, add a comma or write the expr in a single line
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
    // lint if it doesn't
    let _ = &[
        -1
        //~^ ERROR: possibly missing a comma here
        //~| NOTE: to remove this lint, add a comma or write the expr in a single line
        -4,
    ];
}
