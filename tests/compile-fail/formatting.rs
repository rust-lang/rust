#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(if_same_then_else)]

fn foo() -> bool { true }

fn main() {
    // weird `else if` formatting:
    if foo() {
    } if foo() {
    //~^ ERROR this looks like an `else if` but the `else` is missing
    //~| NOTE add the missing `else` or
    }

    let _ = { // if as the last expression
        let _ = 0;

        if foo() {
        } if foo() {
        //~^ ERROR this looks like an `else if` but the `else` is missing
        //~| NOTE add the missing `else` or
        }
        else {
        }
    };

    let _ = { // if in the middle of a block
        if foo() {
        } if foo() {
        //~^ ERROR this looks like an `else if` but the `else` is missing
        //~| NOTE add the missing `else` or
        }
        else {
        }

        let _ = 0;
    };

    if foo() {
    } else
    //~^ ERROR this is an `else if` but the formatting might hide it
    //~| NOTE remove the `else` or
    if foo() { // the span of the above error should continue here
    }

    if foo() {
    }
    //~^ ERROR this is an `else if` but the formatting might hide it
    //~| NOTE remove the `else` or
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
    //~^ ERROR this looks like you are trying to use `.. -= ..`, but you really are doing `.. = (- ..)`
    //~| NOTE to remove this lint, use either `-=` or `= -`
    a =* &191;
    //~^ ERROR this looks like you are trying to use `.. *= ..`, but you really are doing `.. = (* ..)`
    //~| NOTE to remove this lint, use either `*=` or `= *`

    let mut b = true;
    b =! false;
    //~^ ERROR this looks like you are trying to use `.. != ..`, but you really are doing `.. = (! ..)`
    //~| NOTE to remove this lint, use either `!=` or `= !`

    // those are ok:
    a = -35;
    a = *&191;
    b = !false;
}
