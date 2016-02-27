#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(if_same_then_else)]

fn main() {
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
