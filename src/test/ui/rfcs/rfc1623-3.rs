#![allow(dead_code)]

fn non_elidable<'a, 'b>(a: &'a u8, b: &'b u8) -> &'a u8 {
    a
}

// the boundaries of elision
static NON_ELIDABLE_FN: &fn(&u8, &u8) -> &u8 =
//~^ ERROR missing lifetime specifier [E0106]
    &(non_elidable as fn(&u8, &u8) -> &u8);
    //~^ ERROR missing lifetime specifier [E0106]
    //~| ERROR non-primitive cast

fn main() {}
