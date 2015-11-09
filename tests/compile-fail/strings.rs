#![feature(plugin)]
#![plugin(clippy)]

#[deny(string_add)]
#[allow(string_add_assign)]
fn add_only() { // ignores assignment distinction
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + "."; //~ERROR you added something to a string.
    }

    let y = "".to_owned();
    let z = y + "..."; //~ERROR you added something to a string.

    assert_eq!(&x, &z);
}

#[deny(string_add_assign)]
fn add_assign_only() {
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + "."; //~ERROR you assigned the result of adding something to this string.
    }

    let y = "".to_owned();
    let z = y + "...";

    assert_eq!(&x, &z);
}

#[deny(string_add, string_add_assign)]
fn both() {
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + "."; //~ERROR you assigned the result of adding something to this string.
    }

    let y = "".to_owned();
    let z = y + "..."; //~ERROR you added something to a string.

    assert_eq!(&x, &z);
}

fn main() {
    add_only();
    add_assign_only();
    both();

    // the add is only caught for String
    let mut x = 1;
    x = x + 1;
    assert_eq!(2, x);
}
