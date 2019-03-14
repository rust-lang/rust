#[warn(clippy::string_add)]
#[allow(clippy::string_add_assign)]
fn add_only() {
    // ignores assignment distinction
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + ".";
    }

    let y = "".to_owned();
    let z = y + "...";

    assert_eq!(&x, &z);
}

#[warn(clippy::string_add_assign)]
fn add_assign_only() {
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + ".";
    }

    let y = "".to_owned();
    let z = y + "...";

    assert_eq!(&x, &z);
}

#[warn(clippy::string_add, clippy::string_add_assign)]
fn both() {
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + ".";
    }

    let y = "".to_owned();
    let z = y + "...";

    assert_eq!(&x, &z);
}

#[allow(clippy::assign_op_pattern)]
fn main() {
    add_only();
    add_assign_only();
    both();

    // the add is only caught for `String`
    let mut x = 1;
    x = x + 1;
    assert_eq!(2, x);
}
