#[allow(clippy::string_add, unused)]
#[warn(clippy::string_add_assign)]
fn main() {
    // ignores assignment distinction
    let mut x = String::new();

    for _ in 1..3 {
        x = x + ".";
        //~^ string_add_assign
        //~| assign_op_pattern
    }

    let y = String::new();
    let z = y + "...";

    assert_eq!(&x, &z);

    let mut x = 1;
    x = x + 1;
    //~^ assign_op_pattern
    assert_eq!(2, x);
}
