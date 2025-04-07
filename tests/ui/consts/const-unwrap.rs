//@ check-fail
// Verify that panicking `const_option` methods do the correct thing

const FOO: i32 = Some(42i32).unwrap();

const BAR: i32 = Option::<i32>::None.unwrap();
//~^ ERROR: evaluation of constant value failed
//~| NOTE: called `Option::unwrap()` on a `None` value

const BAZ: i32 = Option::<i32>::None.expect("absolutely not!");
//~^ ERROR: evaluation of constant value failed
//~| NOTE: absolutely not!

fn main() {
    println!("{}", FOO);
    println!("{}", BAR);
}
