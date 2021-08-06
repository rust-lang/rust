// check-pass

#![feature(lint_reasons)]

#![warn(unused)]

#[expect(unused_variables, unused_mut, while_true)]
fn check_multiple_lints_1() {
    // This only trigger `unused_variables`
    let who_am_i = 666;
}

#[expect(unused_variables, unused_mut, while_true)]
fn check_multiple_lints_2() {
    // This only triggers `unused_mut`
    let mut x = 0;
    println!("I use x: {}", x);
}


#[expect(unused_variables, unused_mut, while_true)]
fn check_multiple_lints_3() {
    // This only triggers `while_true` which is also an early lint
    while true {}
}

fn main() {
    check_multiple_lints_1();
    check_multiple_lints_2();
    check_multiple_lints_3();
}
