#![warn(clippy::useless_concat)]
#![allow(clippy::print_literal)]

macro_rules! my_concat {
    ($fmt:literal $(, $e:expr)*) => {
        println!(concat!("ERROR: ", $fmt), $($e,)*);
    }
}

fn main() {
    let x = concat!(); //~ useless_concat
    let x = concat!("a"); //~ useless_concat
    let x = concat!(1); //~ useless_concat
    println!("b: {}", concat!("a")); //~ useless_concat
    // Should not lint.
    let x = concat!("a", "b");
    let local_i32 = 1;
    my_concat!("{}", local_i32);
    let x = concat!(file!(), "#L", line!());
}
