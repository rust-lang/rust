#![warn(clippy::all)]
#![allow(unused)]

fn the_answer(ref mut x: u8) {
    *x = 42;
}

fn main() {
    let mut x = 0;
    the_answer(x);
    // Closures should not warn
    let y = |ref x| println!("{:?}", x);
    y(1u8);

    let ref x = 1;

    let ref y: (&_, u8) = (&1, 2);

    let ref z = 1 + 2;

    let ref mut z = 1 + 2;

    let (ref x, _) = (1, 2); // ok, not top level
    println!("The answer is {}.", x);

    // Make sure that allowing the lint works
    #[allow(clippy::toplevel_ref_arg)]
    let ref mut x = 1_234_543;
}
