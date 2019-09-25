// run-rustfix

#![warn(clippy::toplevel_ref_arg)]
#![allow(unused)]

fn main() {
    // Closures should not warn
    let y = |ref x| println!("{:?}", x);
    y(1u8);

    let ref x = 1;

    let ref y: (&_, u8) = (&1, 2);

    let ref z = 1 + 2;

    let ref mut z = 1 + 2;

    let (ref x, _) = (1, 2); // ok, not top level
    println!("The answer is {}.", x);

    let ref x = vec![1, 2, 3];

    // Make sure that allowing the lint works
    #[allow(clippy::toplevel_ref_arg)]
    let ref mut x = 1_234_543;
}
