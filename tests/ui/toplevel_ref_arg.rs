// run-rustfix

#![warn(clippy::toplevel_ref_arg)]

fn main() {
    // Closures should not warn
    let y = |ref x| println!("{:?}", x);
    y(1u8);

    let ref _x = 1;

    let ref _y: (&_, u8) = (&1, 2);

    let ref _z = 1 + 2;

    let ref mut _z = 1 + 2;

    let (ref x, _) = (1, 2); // ok, not top level
    println!("The answer is {}.", x);

    let ref _x = vec![1, 2, 3];

    // Make sure that allowing the lint works
    #[allow(clippy::toplevel_ref_arg)]
    let ref mut _x = 1_234_543;

    // ok
    for ref _x in 0..10 {}
}
