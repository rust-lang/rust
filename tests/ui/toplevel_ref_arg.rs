// run-rustfix
// aux-build:macro_rules.rs

#![warn(clippy::toplevel_ref_arg)]

#[macro_use]
extern crate macro_rules;

macro_rules! gen_binding {
    () => {
        let ref _y = 42;
    };
}

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

    // lint in macro
    #[allow(unused)]
    {
        gen_binding!();
    }

    // do not lint in external macro
    {
        ref_arg_binding!();
    }
}
