//@no-rustfix
#![warn(clippy::swap_with_temporary)]

use std::mem::swap;

fn func() -> String {
    String::from("func")
}

fn func_returning_refmut(s: &mut String) -> &mut String {
    s
}

fn main() {
    let mut x = String::from("x");
    let mut y = String::from("y");
    let mut zz = String::from("zz");
    let z = &mut zz;

    swap(&mut func(), &mut func());
    //~^ ERROR: swapping temporary values has no effect

    if matches!(swap(&mut func(), &mut func()), ()) {
        //~^ ERROR: swapping temporary values has no effect
        println!("Yeah");
    }

    if matches!(swap(z, &mut func()), ()) {
        //~^ ERROR: swapping with a temporary value is inefficient
        println!("Yeah");
    }

    macro_rules! mac {
        (refmut $x:expr) => {
            &mut $x
        };
        (refmut) => {
            mac!(refmut String::new())
        };
        (funcall $f:ident) => {
            $f()
        };
    }

    swap(mac!(refmut func()), z);
    //~^ ERROR: swapping with a temporary value is inefficient
    swap(&mut mac!(funcall func), &mut mac!(funcall func));
    //~^ ERROR: swapping temporary values has no effect
    swap(mac!(refmut), mac!(refmut));
    //~^ ERROR: swapping temporary values has no effect
    swap(mac!(refmut y), mac!(refmut));
    //~^ ERROR: swapping with a temporary value is inefficient
}

fn bug(v1: &mut [i32], v2: &mut [i32]) {
    // Incorrect: swapping temporary references (`&mut &mut` passed to swap)
    std::mem::swap(&mut v1.last_mut().unwrap(), &mut v2.last_mut().unwrap());
    //~^ ERROR: swapping temporary values has no effect

    // Correct
    std::mem::swap(v1.last_mut().unwrap(), v2.last_mut().unwrap());
}
