//@ run-pass
#![allow(dead_code)]
#![warn(unused_assignments)]

fn diverging_else() {
    let x;
    if true {
        x = 42;
    } else {
        x = 35;
        //~^ WARN value assigned to `x` is never read
        panic!();
    }
    println!("{x}");
}

fn diverging_match_arm() {
    let x;
    match true {
        false => {
            x = 35;
            //~^ WARN value assigned to `x` is never read
            panic!();
        }
        true => x = 42,
    }
    println!("{x}");
}

fn real_overwrite_after_if(cond: bool) {
    let mut x;
    if cond {
        x = 35;
        //~^ WARN value assigned to `x` is never read
    } else {
        x = 42;
        //~^ WARN value assigned to `x` is never read
    }
    x = 99;
    println!("{x}");
}

fn main() {}
