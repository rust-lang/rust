//@ check-pass

#![allow(unreachable_code)]
#![allow(dead_code)]
#![warn(unused_variables)]

fn after_todo() {
    todo!("not implemented");

    // This should not warn - the code is unreachable
    let a = 1;
    if a < 2 {
        eprintln!("a: {}", a);
    }
}

fn after_panic() {
    panic!("oops");

    // This should not warn - the code is unreachable
    let b = 2;
    println!("{}", b);
}

fn after_unimplemented() {
    unimplemented!();

    // This should not warn - the code is unreachable
    let c = 3;
    println!("{}", c);
}

fn after_unreachable() {
    unsafe { std::hint::unreachable_unchecked() }

    // This should not warn - the code is unreachable
    let d = 4;
    println!("{}", d);
}

fn reachable_unused() {
    // This SHOULD warn - the code is reachable
    let e = 5; //~ WARN unused variable: `e`
}

fn main() {}
