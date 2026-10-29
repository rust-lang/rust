// Test that closures and coroutines are "must use" types.
//@ edition:2018

#![feature(coroutines, stmt_expr_attributes)]
#![deny(unused_must_use)]

fn unused() {
    || { //~ ERROR unused closure that must be used
        println!("Hello!");
    };

    async {};    //~ ERROR unused implementer of `Future` that must be used
    || async {}; //~ ERROR unused closure that must be used
    async || {}; //~ ERROR unused closure that must be used


    [Box::new([|| {}; 10]); 1]; //~ ERROR unused array of boxed arrays of closures that must be used

    vec![|| "a"].pop().unwrap(); //~ ERROR unused closure that must be used

    let b = false;
        || true; //~ ERROR unused closure that must be used
    println!("{}", b);
}

fn ignored() {
    let _ = || {};
    let _ = #[coroutine] || yield 42;
}

fn main() {
    unused();
    ignored();
}
