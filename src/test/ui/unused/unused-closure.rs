// Test that closures and generators are "must use" types.
// edition:2018

#![feature(async_closure)]
#![feature(const_in_array_repeat_expressions)]
#![feature(generators)]
#![deny(unused_must_use)]

fn unused() {
    || { //~ ERROR unused closure that must be used
        println!("Hello!");
    };

    async {};    //~ ERROR unused implementer of `Future` that must be used
    || async {}; //~ ERROR unused closure that must be used
    async || {}; //~ ERROR unused closure that must be used


    [Box::new([|| {}; 10]); 1]; //~ ERROR unused array of boxed arrays of closures that must be used

    [|| { //~ ERROR unused array of generators that must be used
        yield 42u32;
    }; 42];

    vec![|| "a"].pop().unwrap(); //~ ERROR unused closure that must be used

    let b = false;
        || true; //~ ERROR unused closure that must be used
    println!("{}", b);
}

fn ignored() {
    let _ = || {};
    let _ = || yield 42;
}

fn main() {
    unused();
    ignored();
}
