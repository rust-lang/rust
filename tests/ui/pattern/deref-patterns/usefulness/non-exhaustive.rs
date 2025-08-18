//! Test non-exhaustive matches involving deref patterns.
#![feature(deref_patterns)]
#![expect(incomplete_features)]
#![deny(unreachable_patterns)]

fn main() {
    match Box::new(false) {
        //~^ ERROR non-exhaustive patterns: `deref!(true)` not covered
        false => {}
    }

    match Box::new(Box::new(false)) {
        //~^ ERROR non-exhaustive patterns: `deref!(deref!(false))` not covered
        true => {}
    }

    match Box::new((true, Box::new(false))) {
        //~^ ERROR non-exhaustive patterns: `deref!((true, deref!(true)))` and `deref!((false, deref!(false)))` not covered
        (true, false) => {}
        (false, true) => {}
    }

    enum T { A, B, C }
    match Box::new((Box::new(T::A), Box::new(T::A))) {
        //~^ ERROR non-exhaustive patterns: `deref!((deref!(T::C), _))` not covered
        (T::A | T::B, T::C) => {}
    }
}
