//! Test unreachable patterns involving deref patterns.
#![feature(deref_patterns)]
#![expect(incomplete_features)]
#![deny(unreachable_patterns)]

fn main() {
    match Box::new(false) {
        true => {}
        false => {}
        false => {} //~ ERROR unreachable pattern
    }

    match Box::new(Box::new(false)) {
        true => {}
        false => {}
        true => {} //~ ERROR unreachable pattern
    }

    match Box::new((true, Box::new(false))) {
        (true, _) => {}
        (_, true) => {}
        (false, false) => {}
        _ => {} //~ ERROR unreachable pattern
    }

    enum T { A, B, C }
    match Box::new((Box::new(T::A), Box::new(T::A))) {
        (T::A | T::B, T::A | T::C) => {}
        (T::A, T::C) => {} //~ ERROR unreachable pattern
        (T::B, T::A) => {} //~ ERROR unreachable pattern
        _ => {}
    }
}
