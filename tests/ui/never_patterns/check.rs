#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn main() {}

macro_rules! never {
    () => { ! }
}

fn no_arms_or_guards(x: Void) {
    match None::<Void> {
        Some(!) => {}
        None => {}
    }
    match None::<Void> {
        Some(!) if true,
        //~^ ERROR expected one of
        None => {}
    }
    match None::<Void> {
        Some(!) if true => {}
        None => {}
    }
    match None::<Void> {
        Some(never!()) => {},
        None => {}
    }
}
