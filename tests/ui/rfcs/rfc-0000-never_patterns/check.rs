// Check that never patterns can't have bodies or guards.
#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn main() {}

macro_rules! never {
    () => { ! }
}

fn no_arms_or_guards(x: Void) {
    match &None::<Void> {
        Some(!) => {}
        //~^ ERROR a never pattern is always unreachable
        None => {}
    }
    match &None::<Void> { //~ ERROR: `&Some(!)` not covered
        Some(!) if true,
        //~^ ERROR guard on a never pattern
        None => {}
    }
    match &None::<Void> { //~ ERROR: `&Some(!)` not covered
        Some(!) if true => {}
        //~^ ERROR a never pattern is always unreachable
        None => {}
    }
    match &None::<Void> {
        Some(never!()) => {}
        //~^ ERROR a never pattern is always unreachable
        None => {}
    }
}
