#![feature(non_exhaustive_omitted_patterns_lint)]
#![warn(clippy::match_same_arms)]
#![no_main]
//@no-rustfix
use std::sync::atomic::Ordering; // #[non_exhaustive] enum

fn repeat() -> ! {
    panic!()
}

pub fn f(x: Ordering) {
    #[deny(non_exhaustive_omitted_patterns)]
    match x {
        Ordering::Relaxed => println!("relaxed"),
        Ordering::Release => println!("release"),
        Ordering::Acquire => println!("acquire"),
        Ordering::AcqRel | Ordering::SeqCst => repeat(),
        _ => repeat(),
    }
}

mod f {
    #![deny(non_exhaustive_omitted_patterns)]

    use super::*;

    pub fn f(x: Ordering) {
        match x {
            Ordering::Relaxed => println!("relaxed"),
            Ordering::Release => println!("release"),
            Ordering::Acquire => println!("acquire"),
            Ordering::AcqRel | Ordering::SeqCst => repeat(),
            _ => repeat(),
        }
    }
}

// Below should still lint

pub fn g(x: Ordering) {
    match x {
        Ordering::Relaxed => println!("relaxed"),
        Ordering::Release => println!("release"),
        Ordering::Acquire => println!("acquire"),
        Ordering::AcqRel | Ordering::SeqCst => repeat(),
        //~^ ERROR: this match arm has an identical body to the `_` wildcard arm
        _ => repeat(),
    }
}

mod g {
    use super::*;

    pub fn g(x: Ordering) {
        match x {
            Ordering::Relaxed => println!("relaxed"),
            Ordering::Release => println!("release"),
            Ordering::Acquire => println!("acquire"),
            Ordering::AcqRel | Ordering::SeqCst => repeat(),
            //~^ ERROR: this match arm has an identical body to the `_` wildcard arm
            _ => repeat(),
        }
    }
}
