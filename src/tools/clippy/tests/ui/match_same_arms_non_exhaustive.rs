#![feature(non_exhaustive_omitted_patterns_lint)]
#![warn(clippy::match_same_arms)]
#![no_main]
use std::sync::atomic::Ordering; // #[non_exhaustive] enum

fn repeat() -> ! {
    panic!()
}

#[deny(non_exhaustive_omitted_patterns)]
pub fn f(x: Ordering) {
    match x {
        Ordering::Relaxed => println!("relaxed"),
        Ordering::Release => println!("release"),
        Ordering::Acquire => println!("acquire"),
        Ordering::AcqRel | Ordering::SeqCst => repeat(),
        //~^ match_same_arms
        _ => repeat(),
    }

    match x {
        Ordering::Relaxed => println!("relaxed"),
        Ordering::Release => println!("release"),
        Ordering::Acquire => println!("acquire"),
        Ordering::AcqRel => repeat(),
        //~^ match_same_arms
        Ordering::SeqCst | _ => repeat(),
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
            //~^ match_same_arms
            _ => repeat(),
        }
    }
}

// Below can still suggest removing the other patterns

pub fn g(x: Ordering) {
    match x {
        Ordering::Relaxed => println!("relaxed"),
        Ordering::Release => println!("release"),
        Ordering::Acquire => println!("acquire"),
        Ordering::AcqRel | Ordering::SeqCst => repeat(),
        _ => repeat(),
        //~^ match_same_arms
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
            _ => repeat(),
            //~^ match_same_arms
        }
    }
}
