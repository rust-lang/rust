#![feature(non_exhaustive_omitted_patterns_lint)]
#![warn(clippy::match_same_arms)]
#![no_main]

use std::sync::atomic::Ordering; // #[non_exhaustive] enum

pub fn f(x: Ordering) {
    match x {
        Ordering::Relaxed => println!("relaxed"),
        Ordering::Release => println!("release"),
        Ordering::Acquire => println!("acquire"),
        Ordering::AcqRel | Ordering::SeqCst => panic!(),
        #[deny(non_exhaustive_omitted_patterns)]
        _ => panic!(),
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
            Ordering::AcqRel | Ordering::SeqCst => panic!(),
            _ => panic!(),
        }
    }
}

// Below should still lint

pub fn g(x: Ordering) {
    match x {
        Ordering::Relaxed => println!("relaxed"),
        Ordering::Release => println!("release"),
        Ordering::Acquire => println!("acquire"),
        Ordering::AcqRel | Ordering::SeqCst => panic!(),
        _ => panic!(),
    }
}

mod g {
    use super::*;

    pub fn g(x: Ordering) {
        match x {
            Ordering::Relaxed => println!("relaxed"),
            Ordering::Release => println!("release"),
            Ordering::Acquire => println!("acquire"),
            Ordering::AcqRel | Ordering::SeqCst => panic!(),
            _ => panic!(),
        }
    }
}
