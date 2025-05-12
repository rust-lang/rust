//@ check-pass

#![feature(box_patterns)]

#![warn(unused)] // UI tests pass `-A unused` (#43896)

struct SoulHistory {
    corridors_of_light: usize,
    hours_are_suns: bool,
    endless_and_singing: bool
}

struct LovelyAmbition {
    lips: usize,
    fire: usize
}

#[derive(Clone, Copy)]
enum Large {
    Suit { case: () }
}

struct Tuple(Large, ());

fn main() {
    let i_think_continually = 2; //~ WARNING unused variable: `i_think_continually`
    let who_from_the_womb_remembered = SoulHistory {
        corridors_of_light: 5,
        hours_are_suns: true,
        endless_and_singing: true
    };

    let mut mut_unused_var = 1;
    //~^ WARNING unused variable: `mut_unused_var`
    //~| WARNING variable does not need to be mutable

    let (mut var, unused_var) = (1, 2);
    //~^ WARNING unused variable: `var`
    //~| WARNING unused variable: `unused_var`
    //~| WARNING variable does not need to be mutable
    // NOTE: `var` comes after `unused_var` lexicographically yet the warning
    // for `var` will be emitted before the one for `unused_var`. We use an
    // `IndexMap` to ensure this is the case instead of a `BTreeMap`.

    if let SoulHistory { corridors_of_light, //~ WARNING unused variable: `corridors_of_light`
                         mut hours_are_suns, //~ WARNING `hours_are_suns` is assigned to, but
                         endless_and_singing: true } = who_from_the_womb_remembered {
        hours_are_suns = false; //~ WARNING unused_assignments
    }

    let the_spirit = LovelyAmbition { lips: 1, fire: 2 };
    let LovelyAmbition { lips, fire } = the_spirit; //~ WARNING unused variable: `fire`
    println!("{}", lips);

    let bag = Large::Suit {
        case: ()
    };

    // Plain struct
    match bag {
        Large::Suit { case } => {} //~ WARNING unused variable: `case`
    };

    // Referenced struct
    match &bag {
        &Large::Suit { case } => {} //~ WARNING unused variable: `case`
    };

    // Boxed struct
    match Box::new(bag) {
        box Large::Suit { case } => {} //~ WARNING unused variable: `case`
    };

    // Tuple with struct
    match (bag,) {
        (Large::Suit { case },) => {} //~ WARNING unused variable: `case`
    };

    // Slice with struct
    match [bag] {
        [Large::Suit { case }] => {} //~ WARNING unused variable: `case`
    };

    // Tuple struct with struct
    match Tuple(bag, ()) {
        Tuple(Large::Suit { case }, ()) => {} //~ WARNING unused variable: `case`
    };
}
