// build-pass (FIXME(62277): could be check-pass?)

#![feature(box_syntax)]
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
    let i_think_continually = 2;
    let who_from_the_womb_remembered = SoulHistory {
        corridors_of_light: 5,
        hours_are_suns: true,
        endless_and_singing: true
    };

    let mut mut_unused_var = 1;

    let (mut var, unused_var) = (1, 2);

    if let SoulHistory { corridors_of_light,
                         mut hours_are_suns,
                         endless_and_singing: true } = who_from_the_womb_remembered {
        hours_are_suns = false;
    }

    let the_spirit = LovelyAmbition { lips: 1, fire: 2 };
    let LovelyAmbition { lips, fire } = the_spirit;
    println!("{}", lips);

    let bag = Large::Suit {
        case: ()
    };

    // Plain struct
    match bag {
        Large::Suit { case } => {}
    };

    // Referenced struct
    match &bag {
        &Large::Suit { case } => {}
    };

    // Boxed struct
    match box bag {
        box Large::Suit { case } => {}
    };

    // Tuple with struct
    match (bag,) {
        (Large::Suit { case },) => {}
    };

    // Slice with struct
    match [bag] {
        [Large::Suit { case }] => {}
    };

    // Tuple struct with struct
    match Tuple(bag, ()) {
        Tuple(Large::Suit { case }, ()) => {}
    };
}
