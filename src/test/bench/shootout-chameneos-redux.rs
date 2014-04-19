// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// chameneos

use std::option;
use std::os;
use std::strbuf::StrBuf;
use std::task;

fn print_complements() {
    let all = [Blue, Red, Yellow];
    for aa in all.iter() {
        for bb in all.iter() {
            println!("{} + {} -> {}", show_color(*aa), show_color(*bb),
                show_color(transform(*aa, *bb)));
        }
    }
}

enum color { Red, Yellow, Blue }

struct CreatureInfo {
    name: uint,
    color: color
}

fn show_color(cc: color) -> &'static str {
    match cc {
        Red    => "red",
        Yellow => "yellow",
        Blue   => "blue"
    }
}

fn show_color_list(set: Vec<color>) -> StrBuf {
    let mut out = StrBuf::new();
    for col in set.iter() {
        out.push_char(' ');
        out.push_str(show_color(*col));
    }
    out
}

fn show_digit(nn: uint) -> &'static str {
    match nn {
        0 => {"zero"}
        1 => {"one"}
        2 => {"two"}
        3 => {"three"}
        4 => {"four"}
        5 => {"five"}
        6 => {"six"}
        7 => {"seven"}
        8 => {"eight"}
        9 => {"nine"}
        _ => {fail!("expected digits from 0 to 9...")}
    }
}

fn show_number(nn: uint) -> StrBuf {
    let mut out = vec![];
    let mut num = nn;
    let mut dig;
    let mut len = 0;
    if num == 0 { out.push(show_digit(0)) };

    while num != 0 {
        dig = num % 10;
        num = num / 10;
        out.push(" ");
        let s = show_digit(dig);
        out.push(s);
        len += 1 + s.len();
    }
    len += 1;
    out.push(" ");

    let mut ret = StrBuf::with_capacity(len);
    for s in out.iter().rev() {
        ret.push_str(*s);
    }
    ret
}

fn transform(aa: color, bb: color) -> color {
    match (aa, bb) {
        (Red,    Red   ) => { Red    }
        (Red,    Yellow) => { Blue   }
        (Red,    Blue  ) => { Yellow }
        (Yellow, Red   ) => { Blue   }
        (Yellow, Yellow) => { Yellow }
        (Yellow, Blue  ) => { Red    }
        (Blue,   Red   ) => { Yellow }
        (Blue,   Yellow) => { Red    }
        (Blue,   Blue  ) => { Blue   }
    }
}

fn creature(
    name: uint,
    color: color,
    from_rendezvous: Receiver<Option<CreatureInfo>>,
    to_rendezvous: Sender<CreatureInfo>,
    to_rendezvous_log: Sender<~str>
) {
    let mut color = color;
    let mut creatures_met = 0;
    let mut evil_clones_met = 0;

    loop {
        // ask for a pairing
        to_rendezvous.send(CreatureInfo {name: name, color: color});
        let resp = from_rendezvous.recv();

        // log and change, or print and quit
        match resp {
            option::Some(other_creature) => {
                color = transform(color, other_creature.color);

                // track some statistics
                creatures_met += 1;
                if other_creature.name == name {
                   evil_clones_met += 1;
                }
            }
            option::None => {
                // log creatures met and evil clones of self
                let report = format!("{} {}",
                                     creatures_met, show_number(evil_clones_met).as_slice());
                to_rendezvous_log.send(report);
                break;
            }
        }
    }
}

fn rendezvous(nn: uint, set: Vec<color>) {

    // these ports will allow us to hear from the creatures
    let (to_rendezvous, from_creatures) = channel::<CreatureInfo>();
    let (to_rendezvous_log, from_creatures_log) = channel::<~str>();

    // these channels will be passed to the creatures so they can talk to us

    // these channels will allow us to talk to each creature by 'name'/index
    let mut to_creature: Vec<Sender<Option<CreatureInfo>>> =
        set.iter().enumerate().map(|(ii, col)| {
            // create each creature as a listener with a port, and
            // give us a channel to talk to each
            let ii = ii;
            let col = *col;
            let to_rendezvous = to_rendezvous.clone();
            let to_rendezvous_log = to_rendezvous_log.clone();
            let (to_creature, from_rendezvous) = channel();
            task::spawn(proc() {
                creature(ii,
                         col,
                         from_rendezvous,
                         to_rendezvous.clone(),
                         to_rendezvous_log.clone());
            });
            to_creature
        }).collect();

    let mut creatures_met = 0;

    // set up meetings...
    for _ in range(0, nn) {
        let mut fst_creature: CreatureInfo = from_creatures.recv();
        let mut snd_creature: CreatureInfo = from_creatures.recv();

        creatures_met += 2;

        to_creature.get_mut(fst_creature.name).send(Some(snd_creature));
        to_creature.get_mut(snd_creature.name).send(Some(fst_creature));
    }

    // tell each creature to stop
    for to_one in to_creature.iter() {
        to_one.send(None);
    }

    // save each creature's meeting stats
    let mut report = Vec::new();
    for _to_one in to_creature.iter() {
        report.push(from_creatures_log.recv());
    }

    // print each color in the set
    println!("{}", show_color_list(set));

    // print each creature's stats
    for rep in report.iter() {
        println!("{}", *rep);
    }

    // print the total number of creatures met
    println!("{}", show_number(creatures_met));
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        vec!("".to_owned(), "200000".to_owned())
    } else if args.len() <= 1u {
        vec!("".to_owned(), "600".to_owned())
    } else {
        args.move_iter().collect()
    };

    let nn = from_str::<uint>(*args.get(1)).unwrap();

    print_complements();
    println!("");

    rendezvous(nn, vec!(Blue, Red, Yellow));
    println!("");

    rendezvous(nn,
        vec!(Blue, Red, Yellow, Red, Yellow, Blue, Red, Yellow, Red, Blue));
}
