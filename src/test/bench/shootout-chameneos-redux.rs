// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-pretty-expanded

#![feature(phase)]
#[phase(plugin)] extern crate green;

use std::string::String;
use std::fmt;

green_start!(main)

fn print_complements() {
    let all = [Blue, Red, Yellow];
    for aa in all.iter() {
        for bb in all.iter() {
            println!("{} + {} -> {}", *aa, *bb, transform(*aa, *bb));
        }
    }
}

enum Color { Red, Yellow, Blue }
impl fmt::Show for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str = match *self {
            Red => "red",
            Yellow => "yellow",
            Blue => "blue",
        };
        write!(f, "{}", str)
    }
}

struct CreatureInfo {
    name: uint,
    color: Color
}

fn show_color_list(set: Vec<Color>) -> String {
    let mut out = String::new();
    for col in set.iter() {
        out.push_char(' ');
        out.push_str(col.to_str().as_slice());
    }
    out
}

fn show_digit(nn: uint) -> &'static str {
    match nn {
        0 => {" zero"}
        1 => {" one"}
        2 => {" two"}
        3 => {" three"}
        4 => {" four"}
        5 => {" five"}
        6 => {" six"}
        7 => {" seven"}
        8 => {" eight"}
        9 => {" nine"}
        _ => {fail!("expected digits from 0 to 9...")}
    }
}

struct Number(uint);
impl fmt::Show for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut out = vec![];
        let Number(mut num) = *self;
        if num == 0 { out.push(show_digit(0)) };

        while num != 0 {
            let dig = num % 10;
            num = num / 10;
            let s = show_digit(dig);
            out.push(s);
        }

        for s in out.iter().rev() {
            try!(write!(f, "{}", s))
        }
        Ok(())
    }
}

fn transform(aa: Color, bb: Color) -> Color {
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
    mut color: Color,
    from_rendezvous: Receiver<CreatureInfo>,
    to_rendezvous: Sender<CreatureInfo>,
    to_rendezvous_log: Sender<String>
) {
    let mut creatures_met = 0;
    let mut evil_clones_met = 0;
    let mut rendezvous = from_rendezvous.iter();

    loop {
        // ask for a pairing
        to_rendezvous.send(CreatureInfo {name: name, color: color});

        // log and change, or quit
        match rendezvous.next() {
            Some(other_creature) => {
                color = transform(color, other_creature.color);

                // track some statistics
                creatures_met += 1;
                if other_creature.name == name {
                   evil_clones_met += 1;
                }
            }
            None => break
        }
    }
    // log creatures met and evil clones of self
    let report = format!("{}{}", creatures_met, Number(evil_clones_met));
    to_rendezvous_log.send(report);
}

fn rendezvous(nn: uint, set: Vec<Color>) {
    // these ports will allow us to hear from the creatures
    let (to_rendezvous, from_creatures) = channel::<CreatureInfo>();

    // these channels will be passed to the creatures so they can talk to us
    let (to_rendezvous_log, from_creatures_log) = channel::<String>();

    // these channels will allow us to talk to each creature by 'name'/index
    let mut to_creature: Vec<Sender<CreatureInfo>> =
        set.iter().enumerate().map(|(ii, &col)| {
            // create each creature as a listener with a port, and
            // give us a channel to talk to each
            let to_rendezvous = to_rendezvous.clone();
            let to_rendezvous_log = to_rendezvous_log.clone();
            let (to_creature, from_rendezvous) = channel();
            spawn(proc() {
                creature(ii,
                         col,
                         from_rendezvous,
                         to_rendezvous,
                         to_rendezvous_log);
            });
            to_creature
        }).collect();

    let mut creatures_met = 0;

    // set up meetings...
    for _ in range(0, nn) {
        let fst_creature = from_creatures.recv();
        let snd_creature = from_creatures.recv();

        creatures_met += 2;

        to_creature.get_mut(fst_creature.name).send(snd_creature);
        to_creature.get_mut(snd_creature.name).send(fst_creature);
    }

    // tell each creature to stop
    drop(to_creature);

    // print each color in the set
    println!("{}", show_color_list(set));

    // print each creature's stats
    drop(to_rendezvous_log);
    for rep in from_creatures_log.iter() {
        println!("{}", rep);
    }

    // print the total number of creatures met
    println!("{}\n", Number(creatures_met));
}

fn main() {
    let nn = if std::os::getenv("RUST_BENCH").is_some() {
        200000
    } else {
        std::os::args().as_slice()
                       .get(1)
                       .and_then(|arg| from_str(arg.as_slice()))
                       .unwrap_or(600)
    };

    print_complements();
    println!("");

    rendezvous(nn, vec!(Blue, Red, Yellow));

    rendezvous(nn,
        vec!(Blue, Red, Yellow, Red, Yellow, Blue, Red, Yellow, Red, Blue));
}
