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

extern mod std;
use std::map;
use std::map::HashMap;
use std::sort;

fn print_complements() {
    let all = ~[Blue, Red, Yellow];
    for vec::each(all) |aa| {
        for vec::each(all) |bb| {
            io::println(show_color(*aa) + ~" + " + show_color(*bb) +
                ~" -> " + show_color(transform(*aa, *bb)));
        }
    }
}

enum color { Red, Yellow, Blue }

type creature_info = { name: uint, color: color };

fn show_color(cc: color) -> ~str {
    match (cc) {
        Red    => {~"red"}
        Yellow => {~"yellow"}
        Blue   => {~"blue"}
    }
}

fn show_color_list(set: ~[color]) -> ~str {
    let mut out = ~"";
    for vec::eachi(set) |_ii, col| {
        out += ~" ";
        out += show_color(*col);
    }
    return out;
}

fn show_digit(nn: uint) -> ~str {
    match (nn) {
        0 => {~"zero"}
        1 => {~"one"}
        2 => {~"two"}
        3 => {~"three"}
        4 => {~"four"}
        5 => {~"five"}
        6 => {~"six"}
        7 => {~"seven"}
        8 => {~"eight"}
        9 => {~"nine"}
        _ => {fail ~"expected digits from 0 to 9..."}
    }
}

fn show_number(nn: uint) -> ~str {
    let mut out = ~"";
    let mut num = nn;
    let mut dig;

    if num == 0 { out = show_digit(0) };

    while num != 0 {
        dig = num % 10;
        num = num / 10;
        out = show_digit(dig) + ~" " + out;
    }

    return out;
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
    from_rendezvous: oldcomm::Port<Option<creature_info>>,
    to_rendezvous: oldcomm::Chan<creature_info>,
    to_rendezvous_log: oldcomm::Chan<~str>
) {
    let mut color = color;
    let mut creatures_met = 0;
    let mut evil_clones_met = 0;

    loop {
        // ask for a pairing
        oldcomm::send(to_rendezvous, {name: name, color: color});
        let resp = oldcomm::recv(from_rendezvous);

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
                let report = fmt!("%u", creatures_met) + ~" " +
                             show_number(evil_clones_met);
                oldcomm::send(to_rendezvous_log, report);
                break;
            }
        }
    }
}

fn rendezvous(nn: uint, set: ~[color]) {

    pub fn spawn_listener<A: Owned>(+f: fn~(oldcomm::Port<A>)) -> oldcomm::Chan<A> {
        let setup_po = oldcomm::Port();
        let setup_ch = oldcomm::Chan(&setup_po);
        do task::spawn |move f| {
            let po = oldcomm::Port();
            let ch = oldcomm::Chan(&po);
            oldcomm::send(setup_ch, ch);
            f(move po);
        }
        oldcomm::recv(setup_po)
    }

    // these ports will allow us to hear from the creatures
    let from_creatures:     oldcomm::Port<creature_info> = oldcomm::Port();
    let from_creatures_log: oldcomm::Port<~str> = oldcomm::Port();

    // these channels will be passed to the creatures so they can talk to us
    let to_rendezvous     = oldcomm::Chan(&from_creatures);
    let to_rendezvous_log = oldcomm::Chan(&from_creatures_log);

    // these channels will allow us to talk to each creature by 'name'/index
    let to_creature: ~[oldcomm::Chan<Option<creature_info>>] =
        vec::mapi(set, |ii, col| {
            // create each creature as a listener with a port, and
            // give us a channel to talk to each
            let ii = ii;
            let col = *col;
            do spawn_listener |from_rendezvous, move ii, move col| {
                creature(ii, col, from_rendezvous, to_rendezvous,
                         to_rendezvous_log);
            }
        });

    let mut creatures_met = 0;

    // set up meetings...
    for nn.times {
        let fst_creature: creature_info = oldcomm::recv(from_creatures);
        let snd_creature: creature_info = oldcomm::recv(from_creatures);

        creatures_met += 2;

        oldcomm::send(to_creature[fst_creature.name], Some(snd_creature));
        oldcomm::send(to_creature[snd_creature.name], Some(fst_creature));
    }

    // tell each creature to stop
    for vec::eachi(to_creature) |_ii, to_one| {
        oldcomm::send(*to_one, None);
    }

    // save each creature's meeting stats
    let mut report = ~[];
    for vec::each(to_creature) |_to_one| {
        report.push(oldcomm::recv(from_creatures_log));
    }

    // print each color in the set
    io::println(show_color_list(set));

    // print each creature's stats
    for vec::each(report) |rep| {
        io::println(*rep);
    }

    // print the total number of creatures met
    io::println(show_number(creatures_met));
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"200000"]
    } else if args.len() <= 1u {
        ~[~"", ~"600"]
    } else {
        args
    };

    let nn = uint::from_str(args[1]).get();

    print_complements();
    io::println(~"");

    rendezvous(nn, ~[Blue, Red, Yellow]);
    io::println(~"");

    rendezvous(nn,
        ~[Blue, Red, Yellow, Red, Yellow, Blue, Red, Yellow, Red, Blue]);
}

