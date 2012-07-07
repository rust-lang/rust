// chameneos

import io::reader_util;

use std;
import std::map;
import std::map::hashmap;
import std::sort;

fn print_complements() {
    let all = ~[Blue, Red, Yellow];
    for vec::each(all) |aa| {
        for vec::each(all) |bb| {
            io::println(show_color(aa) + " + " + show_color(bb) +
                " -> " + show_color(transform(aa,bb)));
        }
    }
}

// can I combine these two lines?
enum color_e { Red, Yellow, Blue }
type color = color_e;

type creature_info = { name: uint, color: color };

fn show_color(cc: color) -> str {
    alt (cc) {
        Red    {"red"}
        Yellow {"yellow"}
        Blue   {"blue"}
    }
}

fn show_color_list(set: ~[color]) -> str {
    let mut out = "";
    for vec::eachi(set) |_ii, col| {
        out += " ";
        out += show_color(col);
    }
    ret out;
}

fn show_digit(nn: uint) -> str {
    alt (nn) {
        0 {"zero"}
        1 {"one"}
        2 {"two"}
        3 {"three"}
        4 {"four"}
        5 {"five"}
        6 {"six"}
        7 {"seven"}
        8 {"eight"}
        9 {"nine"}
        _ {fail "expected digits from 0 to 9..."}
    }
}

fn show_number(nn: uint) -> str {
    let mut out = "";
    let mut num = nn;
    let mut dig;

    if num == 0 { out = show_digit(0) };

    while num != 0 {
        dig = num % 10;
        num = num / 10;
        out = show_digit(dig) + " " + out;
    }

    ret out;
}

fn transform(aa: color, bb: color) -> color {
    alt (aa, bb) {
        (Red,    Red   ) { Red    }
        (Red,    Yellow) { Blue   }
        (Red,    Blue  ) { Yellow }
        (Yellow, Red   ) { Blue   }
        (Yellow, Yellow) { Yellow }
        (Yellow, Blue  ) { Red    }
        (Blue,   Red   ) { Yellow }
        (Blue,   Yellow) { Red    }
        (Blue,   Blue  ) { Blue   }
    }
}

fn creature(
    name: uint,
    color: color,
    from_rendezvous: comm::port<option<creature_info>>,
    to_rendezvous: comm::chan<creature_info>,
    to_rendezvous_log: comm::chan<str>
) {
    let mut color = color;
    let mut creatures_met = 0;
    let mut evil_clones_met = 0;

    loop {
        // ask for a pairing
        comm::send(to_rendezvous, {name: name, color: color});
        let resp = comm::recv(from_rendezvous);

        // log and change, or print and quit
        alt resp {
            option::some(other_creature) {
                color = transform(color, other_creature.color);

                // track some statistics
                creatures_met += 1;
                if other_creature.name == name {
                   evil_clones_met += 1;
                }
            }
            option::none {
                // log creatures met and evil clones of self
                let report = #fmt("%u", creatures_met) + " " +
                             show_number(evil_clones_met);
                comm::send(to_rendezvous_log, report);
                break;
            }
        }
    }
}

fn rendezvous(nn: uint, set: ~[color]) {
    let from_creatures:     comm::port<creature_info> = comm::port();
    let from_creatures_log: comm::port<str> = comm::port();
    let to_rendezvous     = comm::chan(from_creatures);
    let to_rendezvous_log = comm::chan(from_creatures_log);
    let to_creature: ~[comm::chan<option<creature_info>>] =
        vec::mapi(set,
            fn@(ii: uint, col: color) -> comm::chan<option<creature_info>> {
                ret do task::spawn_listener |from_rendezvous| {
                    creature(ii, col, from_rendezvous, to_rendezvous,
                             to_rendezvous_log);
                };
            }
        );

    let mut meetings = 0;
    let mut creatures_met = 0;
    let mut creatures_present = 0;

    // use option type instead of initializing to junk?
    let mut first_creature  = { name: 0, color: Red };
    let mut second_creature = { name: 0, color: Red };

    // set up meetings...
    while meetings < nn {
        let creature_req: creature_info = comm::recv(from_creatures);
        creatures_met += 1;

        alt creatures_present {
            0 {
                first_creature = creature_req;
                creatures_present = 1;
              }
            1 {
                second_creature = creature_req;
                comm::send(to_creature[first_creature.name],
                           some(second_creature));
                comm::send(to_creature[second_creature.name],
                           some(first_creature));
                creatures_present = 0;
                meetings += 1;
              }
            _ { fail "too many creatures are here!" }
        }
    }

    // tell each creature to stop
    for vec::eachi(to_creature) |_ii, to_one| {
        comm::send(to_one, none);
    }

    // save each creature's meeting stats
    let mut report = ~[];
    for vec::each(to_creature) |_to_one| {
        vec::push(report, comm::recv(from_creatures_log));
    }

    // print each color in the set
    io::println(show_color_list(set));

    // print each creature's stats
    for vec::each(report) |rep| {
        io::println(rep);
    }

    // print the total number of creatures met
    io::println(show_number(creatures_met));
}

fn main(args: ~[str]) {
    let args = if os::getenv("RUST_BENCH").is_some() || args.len() <= 1u {
        ~["", "600"]
    } else {
        args
    };

    let nn = uint::from_str(args[1]).get();

    print_complements();
    io::println("");

    rendezvous(nn, ~[Blue, Red, Yellow]);
    io::println("");

    rendezvous(nn,
        ~[Blue, Red, Yellow, Red, Yellow, Blue, Red, Yellow, Red, Blue]);
}

