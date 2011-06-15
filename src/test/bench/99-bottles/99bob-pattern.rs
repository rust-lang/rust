

/* -*- mode::rust;indent-tabs-mode::nil -*- 
 * Implementation of 99 Bottles of Beer
 * http://99-bottles-of-beer.net/
 */
use std;
import std::int;
import std::str;

tag bottle { none; dual; single; multiple(int); }

fn show(bottle b) {
    alt (b) {
        case (none) {
            log "No more bottles of beer on the wall, " +
              "no more bottles of beer,";
            log "Go to the store and buy some more, " +
                    "99 bottles of beer on the wall.";
        }
        case (single) {
            log "1 bottle of beer on the wall, 1 bottle of beer,";
            log "Take one down and pass it around, " +
                    "no more bottles of beer on the wall.";
        }
        case (dual) {
            log "2 bottles of beer on the wall, 2 bottles of beer,";
            log "Take one down and pass it around, " +
              "1 bottle of beer on the wall.";
        }
        case (multiple(?n)) {
            let str nb = int::to_str(n, 10u);
            let str mb = int::to_str(n - 1, 10u);
            log nb + " bottles of beer on the wall, " + nb +
                    " bottles of beer,";
            log "Take one down and pass it around, " + mb +
                    " bottles of beer on the wall.";
        }
    }
}

fn next(bottle b) -> bottle {
    alt (b) {
        case (none) { ret none; }
        case (single) { ret none; }
        case (dual) { ret single; }
        case (multiple(3)) { ret dual; }
        case (multiple(?n)) { ret multiple(n - 1); }
    }
}


// Won't need this when tags can be compared with ==
fn more(bottle b) -> bool {
    alt (b) { case (none) { ret false; } case (_) { ret true; } }
}

fn main() {
    let bottle b = multiple(99);
    let bool running = true;
    while (running) { show(b); log ""; running = more(b); b = next(b); }
}