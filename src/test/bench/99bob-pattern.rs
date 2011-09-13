

/* -*- mode::rust;indent-tabs-mode::nil -*-
 * Implementation of 99 Bottles of Beer
 * http://99-bottles-of-beer.net/
 */
use std;
import std::int;
import std::str;

tag bottle { none; dual; single; multiple(int); }

fn show(b: bottle) {
    alt b {
      none. {
        log "No more bottles of beer on the wall, " +
                "no more bottles of beer,";
        log "Go to the store and buy some more, " +
                "99 bottles of beer on the wall.";
      }
      single. {
        log "1 bottle of beer on the wall, 1 bottle of beer,";
        log "Take one down and pass it around, " +
                "no more bottles of beer on the wall.";
      }
      dual. {
        log "2 bottles of beer on the wall, 2 bottles of beer,";
        log "Take one down and pass it around, " +
                "1 bottle of beer on the wall.";
      }
      multiple(n) {
        let nb: str = int::to_str(n, 10u);
        let mb: str = int::to_str(n - 1, 10u);
        log nb + " bottles of beer on the wall, " + nb + " bottles of beer,";
        log "Take one down and pass it around, " + mb +
                " bottles of beer on the wall.";
      }
    }
}

fn next(b: bottle) -> bottle {
    alt b {
      none. { ret none; }
      single. { ret none; }
      dual. { ret single; }
      multiple(3) { ret dual; }
      multiple(n) { ret multiple(n - 1); }
    }
}


// Won't need this when tags can be compared with ==
fn more(b: bottle) -> bool { alt b { none. { ret false; } _ { ret true; } } }

fn main() {
    let b: bottle = multiple(99);
    let running: bool = true;
    while running { show(b); log ""; running = more(b); b = next(b); }
}
