

/* -*- mode::rust;indent-tabs-mode::nil -*-
 * Implementation of 99 Bottles of Beer
 * http://99-bottles-of-beer.net/
 */
use std;
import int;
import str;

tag bottle { none; dual; single; multiple(int); }

fn show(b: bottle) {
    alt b {
      none {
        #debug("No more bottles of beer on the wall, \
                no more bottles of beer,");
        #debug("Go to the store and buy some more, \
                99 bottles of beer on the wall.");
      }
      single {
        #debug("1 bottle of beer on the wall, 1 bottle of beer,");
        #debug("Take one down and pass it around, \
                no more bottles of beer on the wall.");
      }
      dual. {
        #debug("2 bottles of beer on the wall, 2 bottles of beer,");
        #debug("Take one down and pass it around, \
                1 bottle of beer on the wall.");
      }
      multiple(n) {
        #debug("%d bottles of beer on the wall, %d bottles of beer,", n, n);
        #debug("Take one down and pass it around, \
                %d bottles of beer on the wall.", n-1);
      }
    }
}

fn next(b: bottle) -> bottle {
    alt b {
      none { ret none; }
      single { ret none; }
      dual. { ret single; }
      multiple(3) { ret dual; }
      multiple(n) { ret multiple(n - 1); }
    }
}


// Won't need this when tags can be compared with ==
fn more(b: bottle) -> bool { alt b { none { ret false; } _ { ret true; } } }

fn main() {
    let b: bottle = multiple(99);
    let running: bool = true;
    while running { show(b); #debug(""); running = more(b); b = next(b); }
}
