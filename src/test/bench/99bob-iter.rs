

/* -*- mode::rust;indent-tabs-mode::nil -*-
 * Implementation of 99 Bottles of Beer
 * http://99-bottles-of-beer.net/
 */
use std;
import int;
import str;

fn b1() -> str { ret "# of beer on the wall, # of beer."; }

fn b2() -> str {
    ret "Take one down and pass it around, # of beer on the wall.";
}

fn b7() -> str {
    ret "No more bottles of beer on the wall, no more bottles of beer.";
}

fn b8() -> str {
    ret "Go to the store and buy some more, # of beer on the wall.";
}

fn sub(t: str, n: int) -> str unsafe {
    let mut b: str = "";
    let mut i: uint = 0u;
    let mut ns: str;
    alt n {
      0 { ns = "no more bottles"; }
      1 { ns = "1 bottle"; }
      _ { ns = int::to_str(n, 10u) + " bottles"; }
    }
    while i < str::len(t) {
        if t[i] == '#' as u8 { b += ns; }
        else { str::unsafe::push_byte(b, t[i]); }
        i += 1u;
    }
    ret b;
}


/* Using an interator */
fn ninetynine(it: fn(int)) {
    let mut n: int = 100;
    while n > 1 { n -= 1; it(n); }
}

fn main() {
    ninetynine {|n|
        log(debug, sub(b1(), n));
        log(debug, sub(b2(), n - 1));
        #debug("");
    };
    log(debug, b7());
    log(debug, b8());
}
