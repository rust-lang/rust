

/* -*- mode::rust;indent-tabs-mode::nil -*-
 * Implementation of 99 Bottles of Beer
 * http://99-bottles-of-beer.net/
 */
use std;
import std::int;
import std::str;

fn b1() -> istr { ret ~"# of beer on the wall, # of beer."; }

fn b2() -> istr {
    ret ~"Take one down and pass it around, # of beer on the wall.";
}

fn b7() -> istr {
    ret ~"No more bottles of beer on the wall, no more bottles of beer.";
}

fn b8() -> istr {
    ret ~"Go to the store and buy some more, # of beer on the wall.";
}

fn sub(t: &istr, n: int) -> istr {
    let b: istr = ~"";
    let i: uint = 0u;
    let ns: istr;
    alt n {
      0 { ns = ~"no more bottles"; }
      1 { ns = ~"1 bottle"; }
      _ { ns = int::to_str(n, 10u) + ~" bottles"; }
    }
    while i < str::byte_len(t) {
        if t[i] == '#' as u8 { b += ns; } else { str::push_byte(b, t[i]); }
        i += 1u;
    }
    ret b;
}


/* Straightforward counter */
fn main() {
    let n: int = 99;
    while n > 0 { log sub(b1(), n); log sub(b2(), n - 1); log ""; n -= 1; }
    log b7();
    log sub(b8(), 99);
}
