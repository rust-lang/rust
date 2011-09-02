/* -*- mode::rust;indent-tabs-mode::nil -*-
 * Implementation of 99 Bottles of Beer
 * http://99-bottles-of-beer.net/
 */
use std;
import std::int;
import std::str;

fn main() {
    fn multiple(n: int) {
        let nb: istr = int::to_str(n, 10u);
        let mb: istr = int::to_str(n - 1, 10u);
        log nb + ~" bottles of beer on the wall, " + nb +
            ~" bottles of beer,";
        log ~"Take one down and pass it around, " + mb +
            ~" bottles of beer on the wall.";
        log "";
        if n > 3 { be multiple(n - 1); } else { be dual(); }
    }
    fn dual() {
        log "2 bottles of beer on the wall, 2 bottles of beer,";
        log "Take one down and pass it around, 1 bottle of beer on the wall.";
        log "";
        be single();
    }
    fn single() {
        log "1 bottle of beer on the wall, 1 bottle of beer,";
        log "Take one down and pass it around, " +
                "no more bottles of beer on the wall.";
        log "";
        be none();
    }
    fn none() {
        log "No more bottles of beer on the wall, no more bottles of beer,";
        log "Go to the store and buy some more, " +
                "99 bottles of beer on the wall.";
        log "";
    }
    multiple(99);
}
