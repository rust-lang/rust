/* -*- mode::rust;indent-tabs-mode::nil -*-
 * Implementation of 99 Bottles of Beer
 * http://99-bottles-of-beer.net/
 */
use std;
import int;
import str;

fn main() {
    fn multiple(n: int) {
        #debug("%d bottles of beer on the wall, %d bottles of beer,", n, n);
        #debug("Take one down and pass it around, %d \
                bottles of beer on the wall.", n-1);
        #debug("");
        if n > 3 { be multiple(n - 1); } else { be dual(); }
    }
    fn dual() {
        #debug("2 bottles of beer on the wall, 2 bottles of beer,");
        #debug("Take one down and pass it around, \
                1 bottle of beer on the wall.");
        #debug("");
        be single();
    }
    fn single() {
        #debug("1 bottle of beer on the wall, 1 bottle of beer,");
        log "Take one down and pass it around, " +
                "no more bottles of beer on the wall.";
        #debug("");
        be none();
    }
    fn none() {
        #debug("No more bottles of beer on the wall, \
                no more bottles of beer,");
        log "Go to the store and buy some more, " +
                "99 bottles of beer on the wall.";
        #debug("");
    }
    multiple(99);
}
