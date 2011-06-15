

// -*- rust -*-
use std;
import std::rand;

fn main() {
    let rand::rng r1 = rand::mk_rng();
    log r1.next();
    log r1.next();
    {
        auto r2 = rand::mk_rng();
        log r1.next();
        log r2.next();
        log r1.next();
        log r1.next();
        log r2.next();
        log r2.next();
        log r1.next();
        log r1.next();
        log r1.next();
        log r2.next();
        log r2.next();
        log r2.next();
    }
    log r1.next();
    log r1.next();
}