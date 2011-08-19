

// -*- rust -*-
use std;
import std::rand;

#[test]
fn test() {
    let r1: rand::rng = rand::mk_rng();
    log r1.next();
    log r1.next();
    {
        let r2 = rand::mk_rng();
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
