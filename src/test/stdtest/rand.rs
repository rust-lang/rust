

// -*- rust -*-
use std;
import std::rand;
import std::str;

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

#[test]
fn genstr() {
    let r: rand::rng = rand::mk_rng();
    log r.gen_str(10u);
    log r.gen_str(10u);
    log r.gen_str(10u);
    assert(str::char_len(r.gen_str(10u)) == 10u);
    assert(str::char_len(r.gen_str(16u)) == 16u);
}
