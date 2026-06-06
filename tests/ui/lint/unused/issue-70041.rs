//@ edition: 2018
//@ check-pass

macro_rules! regex {
    () => {};
}

#[allow(dead_code)]
use regex;

fn main() {}
