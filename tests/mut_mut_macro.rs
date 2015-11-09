#![feature(plugin)]
#![plugin(clippy, regex_macros)]

#[macro_use]
extern crate lazy_static;
extern crate regex;

use std::collections::HashMap;

#[test]
#[deny(mut_mut)]
fn test_regex() {
    let pattern = regex!(r"^(?P<level>[#]+)\s(?P<title>.+)$");
    assert!(pattern.is_match("# headline"));
}

#[test]
#[deny(mut_mut)]
#[allow(unused_variables, unused_mut)]
fn test_lazy_static() {
    lazy_static! {
        static ref MUT_MAP : HashMap<usize, &'static str> = {
            let mut m = HashMap::new();
            let mut zero = &mut &mut "zero";
            m.insert(0, "zero");
            m
        };
        static ref MUT_COUNT : usize = MUT_MAP.len();
    }
    assert!(*MUT_COUNT == 1);
}
