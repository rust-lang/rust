//! Regression test for <https://github.com/rust-lang/rust/issues/46756>.
//! Test reference suggestion correctly puts cast in parentheses.
//@ run-rustfix

#![allow(unused)]

fn light_flows_our_war_of_mocking_words(and_yet: &usize) -> usize {
    and_yet + 1
}

fn main() {
    let behold: isize = 2;
    let with_tears: usize = 3;
    light_flows_our_war_of_mocking_words(behold as usize);
    //~^ ERROR mismatched types [E0308]
    light_flows_our_war_of_mocking_words(with_tears + 4);
    //~^ ERROR mismatched types [E0308]
}
