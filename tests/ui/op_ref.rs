#![allow(unused_variables, clippy::blacklisted_name)]

use std::collections::HashSet;

fn main() {
    let tracked_fds: HashSet<i32> = HashSet::new();
    let new_fds = HashSet::new();
    let unwanted = &tracked_fds - &new_fds;

    let foo = &5 - &6;

    let bar = String::new();
    let bar = "foo" == &bar;

    let a = "a".to_string();
    let b = "a";

    if b < &a {
        println!("OK");
    }
}
