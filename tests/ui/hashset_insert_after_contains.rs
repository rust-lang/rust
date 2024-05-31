#![allow(unused)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::needless_borrow)]
#![warn(clippy::hashset_insert_after_contains)]

use std::collections::HashSet;

fn main() {
    should_warn_cases();

    should_not_warn_cases();
}

fn should_warn_cases() {
    let mut set = HashSet::new();
    let value = 5;

    if !set.contains(&value) {
        set.insert(value);
        println!("Just a comment");
    }

    if set.contains(&value) {
        set.insert(value);
        println!("Just a comment");
    }

    if !set.contains(&value) {
        set.insert(value);
    }

    if !!set.contains(&value) {
        set.insert(value);
        println!("Just a comment");
    }

    if (&set).contains(&value) {
        set.insert(value);
    }

    let borrow_value = &6;
    if !set.contains(borrow_value) {
        set.insert(*borrow_value);
    }

    let borrow_set = &mut set;
    if !borrow_set.contains(&value) {
        borrow_set.insert(value);
    }
}

fn should_not_warn_cases() {
    let mut set = HashSet::new();
    let value = 5;
    let another_value = 6;

    if !set.contains(&value) {
        set.insert(another_value);
    }

    if !set.contains(&value) {
        println!("Just a comment");
    }

    if simply_true() {
        set.insert(value);
    }

    if !set.contains(&value) {
        set.replace(value); //it is not insert
        println!("Just a comment");
    }
}

fn simply_true() -> bool {
    true
}
