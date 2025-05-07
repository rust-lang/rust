#![allow(unused)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::needless_borrow)]
#![warn(clippy::set_contains_or_insert)]

use std::collections::{BTreeSet, HashSet};

fn should_warn_hashset() {
    let mut set = HashSet::new();
    let value = 5;

    if !set.contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
        println!("Just a comment");
    }

    if set.contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
        println!("Just a comment");
    }

    if !set.contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
    }

    if !!set.contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
        println!("Just a comment");
    }

    if (&set).contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
    }

    let borrow_value = &6;
    if !set.contains(borrow_value) {
        //~^ set_contains_or_insert
        set.insert(*borrow_value);
    }

    let borrow_set = &mut set;
    if !borrow_set.contains(&value) {
        //~^ set_contains_or_insert
        borrow_set.insert(value);
    }
}

fn should_not_warn_hashset() {
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

    if set.contains(&value) {
        println!("value is already in set");
    } else {
        set.insert(value);
    }
}

fn should_warn_btreeset() {
    let mut set = BTreeSet::new();
    let value = 5;

    if !set.contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
        println!("Just a comment");
    }

    if set.contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
        println!("Just a comment");
    }

    if !set.contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
    }

    if !!set.contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
        println!("Just a comment");
    }

    if (&set).contains(&value) {
        //~^ set_contains_or_insert
        set.insert(value);
    }

    let borrow_value = &6;
    if !set.contains(borrow_value) {
        //~^ set_contains_or_insert
        set.insert(*borrow_value);
    }

    let borrow_set = &mut set;
    if !borrow_set.contains(&value) {
        //~^ set_contains_or_insert
        borrow_set.insert(value);
    }
}

fn should_not_warn_btreeset() {
    let mut set = BTreeSet::new();
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

    if set.contains(&value) {
        println!("value is already in set");
    } else {
        set.insert(value);
    }
}

fn simply_true() -> bool {
    true
}

// This is placed last in order to be able to add new tests without changing line numbers
fn main() {
    should_warn_hashset();
    should_warn_btreeset();
    should_not_warn_hashset();
    should_not_warn_btreeset();
}
