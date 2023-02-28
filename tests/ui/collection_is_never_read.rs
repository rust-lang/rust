#![allow(unused)]
#![warn(clippy::collection_is_never_read)]

use std::collections::{HashMap, HashSet};

fn main() {}

fn not_a_collection() {
    // TODO: Expand `collection_is_never_read` beyond collections?
    let mut x = 10; // Ok
    x += 1;
}

fn no_access_at_all() {
    // Other lints should catch this.
    let x = vec![1, 2, 3]; // Ok
}

fn write_without_read() {
    // The main use case for `collection_is_never_read`.
    let mut x = HashMap::new(); // WARNING
    x.insert(1, 2);
}

fn read_without_write() {
    let mut x = vec![1, 2, 3]; // Ok
    let _ = x.len();
}

fn write_and_read() {
    let mut x = vec![1, 2, 3]; // Ok
    x.push(4);
    let _ = x.len();
}

fn write_after_read() {
    // TODO: Warn here, but this requires more extensive data flow analysis.
    let mut x = vec![1, 2, 3]; // Ok
    let _ = x.len();
    x.push(4); // Pointless
}

fn write_before_reassign() {
    // TODO: Warn here, but this requires more extensive data flow analysis.
    let mut x = HashMap::new(); // Ok
    x.insert(1, 2); // Pointless
    x = HashMap::new();
    let _ = x.len();
}

fn read_in_closure() {
    let mut x = HashMap::new(); // Ok
    x.insert(1, 2);
    let _ = || {
        let _ = x.len();
    };
}

fn write_in_closure() {
    let mut x = vec![1, 2, 3]; // WARNING
    let _ = || {
        x.push(4);
    };
}

fn read_in_format() {
    let mut x = HashMap::new(); // Ok
    x.insert(1, 2);
    format!("{x:?}");
}

fn shadowing_1() {
    let x = HashMap::<usize, usize>::new(); // Ok
    let _ = x.len();
    let mut x = HashMap::new(); // WARNING
    x.insert(1, 2);
}

fn shadowing_2() {
    let mut x = HashMap::new(); // WARNING
    x.insert(1, 2);
    let x = HashMap::<usize, usize>::new(); // Ok
    let _ = x.len();
}

#[allow(clippy::let_unit_value)]
fn fake_read() {
    let mut x = vec![1, 2, 3]; // Ok
    x.reverse();
    // `collection_is_never_read` gets fooled, but other lints should catch this.
    let _: () = x.clear();
}

fn assignment() {
    let mut x = vec![1, 2, 3]; // WARNING
    let y = vec![4, 5, 6]; // Ok
    x = y;
}

#[allow(clippy::self_assignment)]
fn self_assignment() {
    let mut x = vec![1, 2, 3]; // WARNING
    x = x;
}

fn method_argument_but_not_target() {
    struct MyStruct;
    impl MyStruct {
        fn my_method(&self, _argument: &[usize]) {}
    }
    let my_struct = MyStruct;

    let mut x = vec![1, 2, 3]; // Ok
    x.reverse();
    my_struct.my_method(&x);
}

fn insert_is_not_a_read() {
    let mut x = HashSet::new(); // WARNING
    x.insert(5);
}

fn insert_is_a_read() {
    let mut x = HashSet::new(); // Ok
    if x.insert(5) {
        println!("5 was inserted");
    }
}

fn not_read_if_return_value_not_used() {
    // `is_empty` does not modify the set, so it's a query. But since the return value is not used, the
    // lint does not consider it a read here.
    let x = vec![1, 2, 3]; // WARNING
    x.is_empty();
}
