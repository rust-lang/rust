#![allow(unused)]
#![warn(clippy::while_pop_unwrap)]

struct VecInStruct {
    numbers: Vec<i32>,
    unrelated: String,
}

fn accept_i32(_: i32) {}

fn main() {
    let mut numbers = vec![1, 2, 3, 4, 5];
    while !numbers.is_empty() {
        let number = numbers.pop().unwrap();
    }

    let mut val = VecInStruct {
        numbers: vec![1, 2, 3, 4, 5],
        unrelated: String::new(),
    };
    while !val.numbers.is_empty() {
        let number = val.numbers.pop().unwrap();
    }

    while !numbers.is_empty() {
        accept_i32(numbers.pop().unwrap());
    }

    while !numbers.is_empty() {
        accept_i32(numbers.pop().expect(""));
    }

    // This should not warn. It "conditionally" pops elements.
    while !numbers.is_empty() {
        if true {
            accept_i32(numbers.pop().unwrap());
        }
    }

    // This should also not warn. It conditionally pops elements.
    while !numbers.is_empty() {
        if false {
            continue;
        }
        accept_i32(numbers.pop().unwrap());
    }
}
