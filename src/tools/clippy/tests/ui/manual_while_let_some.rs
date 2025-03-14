#![allow(unused)]
#![warn(clippy::manual_while_let_some)]

struct VecInStruct {
    numbers: Vec<i32>,
    unrelated: String,
}

struct Foo {
    a: i32,
    b: i32,
}

fn accept_i32(_: i32) {}
fn accept_optional_i32(_: Option<i32>) {}
fn accept_i32_tuple(_: (i32, i32)) {}

fn main() {
    let mut numbers = vec![1, 2, 3, 4, 5];
    while !numbers.is_empty() {
        let number = numbers.pop().unwrap();
        //~^ manual_while_let_some
    }

    let mut val = VecInStruct {
        numbers: vec![1, 2, 3, 4, 5],
        unrelated: String::new(),
    };
    while !val.numbers.is_empty() {
        let number = val.numbers.pop().unwrap();
        //~^ manual_while_let_some
    }

    while !numbers.is_empty() {
        accept_i32(numbers.pop().unwrap());
        //~^ manual_while_let_some
    }

    while !numbers.is_empty() {
        accept_i32(numbers.pop().expect(""));
        //~^ manual_while_let_some
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

    // This should not warn. It pops elements, but does not unwrap it.
    // Might handle the Option in some other arbitrary way.
    while !numbers.is_empty() {
        accept_optional_i32(numbers.pop());
    }

    let unrelated_vec: Vec<String> = Vec::new();
    // This should not warn. It pops elements from a different vector.
    while !unrelated_vec.is_empty() {
        accept_i32(numbers.pop().unwrap());
    }

    macro_rules! generate_loop {
        () => {
            while !numbers.is_empty() {
                accept_i32(numbers.pop().unwrap());
            }
        };
    }
    // Do not warn if the loop comes from a macro.
    generate_loop!();

    // Try other kinds of patterns
    let mut numbers = vec![(0, 0), (1, 1), (2, 2)];
    while !numbers.is_empty() {
        let (a, b) = numbers.pop().unwrap();
        //~^ manual_while_let_some
    }

    while !numbers.is_empty() {
        accept_i32_tuple(numbers.pop().unwrap());
        //~^ manual_while_let_some
    }

    let mut results = vec![Foo { a: 1, b: 2 }, Foo { a: 3, b: 4 }];
    while !results.is_empty() {
        let Foo { a, b } = results.pop().unwrap();
        //~^ manual_while_let_some
    }
}
