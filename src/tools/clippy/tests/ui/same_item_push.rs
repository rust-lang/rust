#![warn(clippy::same_item_push)]

const VALUE: u8 = 7;

fn mutate_increment(x: &mut u8) -> u8 {
    *x += 1;
    *x
}

fn increment(x: u8) -> u8 {
    x + 1
}

fn fun() -> usize {
    42
}

fn main() {
    // ** linted cases **
    let mut vec: Vec<u8> = Vec::new();
    let item = 2;
    for _ in 5..=20 {
        vec.push(item);
    }

    let mut vec: Vec<u8> = Vec::new();
    for _ in 0..15 {
        let item = 2;
        vec.push(item);
    }

    let mut vec: Vec<u8> = Vec::new();
    for _ in 0..15 {
        vec.push(13);
    }

    let mut vec = Vec::new();
    for _ in 0..20 {
        vec.push(VALUE);
    }

    let mut vec = Vec::new();
    let item = VALUE;
    for _ in 0..20 {
        vec.push(item);
    }

    // ** non-linted cases **
    let mut spaces = Vec::with_capacity(10);
    for _ in 0..10 {
        spaces.push(vec![b' ']);
    }

    // Suggestion should not be given as pushed variable can mutate
    let mut vec: Vec<u8> = Vec::new();
    let mut item: u8 = 2;
    for _ in 0..30 {
        vec.push(mutate_increment(&mut item));
    }

    let mut vec: Vec<u8> = Vec::new();
    let mut item: u8 = 2;
    let mut item2 = &mut mutate_increment(&mut item);
    for _ in 0..30 {
        vec.push(mutate_increment(item2));
    }

    let mut vec: Vec<usize> = Vec::new();
    for (a, b) in [0, 1, 4, 9, 16].iter().enumerate() {
        vec.push(a);
    }

    let mut vec: Vec<u8> = Vec::new();
    for i in 0..30 {
        vec.push(increment(i));
    }

    let mut vec: Vec<u8> = Vec::new();
    for i in 0..30 {
        vec.push(i + i * i);
    }

    // Suggestion should not be given as there are multiple pushes that are not the same
    let mut vec: Vec<u8> = Vec::new();
    let item: u8 = 2;
    for _ in 0..30 {
        vec.push(item);
        vec.push(item * 2);
    }

    // Suggestion should not be given as Vec is not involved
    for _ in 0..5 {
        println!("Same Item Push");
    }

    struct A {
        kind: u32,
    }
    let mut vec_a: Vec<A> = Vec::new();
    for i in 0..30 {
        vec_a.push(A { kind: i });
    }
    let mut vec: Vec<u8> = Vec::new();
    for a in vec_a {
        vec.push(2u8.pow(a.kind));
    }

    // Fix #5902
    let mut vec: Vec<u8> = Vec::new();
    let mut item = 0;
    for _ in 0..10 {
        vec.push(item);
        item += 10;
    }

    // Fix #5979
    let mut vec: Vec<std::fs::File> = Vec::new();
    for _ in 0..10 {
        vec.push(std::fs::File::open("foobar").unwrap());
    }
    // Fix #5979
    #[derive(Clone)]
    struct S;

    trait T {}
    impl T for S {}

    let mut vec: Vec<Box<dyn T>> = Vec::new();
    for _ in 0..10 {
        vec.push(Box::new(S {}));
    }

    // Fix #5985
    let mut vec = Vec::new();
    let item = 42;
    let item = fun();
    for _ in 0..20 {
        vec.push(item);
    }

    // Fix #5985
    let mut vec = Vec::new();
    let key = 1;
    for _ in 0..20 {
        let item = match key {
            1 => 10,
            _ => 0,
        };
        vec.push(item);
    }

    // Fix #6987
    let mut vec = Vec::new();
    #[allow(clippy::needless_borrow)]
    for _ in 0..10 {
        vec.push(1);
        vec.extend(&[2]);
    }
}
