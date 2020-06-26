// run-rustfix

#![warn(clippy::let_unit_value)]
#![allow(clippy::no_effect)]
#![allow(unused_variables)]

macro_rules! let_and_return {
    ($n:expr) => {{
        let ret = $n;
    }};
}

fn main() {
    let _x = println!("x");
    let _y = 1; // this is fine
    let _z = ((), 1); // this as well
    if true {
        let _a = ();
    }

    consume_units_with_for_loop(); // should be fine as well

    multiline_sugg();

    let_and_return!(()) // should be fine
}

// Related to issue #1964
fn consume_units_with_for_loop() {
    // `for_let_unit` lint should not be triggered by consuming them using for loop.
    let v = vec![(), (), ()];
    let mut count = 0;
    for _ in v {
        count += 1;
    }
    assert_eq!(count, 3);

    // Same for consuming from some other Iterator<Item = ()>.
    let (tx, rx) = ::std::sync::mpsc::channel();
    tx.send(()).unwrap();
    drop(tx);

    count = 0;
    for _ in rx.iter() {
        count += 1;
    }
    assert_eq!(count, 1);
}

fn multiline_sugg() {
    let v: Vec<u8> = vec![2];

    let _ = v
        .into_iter()
        .map(|i| i * 2)
        .filter(|i| i % 2 == 0)
        .map(|_| ())
        .next()
        .unwrap();
}

#[derive(Copy, Clone)]
pub struct ContainsUnit(()); // should be fine
