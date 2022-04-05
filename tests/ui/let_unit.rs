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

fn _returns_generic() {
    fn f<T>() -> T {
        unimplemented!()
    }
    fn f2<T, U>(_: T) -> U {
        unimplemented!()
    }
    fn f3<T>(x: T) -> T {
        x
    }
    fn f4<T>(mut x: Vec<T>) -> T {
        x.pop().unwrap()
    }

    let _: () = f(); // Ok
    let x: () = f(); // Lint.

    let _: () = f2(0i32); // Ok
    let x: () = f2(0i32); // Lint.

    let _: () = f3(()); // Lint
    let x: () = f3(()); // Lint

    let _: () = f4(vec![()]); // Lint
    let x: () = f4(vec![()]); // Lint

    // Ok
    let _: () = {
        let x = 5;
        f2(x)
    };

    let _: () = if true { f() } else { f2(0) }; // Ok
    let x: () = if true { f() } else { f2(0) }; // Lint

    // Ok
    let _: () = match Some(0) {
        None => f2(1),
        Some(0) => f(),
        Some(1) => f2(3),
        Some(_) => f2('x'),
    };

    // Lint
    let _: () = match Some(0) {
        None => f2(1),
        Some(0) => f(),
        Some(1) => f2(3),
        Some(_) => (),
    };
}
