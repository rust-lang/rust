//@ edition: 2024
//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc
#![feature(gen_blocks)]

fn main() {
    let mut iter = gen {
        42.yield;
        panic!("foo");
        69.yield; //~ WARN: unreachable statement
    };
    assert_eq!(iter.next(), Some(42));
    let mut tmp = std::panic::AssertUnwindSafe(&mut iter);
    match std::panic::catch_unwind(move || tmp.next()) {
        Ok(_) => unreachable!(),
        Err(err) => assert_eq!(*err.downcast::<&'static str>().unwrap(), "foo"),
    }

    match std::panic::catch_unwind(move || iter.next()) {
        Ok(_) => unreachable!(),
        Err(err) => assert_eq!(
            *err.downcast::<&'static str>().unwrap(),
            "`gen fn` should just keep returning `None` after panicking",
        ),
    }
}
