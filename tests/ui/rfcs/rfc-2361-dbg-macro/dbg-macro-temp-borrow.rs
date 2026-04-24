//@ run-pass

// Regression test for <https://github.com/rust-lang/rust/issues/153850>
// `dbg!` must not drop arguments' temporaries early in multi-arg form.

fn id() -> i32 {
    42
}

fn main() {
    assert_eq!(*dbg!(&id()), 42);

    assert_eq!(dbg!(0, &id()).0, 0);
    assert_eq!(*dbg!(&id(), 1).0, 42);
    assert_eq!(*dbg!(0, &id(), 2).1, 42);

    let f = || *dbg!(0, &id()).1;
    assert_eq!(f(), 42);
}