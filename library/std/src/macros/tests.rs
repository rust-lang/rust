// ignore-tidy-dbg

use core::fmt::Debug;

/// Test for <https://github.com/rust-lang/rust/issues/153850>:
/// `dbg!` shouldn't drop arguments' temporaries.
#[test]
fn no_dropping_temps() {
    fn temp() {}

    *dbg!(&temp());
    *dbg!(&temp(), 1).0;
    *dbg!(0, &temp()).1;
    *dbg!(0, &temp(), 2).1;
}

/// Test for <https://github.com/rust-lang/rust/issues/154988>:
/// `dbg!` shouldn't create a temporary that lives past its invocation.
#[test]
fn no_leaking_internal_temps_from_dbg() {
    #[derive(Debug)]
    struct Foo;

    #[derive(Debug)]
    struct Bar<'a>(#[allow(unused)] &'a Foo);
    impl Drop for Bar<'_> {
        fn drop(&mut self) {}
    }

    let foo = Foo;
    let bar = Bar(&foo);
    // If `dbg!` creates a `(Bar<'_>,)` temporary that lasts past its expansion, this will fail
    // to compile, because it will be dropped after `foo`, which it borrows from. The tuple
    // mimics the drop order of block tail expressions before Rust 2024: first the result of `dbg!`
    // is dropped, then `foo`, then any temporaries left over from `dbg!` are dropped, if present.
    (drop(dbg!(bar)), drop(foo));
}

/// Test for <https://github.com/rust-lang/rust/issues/155902>:
/// `dbg!` shouldn't create a temporary that borrowck thinks can live past its invocation on a false
/// unwind path.
#[test]
fn no_leaking_internal_temps_from_dbg_even_on_false_unwind() {
    #[derive(Debug)]
    struct Foo;
    impl Drop for Foo {
        fn drop(&mut self) {}
    }

    #[derive(Debug)]
    struct Bar<'a>(#[allow(unused)] &'a Foo);
    impl Drop for Bar<'_> {
        fn drop(&mut self) {}
    }

    {
        let foo = Foo;
        let bar = Bar(&foo);
        // The temporaries of this `super let`'s scrutinee will outlive `bar` and `foo`, emulating
        // the drop order of block tail expressions before Rust 2024. If borrowck thinks that a
        // panic from moving `bar` is possible and that a `Bar<'_>`-containing temporary lives past
        // the end of the block because of that, this will fail to compile. Because `Foo` implements
        // `Drop`, it's an error for `foo` to be dropped before such a temporary when unwinding;
        // otherwise, `foo` would just live to the end of the stack frame when unwinding.
        super let _ = drop(dbg!(bar));
    }
}
