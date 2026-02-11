//! A regression test for <https://github.com/rust-lang/rust/issues/145739>.
//! `format_args!` should not deduplicate captured identifiers referring to values (e.g. consts and
//! constructors). They must be evaluated once per occurrence, whereas explicitly provided arguments
//! are evaluated only once, regardless of how many times they appear in the format string.
//@ run-pass

use std::sync::atomic::{AtomicUsize, Ordering};

static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        DROP_COUNTER.fetch_add(1, Ordering::Relaxed);
    }
}

const FOO: Foo = Foo;

fn main() {
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 0);

    {
        let _x = format_args!("{Foo:?}");
    }
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 1);
    {
        let _x = format_args!("{FOO:?}");
    }
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 2);

    {
        let _x = format_args!("{Foo:?}{Foo:?}");
    }
    // Increased by 2, as `Foo` is constructed for each captured `Foo`.
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 4);
    {
        let _x = format_args!("{FOO:?}{FOO:?}");
    }
    // Increased by 2, as `Foo` is constructed for each captured `FOO`.
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 6);

    {
        let _x = format_args!("{:?}{0:?}", Foo);
    }
    // Increased by 1, as `Foo` is constructed just once as an explicit argument.
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 7);
    {
        let _x = format_args!("{:?}{0:?}", FOO);
    }
    // Increased by 1, as `Foo` is constructed just once as an explicit argument.
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 8);

    {
        let _x = format_args!("{foo:?}{foo:?}{bar:?}{Foo:?}{Foo:?}", foo = Foo, bar = Foo);
    }
    // Increased by 4, as `Foo` is constructed twice for captured `Foo`, and once for each
    // `foo` and `bar`.
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 12);
    {
        let _x = format_args!("{foo:?}{foo:?}{bar:?}{FOO:?}{FOO:?}", foo = FOO, bar = FOO);
    }
    // Increased by 4, as `Foo` is constructed twice for captured `FOO`, and once for each
    // `foo` and `bar`.
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 16);

    {
        let _x = format_args!("{Foo:?}{Foo:?}", Foo = Foo);
    }
    // Increased by 1, as `Foo` is shadowed by an explicit argument.
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 17);
    {
        let _x = format_args!("{FOO:?}{FOO:?}", FOO = FOO);
    }
    // Increased by 1, as `FOO` is shadowed by an explicit argument.
    assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 18);
}
