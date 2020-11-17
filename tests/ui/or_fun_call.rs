// run-rustfix

#![warn(clippy::or_fun_call)]
#![allow(dead_code)]
#![allow(clippy::unnecessary_wraps)]

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::time::Duration;

/// Checks implementation of the `OR_FUN_CALL` lint.
fn or_fun_call() {
    struct Foo;

    impl Foo {
        fn new() -> Foo {
            Foo
        }
    }

    enum Enum {
        A(i32),
    }

    fn make<T>() -> T {
        unimplemented!();
    }

    let with_enum = Some(Enum::A(1));
    with_enum.unwrap_or(Enum::A(5));

    let with_const_fn = Some(Duration::from_secs(1));
    with_const_fn.unwrap_or(Duration::from_secs(5));

    let with_constructor = Some(vec![1]);
    with_constructor.unwrap_or(make());

    let with_new = Some(vec![1]);
    with_new.unwrap_or(Vec::new());

    let with_const_args = Some(vec![1]);
    with_const_args.unwrap_or(Vec::with_capacity(12));

    let with_err: Result<_, ()> = Ok(vec![1]);
    with_err.unwrap_or(make());

    let with_err_args: Result<_, ()> = Ok(vec![1]);
    with_err_args.unwrap_or(Vec::with_capacity(12));

    let with_default_trait = Some(1);
    with_default_trait.unwrap_or(Default::default());

    let with_default_type = Some(1);
    with_default_type.unwrap_or(u64::default());

    let with_vec = Some(vec![1]);
    with_vec.unwrap_or(vec![]);

    let without_default = Some(Foo);
    without_default.unwrap_or(Foo::new());

    let mut map = HashMap::<u64, String>::new();
    map.entry(42).or_insert(String::new());

    let mut btree = BTreeMap::<u64, String>::new();
    btree.entry(42).or_insert(String::new());

    let stringy = Some(String::from(""));
    let _ = stringy.unwrap_or("".to_owned());

    let opt = Some(1);
    let hello = "Hello";
    let _ = opt.ok_or(format!("{} world.", hello));

    // index
    let map = HashMap::<u64, u64>::new();
    let _ = Some(1).unwrap_or(map[&1]);
    let map = BTreeMap::<u64, u64>::new();
    let _ = Some(1).unwrap_or(map[&1]);
    // don't lint index vec
    let vec = vec![1];
    let _ = Some(1).unwrap_or(vec[1]);
}

struct Foo(u8);
struct Bar(String, Duration);
#[rustfmt::skip]
fn test_or_with_ctors() {
    let opt = Some(1);
    let opt_opt = Some(Some(1));
    // we also test for const promotion, this makes sure we don't hit that
    let two = 2;

    let _ = opt_opt.unwrap_or(Some(2));
    let _ = opt_opt.unwrap_or(Some(two));
    let _ = opt.ok_or(Some(2));
    let _ = opt.ok_or(Some(two));
    let _ = opt.ok_or(Foo(2));
    let _ = opt.ok_or(Foo(two));
    let _ = opt.or(Some(2));
    let _ = opt.or(Some(two));

    let _ = Some("a".to_string()).or(Some("b".to_string()));

    let b = "b".to_string();
    let _ = Some(Bar("a".to_string(), Duration::from_secs(1)))
        .or(Some(Bar(b, Duration::from_secs(2))));

    let vec = vec!["foo"];
    let _ = opt.ok_or(vec.len());

    let array = ["foo"];
    let _ = opt.ok_or(array.len());

    let slice = &["foo"][..];
    let _ = opt.ok_or(slice.len());
}

// Issue 4514 - early return
fn f() -> Option<()> {
    let a = Some(1);
    let b = 1i32;

    let _ = a.unwrap_or(b.checked_mul(3)?.min(240));

    Some(())
}

fn main() {}
