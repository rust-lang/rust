#![warn(clippy::or_fun_call)]

use std::collections::BTreeMap;
use std::collections::HashMap;

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

    let with_const_fn = Some(::std::time::Duration::from_secs(1));
    with_const_fn.unwrap_or(::std::time::Duration::from_secs(5));

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

    // FIXME #944: ~|SUGGESTION with_vec.unwrap_or_else(|| vec![]);

    let without_default = Some(Foo);
    without_default.unwrap_or(Foo::new());

    let mut map = HashMap::<u64, String>::new();
    map.entry(42).or_insert(String::new());

    let mut btree = BTreeMap::<u64, String>::new();
    btree.entry(42).or_insert(String::new());

    let stringy = Some(String::from(""));
    let _ = stringy.unwrap_or("".to_owned());
}

fn main() {}
