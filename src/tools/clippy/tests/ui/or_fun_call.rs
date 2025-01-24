#![warn(clippy::or_fun_call)]
#![allow(dead_code)]
#![allow(
    clippy::borrow_as_ptr,
    clippy::uninlined_format_args,
    clippy::unnecessary_wraps,
    clippy::unnecessary_literal_unwrap,
    clippy::useless_vec
)]

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

/// Checks implementation of the `OR_FUN_CALL` lint.
fn or_fun_call() {
    struct Foo;

    impl Foo {
        fn new() -> Foo {
            Foo
        }
    }

    struct FakeDefault;
    impl FakeDefault {
        fn default() -> Self {
            FakeDefault
        }
    }

    impl Default for FakeDefault {
        fn default() -> Self {
            FakeDefault
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

    let self_default = None::<FakeDefault>;
    self_default.unwrap_or(<FakeDefault>::default());

    let real_default = None::<FakeDefault>;
    real_default.unwrap_or(<FakeDefault as Default>::default());

    let with_vec = Some(vec![1]);
    with_vec.unwrap_or(vec![]);

    let without_default = Some(Foo);
    without_default.unwrap_or(Foo::new());

    let mut map = HashMap::<u64, String>::new();
    map.entry(42).or_insert(String::new());

    let mut map_vec = HashMap::<u64, Vec<i32>>::new();
    map_vec.entry(42).or_insert(vec![]);

    let mut btree = BTreeMap::<u64, String>::new();
    btree.entry(42).or_insert(String::new());

    let mut btree_vec = BTreeMap::<u64, Vec<i32>>::new();
    btree_vec.entry(42).or_insert(vec![]);

    let stringy = Some(String::new());
    let _ = stringy.unwrap_or(String::new());

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

    let string = "foo";
    let _ = opt.ok_or(string.len());
}

// Issue 4514 - early return
fn f() -> Option<()> {
    let a = Some(1);
    let b = 1i32;

    let _ = a.unwrap_or(b.checked_mul(3)?.min(240));

    Some(())
}

mod issue6675 {
    unsafe fn ptr_to_ref<'a, T>(p: *const T) -> &'a T {
        #[allow(unused)]
        let x = vec![0; 1000]; // future-proofing, make this function expensive.
        &*p
    }

    unsafe fn foo() {
        let s = "test".to_owned();
        let s = &s as *const _;
        None.unwrap_or(ptr_to_ref(s));
    }

    fn bar() {
        let s = "test".to_owned();
        let s = &s as *const _;
        None.unwrap_or(unsafe { ptr_to_ref(s) });
        #[rustfmt::skip]
        None.unwrap_or( unsafe { ptr_to_ref(s) }    );
    }
}

mod issue8239 {
    fn more_than_max_suggestion_highest_lines_0() {
        let frames = Vec::new();
        frames
            .iter()
            .map(|f: &String| f.to_lowercase())
            .reduce(|mut acc, f| {
                acc.push_str(&f);
                acc
            })
            .unwrap_or(String::new());
    }

    fn more_to_max_suggestion_highest_lines_1() {
        let frames = Vec::new();
        let iter = frames.iter();
        iter.map(|f: &String| f.to_lowercase())
            .reduce(|mut acc, f| {
                let _ = "";
                let _ = "";
                acc.push_str(&f);
                acc
            })
            .unwrap_or(String::new());
    }

    fn equal_to_max_suggestion_highest_lines() {
        let frames = Vec::new();
        let iter = frames.iter();
        iter.map(|f: &String| f.to_lowercase())
            .reduce(|mut acc, f| {
                let _ = "";
                acc.push_str(&f);
                acc
            })
            .unwrap_or(String::new());
    }

    fn less_than_max_suggestion_highest_lines() {
        let frames = Vec::new();
        let iter = frames.iter();
        let map = iter.map(|f: &String| f.to_lowercase());
        map.reduce(|mut acc, f| {
            acc.push_str(&f);
            acc
        })
        .unwrap_or(String::new());
    }
}

mod issue9608 {
    fn sig_drop() {
        enum X {
            X(std::fs::File),
            Y(u32),
        }

        let _ = None.unwrap_or(X::Y(0));
    }
}

mod issue8993 {
    fn g() -> i32 {
        3
    }

    fn f(n: i32) -> i32 {
        n
    }

    fn test_map_or() {
        let _ = Some(4).map_or(g(), |v| v);
        let _ = Some(4).map_or(g(), f);
        let _ = Some(4).map_or(0, f);
    }
}

mod lazy {
    use super::*;

    fn foo() {
        struct Foo;

        impl Foo {
            fn new() -> Foo {
                Foo
            }
        }

        struct FakeDefault;
        impl FakeDefault {
            fn default() -> Self {
                FakeDefault
            }
        }

        impl Default for FakeDefault {
            fn default() -> Self {
                FakeDefault
            }
        }

        let with_new = Some(vec![1]);
        with_new.unwrap_or_else(Vec::new);

        let with_default_trait = Some(1);
        with_default_trait.unwrap_or_else(Default::default);

        let with_default_type = Some(1);
        with_default_type.unwrap_or_else(u64::default);

        let real_default = None::<FakeDefault>;
        real_default.unwrap_or_else(<FakeDefault as Default>::default);

        let mut map = HashMap::<u64, String>::new();
        map.entry(42).or_insert_with(String::new);

        let mut btree = BTreeMap::<u64, String>::new();
        btree.entry(42).or_insert_with(String::new);

        let stringy = Some(String::new());
        let _ = stringy.unwrap_or_else(String::new);

        // negative tests
        let self_default = None::<FakeDefault>;
        self_default.unwrap_or_else(<FakeDefault>::default);

        let without_default = Some(Foo);
        without_default.unwrap_or_else(Foo::new);
    }
}

fn host_effect() {
    // #12877 - make sure we don't ICE in type_certainty
    use std::ops::Add;

    Add::<i32>::add(1, 1).add(i32::MIN);
}

mod issue_10228 {
    struct Entry;

    impl Entry {
        fn or_insert(self, _default: i32) {}
        fn or_default(self) {
            // Don't lint, suggested code is an infinite recursion
            self.or_insert(Default::default())
        }
    }
}

// issue #12973
fn fn_call_in_nested_expr() {
    struct Foo {
        val: String,
    }

    fn f() -> i32 {
        1
    }
    let opt: Option<i32> = Some(1);

    //~v ERROR: function call inside of `unwrap_or`
    let _ = opt.unwrap_or({ f() }); // suggest `.unwrap_or_else(f)`
    //
    //~v ERROR: function call inside of `unwrap_or`
    let _ = opt.unwrap_or(f() + 1); // suggest `.unwrap_or_else(|| f() + 1)`
    //
    //~v ERROR: function call inside of `unwrap_or`
    let _ = opt.unwrap_or({
        let x = f();
        x + 1
    });
    //~v ERROR: function call inside of `map_or`
    let _ = opt.map_or(f() + 1, |v| v); // suggest `.map_or_else(|| f() + 1, |v| v)`
    //
    //~v ERROR: use of `unwrap_or` to construct default value
    let _ = opt.unwrap_or({ i32::default() });

    let opt_foo = Some(Foo {
        val: String::from("123"),
    });
    //~v ERROR: function call inside of `unwrap_or`
    let _ = opt_foo.unwrap_or(Foo { val: String::default() });
}

fn main() {}
