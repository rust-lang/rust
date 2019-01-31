// aux-build:option_helpers.rs

#![warn(clippy::all, clippy::pedantic, clippy::option_unwrap_used)]
#![allow(
    clippy::blacklisted_name,
    unused,
    clippy::print_stdout,
    clippy::non_ascii_literal,
    clippy::new_without_default,
    clippy::missing_docs_in_private_items,
    clippy::needless_pass_by_value,
    clippy::default_trait_access,
    clippy::use_self,
    clippy::new_ret_no_self,
    clippy::useless_format
)]

#[macro_use]
extern crate option_helpers;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::iter::FromIterator;
use std::ops::Mul;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

use option_helpers::IteratorFalsePositives;

pub struct T;

impl T {
    pub fn add(self, other: T) -> T {
        self
    }

    // no error, not public interface
    pub(crate) fn drop(&mut self) {}

    // no error, private function
    fn neg(self) -> Self {
        self
    }

    // no error, private function
    fn eq(&self, other: T) -> bool {
        true
    }

    // No error; self is a ref.
    fn sub(&self, other: T) -> &T {
        self
    }

    // No error; different number of arguments.
    fn div(self) -> T {
        self
    }

    // No error; wrong return type.
    fn rem(self, other: T) {}

    // Fine
    fn into_u32(self) -> u32 {
        0
    }

    fn into_u16(&self) -> u16 {
        0
    }

    fn to_something(self) -> u32 {
        0
    }

    fn new(self) -> Self {
        unimplemented!();
    }
}

struct Lt<'a> {
    foo: &'a u32,
}

impl<'a> Lt<'a> {
    // The lifetime is different, but that’s irrelevant; see issue #734.
    #[allow(clippy::needless_lifetimes)]
    pub fn new<'b>(s: &'b str) -> Lt<'b> {
        unimplemented!()
    }
}

struct Lt2<'a> {
    foo: &'a u32,
}

impl<'a> Lt2<'a> {
    // The lifetime is different, but that’s irrelevant; see issue #734.
    pub fn new(s: &str) -> Lt2 {
        unimplemented!()
    }
}

struct Lt3<'a> {
    foo: &'a u32,
}

impl<'a> Lt3<'a> {
    // The lifetime is different, but that’s irrelevant; see issue #734.
    pub fn new() -> Lt3<'static> {
        unimplemented!()
    }
}

#[derive(Clone, Copy)]
struct U;

impl U {
    fn new() -> Self {
        U
    }
    // Ok because `U` is `Copy`.
    fn to_something(self) -> u32 {
        0
    }
}

struct V<T> {
    _dummy: T,
}

impl<T> V<T> {
    fn new() -> Option<V<T>> {
        None
    }
}

impl Mul<T> for T {
    type Output = T;
    // No error, obviously.
    fn mul(self, other: T) -> T {
        self
    }
}

/// Checks implementation of the following lints:
/// * `OPTION_MAP_UNWRAP_OR`
/// * `OPTION_MAP_UNWRAP_OR_ELSE`
/// * `OPTION_MAP_OR_NONE`
#[rustfmt::skip]
fn option_methods() {
    let opt = Some(1);

    // Check `OPTION_MAP_UNWRAP_OR`.
    // Single line case.
    let _ = opt.map(|x| x + 1)
                // Should lint even though this call is on a separate line.
               .unwrap_or(0);
    // Multi-line cases.
    let _ = opt.map(|x| {
                        x + 1
                    }
              ).unwrap_or(0);
    let _ = opt.map(|x| x + 1)
               .unwrap_or({
                    0
                });
    // Single line `map(f).unwrap_or(None)` case.
    let _ = opt.map(|x| Some(x + 1)).unwrap_or(None);
    // Multi-line `map(f).unwrap_or(None)` cases.
    let _ = opt.map(|x| {
        Some(x + 1)
    }
    ).unwrap_or(None);
    let _ = opt
        .map(|x| Some(x + 1))
        .unwrap_or(None);
    // macro case
    let _ = opt_map!(opt, |x| x + 1).unwrap_or(0); // should not lint

    // Should not lint if not copyable
    let id: String = "identifier".to_string();
    let _ = Some("prefix").map(|p| format!("{}.{}", p, id)).unwrap_or(id);
    // ...but DO lint if the `unwrap_or` argument is not used in the `map`
    let id: String = "identifier".to_string();
    let _ = Some("prefix").map(|p| format!("{}.", p)).unwrap_or(id);

    // Check OPTION_MAP_UNWRAP_OR_ELSE
    // single line case
    let _ = opt.map(|x| x + 1)
                // Should lint even though this call is on a separate line.
               .unwrap_or_else(|| 0);
    // Multi-line cases.
    let _ = opt.map(|x| {
                        x + 1
                    }
              ).unwrap_or_else(|| 0);
    let _ = opt.map(|x| x + 1)
               .unwrap_or_else(||
                    0
                );
    // Macro case.
    // Should not lint.
    let _ = opt_map!(opt, |x| x + 1).unwrap_or_else(|| 0);

    // Check `OPTION_MAP_OR_NONE`.
    // Single line case.
    let _ = opt.map_or(None, |x| Some(x + 1));
    // Multi-line case.
    let _ = opt.map_or(None, |x| {
                        Some(x + 1)
                       }
                );
}

/// Struct to generate false positives for things with `.iter()`.
#[derive(Copy, Clone)]
struct HasIter;

impl HasIter {
    fn iter(self) -> IteratorFalsePositives {
        IteratorFalsePositives { foo: 0 }
    }

    fn iter_mut(self) -> IteratorFalsePositives {
        IteratorFalsePositives { foo: 0 }
    }
}

/// Checks implementation of `FILTER_NEXT` lint.
#[rustfmt::skip]
fn filter_next() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];

    // Single-line case.
    let _ = v.iter().filter(|&x| *x < 0).next();

    // Multi-line case.
    let _ = v.iter().filter(|&x| {
                                *x < 0
                            }
                   ).next();

    // Check that hat we don't lint if the caller is not an `Iterator`.
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.filter().next();
}

/// Checks implementation of `SEARCH_IS_SOME` lint.
#[rustfmt::skip]
fn search_is_some() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];

    // Check `find().is_some()`, single-line case.
    let _ = v.iter().find(|&x| *x < 0).is_some();

    // Check `find().is_some()`, multi-line case.
    let _ = v.iter().find(|&x| {
                              *x < 0
                          }
                   ).is_some();

    // Check `position().is_some()`, single-line case.
    let _ = v.iter().position(|&x| x < 0).is_some();

    // Check `position().is_some()`, multi-line case.
    let _ = v.iter().position(|&x| {
                                  x < 0
                              }
                   ).is_some();

    // Check `rposition().is_some()`, single-line case.
    let _ = v.iter().rposition(|&x| x < 0).is_some();

    // Check `rposition().is_some()`, multi-line case.
    let _ = v.iter().rposition(|&x| {
                                   x < 0
                               }
                   ).is_some();

    // Check that we don't lint if the caller is not an `Iterator`.
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.find().is_some();
    let _ = foo.position().is_some();
    let _ = foo.rposition().is_some();
}

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

/// Checks implementation of `ITER_NTH` lint.
fn iter_nth() {
    let mut some_vec = vec![0, 1, 2, 3];
    let mut boxed_slice: Box<[u8]> = Box::new([0, 1, 2, 3]);
    let mut some_vec_deque: VecDeque<_> = some_vec.iter().cloned().collect();

    {
        // Make sure we lint `.iter()` for relevant types.
        let bad_vec = some_vec.iter().nth(3);
        let bad_slice = &some_vec[..].iter().nth(3);
        let bad_boxed_slice = boxed_slice.iter().nth(3);
        let bad_vec_deque = some_vec_deque.iter().nth(3);
    }

    {
        // Make sure we lint `.iter_mut()` for relevant types.
        let bad_vec = some_vec.iter_mut().nth(3);
    }
    {
        let bad_slice = &some_vec[..].iter_mut().nth(3);
    }
    {
        let bad_vec_deque = some_vec_deque.iter_mut().nth(3);
    }

    // Make sure we don't lint for non-relevant types.
    let false_positive = HasIter;
    let ok = false_positive.iter().nth(3);
    let ok_mut = false_positive.iter_mut().nth(3);
}

#[allow(clippy::similar_names)]
fn main() {
    let opt = Some(0);
    let _ = opt.unwrap();
}
