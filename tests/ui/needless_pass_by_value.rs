#![warn(clippy::needless_pass_by_value)]
#![allow(
    dead_code,
    clippy::single_match,
    clippy::redundant_pattern_matching,
    clippy::option_option,
    clippy::redundant_clone
)]

use std::borrow::Borrow;
use std::collections::HashSet;
use std::convert::AsRef;
use std::mem::MaybeUninit;

// `v` should be warned
// `w`, `x` and `y` are allowed (moved or mutated)
fn foo<T: Default>(v: Vec<T>, w: Vec<T>, mut x: Vec<T>, y: Vec<T>) -> Vec<T> {
    assert_eq!(v.len(), 42);

    consume(w);

    x.push(T::default());

    y
}

fn consume<T>(_: T) {}

struct Wrapper(String);

fn bar(x: String, y: Wrapper) {
    assert_eq!(x.len(), 42);
    assert_eq!(y.0.len(), 42);
}

// V implements `Borrow<V>`, but should be warned correctly
fn test_borrow_trait<T: Borrow<str>, U: AsRef<str>, V>(t: T, u: U, v: V) {
    println!("{}", t.borrow());
    println!("{}", u.as_ref());
    consume(&v);
}

// ok
fn test_fn<F: Fn(i32) -> i32>(f: F) {
    f(1);
}

// x should be warned, but y is ok
fn test_match(x: Option<Option<String>>, y: Option<Option<String>>) {
    match x {
        Some(Some(_)) => 1, // not moved
        _ => 0,
    };

    match y {
        Some(Some(s)) => consume(s), // moved
        _ => (),
    };
}

// x and y should be warned, but z is ok
fn test_destructure(x: Wrapper, y: Wrapper, z: Wrapper) {
    let Wrapper(s) = z; // moved
    let Wrapper(ref t) = y; // not moved
    let Wrapper(_) = y; // still not moved

    assert_eq!(x.0.len(), s.len());
    println!("{}", t);
}

trait Foo {}

// `S: Serialize` is allowed to be passed by value, since a caller can pass `&S` instead
trait Serialize {}
impl<'a, T> Serialize for &'a T where T: Serialize {}
impl Serialize for i32 {}

fn test_blanket_ref<T: Foo, S: Serialize>(_foo: T, _serializable: S) {}

fn issue_2114(s: String, t: String, u: Vec<i32>, v: Vec<i32>) {
    s.capacity();
    let _ = t.clone();
    u.capacity();
    let _ = v.clone();
}

struct S<T, U>(T, U);

impl<T: Serialize, U> S<T, U> {
    fn foo(
        self,
        // taking `self` by value is always allowed
        s: String,
        t: String,
    ) -> usize {
        s.len() + t.capacity()
    }

    fn bar(_t: T, // Ok, since `&T: Serialize` too
    ) {
    }

    fn baz(&self, _u: U, _s: Self) {}
}

trait FalsePositive {
    fn visit_str(s: &str);
    fn visit_string(s: String) {
        Self::visit_str(&s);
    }
}

// shouldn't warn on extern funcs
extern "C" fn ext(x: MaybeUninit<usize>) -> usize {
    unsafe { x.assume_init() }
}

// exempt RangeArgument
fn range<T: ::std::ops::RangeBounds<usize>>(range: T) {
    let _ = range.start_bound();
}

struct CopyWrapper(u32);

fn bar_copy(x: u32, y: CopyWrapper) {
    assert_eq!(x, 42);
    assert_eq!(y.0, 42);
}

// x and y should be warned, but z is ok
fn test_destructure_copy(x: CopyWrapper, y: CopyWrapper, z: CopyWrapper) {
    let CopyWrapper(s) = z; // moved
    let CopyWrapper(ref t) = y; // not moved
    let CopyWrapper(_) = y; // still not moved

    assert_eq!(x.0, s);
    println!("{}", t);
}

// The following 3 lines should not cause an ICE. See #2831
trait Bar<'a, A> {}
impl<'b, T> Bar<'b, T> for T {}
fn some_fun<'b, S: Bar<'b, ()>>(_item: S) {}

// Also this should not cause an ICE. See #2831
trait Club<'a, A> {}
impl<T> Club<'static, T> for T {}
fn more_fun(_item: impl Club<'static, i32>) {}

fn is_sync<T>(_: T)
where
    T: Sync,
{
}

fn main() {
    // This should not cause an ICE either
    // https://github.com/rust-lang/rust-clippy/issues/3144
    is_sync(HashSet::<usize>::new());
}
