// run-rustfix

#![allow(
    unused,
    clippy::no_effect,
    clippy::redundant_closure_call,
    clippy::needless_pass_by_value,
    clippy::option_map_unit_fn,
    clippy::needless_borrow
)]
#![warn(clippy::redundant_closure, clippy::redundant_closure_for_method_calls)]

use std::path::{Path, PathBuf};

macro_rules! mac {
    () => {
        foobar()
    };
}

macro_rules! closure_mac {
    () => {
        |n| foo(n)
    };
}

fn main() {
    let a = Some(1u8).map(|a| foo(a));
    let c = Some(1u8).map(|a| {1+2; foo}(a));
    true.then(|| mac!()); // don't lint function in macro expansion
    Some(1).map(closure_mac!()); // don't lint closure in macro expansion
    let _: Option<Vec<u8>> = true.then(|| vec![]); // special case vec!
    let d = Some(1u8).map(|a| foo((|b| foo2(b))(a))); //is adjusted?
    all(&[1, 2, 3], &&2, |x, y| below(x, y)); //is adjusted
    unsafe {
        Some(1u8).map(|a| unsafe_fn(a)); // unsafe fn
    }

    // See #815
    let e = Some(1u8).map(|a| divergent(a));
    let e = Some(1u8).map(|a| generic(a));
    let e = Some(1u8).map(generic);
    // See #515
    let a: Option<Box<dyn (::std::ops::Deref<Target = [i32]>)>> =
        Some(vec![1i32, 2]).map(|v| -> Box<dyn (::std::ops::Deref<Target = [i32]>)> { Box::new(v) });

    // issue #7224
    let _: Option<Vec<u32>> = Some(0).map(|_| vec![]);
}

trait TestTrait {
    fn trait_foo(self) -> bool;
    fn trait_foo_ref(&self) -> bool;
}

struct TestStruct<'a> {
    some_ref: &'a i32,
}

impl<'a> TestStruct<'a> {
    fn foo(self) -> bool {
        false
    }
    unsafe fn foo_unsafe(self) -> bool {
        true
    }
}

impl<'a> TestTrait for TestStruct<'a> {
    fn trait_foo(self) -> bool {
        false
    }
    fn trait_foo_ref(&self) -> bool {
        false
    }
}

impl<'a> std::ops::Deref for TestStruct<'a> {
    type Target = char;
    fn deref(&self) -> &char {
        &'a'
    }
}

fn test_redundant_closures_containing_method_calls() {
    let i = 10;
    let e = Some(TestStruct { some_ref: &i }).map(|a| a.foo());
    let e = Some(TestStruct { some_ref: &i }).map(|a| a.trait_foo());
    let e = Some(TestStruct { some_ref: &i }).map(|a| a.trait_foo_ref());
    let e = Some(&mut vec![1, 2, 3]).map(|v| v.clear());
    unsafe {
        let e = Some(TestStruct { some_ref: &i }).map(|a| a.foo_unsafe());
    }
    let e = Some("str").map(|s| s.to_string());
    let e = Some('a').map(|s| s.to_uppercase());
    let e: std::vec::Vec<usize> = vec!['a', 'b', 'c'].iter().map(|c| c.len_utf8()).collect();
    let e: std::vec::Vec<char> = vec!['a', 'b', 'c'].iter().map(|c| c.to_ascii_uppercase()).collect();
    let e = Some(PathBuf::new()).as_ref().and_then(|s| s.to_str());
    let c = Some(TestStruct { some_ref: &i })
        .as_ref()
        .map(|c| c.to_ascii_uppercase());

    fn test_different_borrow_levels<T>(t: &[&T])
    where
        T: TestTrait,
    {
        t.iter().filter(|x| x.trait_foo_ref());
        t.iter().map(|x| x.trait_foo_ref());
    }
}

struct Thunk<T>(Box<dyn FnMut() -> T>);

impl<T> Thunk<T> {
    fn new<F: 'static + FnOnce() -> T>(f: F) -> Thunk<T> {
        let mut option = Some(f);
        // This should not trigger redundant_closure (#1439)
        Thunk(Box::new(move || option.take().unwrap()()))
    }

    fn unwrap(self) -> T {
        let Thunk(mut f) = self;
        f()
    }
}

fn foobar() {
    let thunk = Thunk::new(|| println!("Hello, world!"));
    thunk.unwrap()
}

fn foo(_: u8) {}

fn foo2(_: u8) -> u8 {
    1u8
}

fn all<X, F>(x: &[X], y: &X, f: F) -> bool
where
    F: Fn(&X, &X) -> bool,
{
    x.iter().all(|e| f(e, y))
}

fn below(x: &u8, y: &u8) -> bool {
    x < y
}

unsafe fn unsafe_fn(_: u8) {}

fn divergent(_: u8) -> ! {
    unimplemented!()
}

fn generic<T>(_: T) -> u8 {
    0
}

fn passes_fn_mut(mut x: Box<dyn FnMut()>) {
    requires_fn_once(|| x());
}
fn requires_fn_once<T: FnOnce()>(_: T) {}

fn test_redundant_closure_with_function_pointer() {
    type FnPtrType = fn(u8);
    let foo_ptr: FnPtrType = foo;
    let a = Some(1u8).map(|a| foo_ptr(a));
}

fn test_redundant_closure_with_another_closure() {
    let closure = |a| println!("{}", a);
    let a = Some(1u8).map(|a| closure(a));
}

fn make_lazy(f: impl Fn() -> fn(u8) -> u8) -> impl Fn(u8) -> u8 {
    // Currently f is called when result of make_lazy is called.
    // If the closure is removed, f will be called when make_lazy itself is
    // called. This changes semantics, so the closure must stay.
    Box::new(move |x| f()(x))
}

fn call<F: FnOnce(&mut String) -> String>(f: F) -> String {
    f(&mut "Hello".to_owned())
}
fn test_difference_in_mutability() {
    call(|s| s.clone());
}

struct Bar;
impl std::ops::Deref for Bar {
    type Target = str;
    fn deref(&self) -> &str {
        "hi"
    }
}

fn test_deref_with_trait_method() {
    let _ = [Bar].iter().map(|s| s.to_string()).collect::<Vec<_>>();
}

fn mutable_closure_used_again(x: Vec<i32>, y: Vec<i32>, z: Vec<i32>) {
    let mut res = Vec::new();
    let mut add_to_res = |n| res.push(n);
    x.into_iter().for_each(|x| add_to_res(x));
    y.into_iter().for_each(|x| add_to_res(x));
    z.into_iter().for_each(|x| add_to_res(x));
}

fn mutable_closure_in_loop() {
    let mut value = 0;
    let mut closure = |n| value += n;
    for _ in 0..5 {
        Some(1).map(|n| closure(n));
    }
}

fn late_bound_lifetimes() {
    fn take_asref_path<P: AsRef<Path>>(path: P) {}

    fn map_str<F>(thunk: F)
    where
        F: FnOnce(&str),
    {
    }

    fn map_str_to_path<F>(thunk: F)
    where
        F: FnOnce(&str) -> &Path,
    {
    }
    map_str(|s| take_asref_path(s));
    map_str_to_path(|s| s.as_ref());
}

mod type_param_bound {
    trait Trait {
        fn fun();
    }

    fn take<T: 'static>(_: T) {}

    fn test<X: Trait>() {
        // don't lint, but it's questionable that rust requires a cast
        take(|| X::fun());
        take(X::fun as fn());
    }
}

// #8073 Don't replace closure with `Arc<F>` or `Rc<F>`
fn arc_fp() {
    let rc = std::rc::Rc::new(|| 7);
    let arc = std::sync::Arc::new(|n| n + 1);
    let ref_arc = &std::sync::Arc::new(|_| 5);

    true.then(|| rc());
    (0..5).map(|n| arc(n));
    Some(4).map(|n| ref_arc(n));
}
