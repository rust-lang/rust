// run-rustfix

#![allow(
    unused,
    clippy::no_effect,
    clippy::redundant_closure_call,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::option_map_unit_fn
)]
#![warn(
    clippy::redundant_closure,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_borrow
)]

use std::path::PathBuf;

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
    meta(|a| foo(a));
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
    let e = Some(TestStruct { some_ref: &i }).map(TestStruct::foo);
    let e = Some(TestStruct { some_ref: &i }).map(|a| a.trait_foo());
    let e = Some(TestStruct { some_ref: &i }).map(|a| a.trait_foo_ref());
    let e = Some(TestStruct { some_ref: &i }).map(TestTrait::trait_foo);
    let e = Some(&mut vec![1, 2, 3]).map(|v| v.clear());
    let e = Some(&mut vec![1, 2, 3]).map(std::vec::Vec::clear);
    unsafe {
        let e = Some(TestStruct { some_ref: &i }).map(|a| a.foo_unsafe());
    }
    let e = Some("str").map(|s| s.to_string());
    let e = Some("str").map(str::to_string);
    let e = Some('a').map(|s| s.to_uppercase());
    let e = Some('a').map(char::to_uppercase);
    let e: std::vec::Vec<usize> = vec!['a', 'b', 'c'].iter().map(|c| c.len_utf8()).collect();
    let e: std::vec::Vec<char> = vec!['a', 'b', 'c'].iter().map(|c| c.to_ascii_uppercase()).collect();
    let e: std::vec::Vec<char> = vec!['a', 'b', 'c'].iter().map(char::to_ascii_uppercase).collect();
    let p = Some(PathBuf::new());
    let e = p.as_ref().and_then(|s| s.to_str());
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

    let mut some = Some(|x| x * x);
    let arr = [Ok(1), Err(2)];
    let _: Vec<_> = arr.iter().map(|x| x.map_err(|e| some.take().unwrap()(e))).collect();
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

fn meta<F>(f: F)
where
    F: Fn(u8),
{
    f(1u8)
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
