// we have some HELP annotations -- don't complain about them not being present everywhere
//@require-annotations-for-level: ERROR

#![warn(clippy::redundant_closure, clippy::redundant_closure_for_method_calls)]
#![allow(unused)]
#![allow(
    clippy::needless_borrow,
    clippy::needless_option_as_deref,
    clippy::needless_pass_by_value,
    clippy::no_effect,
    clippy::option_map_unit_fn,
    clippy::redundant_closure_call,
    clippy::uninlined_format_args,
    clippy::useless_vec,
    clippy::unnecessary_map_on_constructor,
    clippy::needless_lifetimes
)]

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
    //~^ redundant_closure
    let c = Some(1u8).map(|a| {1+2; foo}(a));
    true.then(|| mac!()); // don't lint function in macro expansion
    Some(1).map(closure_mac!()); // don't lint closure in macro expansion
    let _: Option<Vec<u8>> = true.then(|| vec![]); // special case vec!
    //
    //~^^ redundant_closure
    let d = Some(1u8).map(|a| foo((|b| foo2(b))(a))); //is adjusted?
    //
    //~^^ redundant_closure
    all(&[1, 2, 3], &&2, |x, y| below(x, y)); //is adjusted
    //
    //~^^ redundant_closure
    unsafe {
        Some(1u8).map(|a| unsafe_fn(a)); // unsafe fn
    }

    // See #815
    let e = Some(1u8).map(|a| divergent(a));
    let e = Some(1u8).map(|a| generic(a));
    //~^ redundant_closure
    let e = Some(1u8).map(generic);
    // See #515
    let a: Option<Box<dyn (::std::ops::Deref<Target = [i32]>)>> =
        Some(vec![1i32, 2]).map(|v| -> Box<dyn (::std::ops::Deref<Target = [i32]>)> { Box::new(v) });

    // issue #7224
    let _: Option<Vec<u32>> = Some(0).map(|_| vec![]);

    // issue #10684
    fn test<T>(x: impl Fn(usize, usize) -> T) -> T {
        x(1, 2)
    }
    test(|start, end| start..=end);
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
    //~^ redundant_closure_for_method_calls
    let e = Some(TestStruct { some_ref: &i }).map(|a| a.trait_foo());
    //~^ redundant_closure_for_method_calls
    let e = Some(TestStruct { some_ref: &i }).map(|a| a.trait_foo_ref());
    let e = Some(&mut vec![1, 2, 3]).map(|v| v.clear());
    //~^ redundant_closure_for_method_calls
    unsafe {
        let e = Some(TestStruct { some_ref: &i }).map(|a| a.foo_unsafe());
    }
    let e = Some("str").map(|s| s.to_string());
    //~^ redundant_closure_for_method_calls
    let e = Some('a').map(|s| s.to_uppercase());
    //~^ redundant_closure_for_method_calls
    let e: std::vec::Vec<usize> = vec!['a', 'b', 'c'].iter().map(|c| c.len_utf8()).collect();
    let e: std::vec::Vec<char> = vec!['a', 'b', 'c'].iter().map(|c| c.to_ascii_uppercase()).collect();
    //~^ redundant_closure_for_method_calls
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

    fn issue14096() {
        let x = Some("42");
        let _ = x.map(|x| x.parse::<i16>());
        //~^ redundant_closure_for_method_calls
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
    //~^ redundant_closure
}
fn requires_fn_once<T: FnOnce()>(_: T) {}

fn test_redundant_closure_with_function_pointer() {
    type FnPtrType = fn(u8);
    let foo_ptr: FnPtrType = foo;
    let a = Some(1u8).map(|a| foo_ptr(a));
    //~^ redundant_closure
}

fn test_redundant_closure_with_another_closure() {
    let closure = |a| println!("{}", a);
    let a = Some(1u8).map(|a| closure(a));
    //~^ redundant_closure
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
    //~^ redundant_closure
    y.into_iter().for_each(|x| add_to_res(x));
    //~^ redundant_closure
    z.into_iter().for_each(|x| add_to_res(x));
    //~^ redundant_closure
}

fn mutable_closure_in_loop() {
    let mut value = 0;
    let mut closure = |n| value += n;
    for _ in 0..5 {
        Some(1).map(|n| closure(n));
        //~^ redundant_closure

        let mut value = 0;
        let mut in_loop = |n| value += n;
        Some(1).map(|n| in_loop(n));
        //~^ redundant_closure
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

// #8460 Don't replace closures with params bounded as `ref`
mod bind_by_ref {
    struct A;
    struct B;

    impl From<&A> for B {
        fn from(A: &A) -> Self {
            B
        }
    }

    fn test() {
        // should not lint
        Some(A).map(|a| B::from(&a));
        // should not lint
        Some(A).map(|ref a| B::from(a));
    }
}

// #7812 False positive on coerced closure
fn coerced_closure() {
    fn function_returning_unit<F: FnMut(i32)>(f: F) {}
    function_returning_unit(|x| std::process::exit(x));

    fn arr() -> &'static [u8; 0] {
        &[]
    }
    fn slice_fn(_: impl FnOnce() -> &'static [u8]) {}
    slice_fn(|| arr());
}

// https://github.com/rust-lang/rust-clippy/issues/7861
fn box_dyn() {
    fn f(_: impl Fn(usize) -> Box<dyn std::any::Any>) {}
    f(|x| Box::new(x));
}

// https://github.com/rust-lang/rust-clippy/issues/5939
fn not_general_enough() {
    fn f(_: impl FnMut(&Path) -> std::io::Result<()>) {}
    f(|path| std::fs::remove_file(path));
}

// https://github.com/rust-lang/rust-clippy/issues/9369
pub fn mutable_impl_fn_mut(mut f: impl FnMut(), mut f_used_once: impl FnMut()) -> impl FnMut() {
    fn takes_fn_mut(_: impl FnMut()) {}
    takes_fn_mut(|| f());
    //~^ redundant_closure

    fn takes_fn_once(_: impl FnOnce()) {}
    takes_fn_once(|| f());
    //~^ redundant_closure

    f();

    move || takes_fn_mut(|| f_used_once())
    //~^ redundant_closure
}

impl dyn TestTrait + '_ {
    fn method_on_dyn(&self) -> bool {
        false
    }
}

// https://github.com/rust-lang/rust-clippy/issues/7746
fn angle_brackets_and_args() {
    let array_opt: Option<&[u8; 3]> = Some(&[4, 8, 7]);
    array_opt.map(|a| a.as_slice());
    //~^ redundant_closure_for_method_calls

    let slice_opt: Option<&[u8]> = Some(b"slice");
    slice_opt.map(|s| s.len());
    //~^ redundant_closure_for_method_calls

    let ptr_opt: Option<*const usize> = Some(&487);
    ptr_opt.map(|p| p.is_null());
    //~^ redundant_closure_for_method_calls

    let test_struct = TestStruct { some_ref: &487 };
    let dyn_opt: Option<&dyn TestTrait> = Some(&test_struct);
    dyn_opt.map(|d| d.method_on_dyn());
    //~^ redundant_closure_for_method_calls
}

// https://github.com/rust-lang/rust-clippy/issues/12199
fn track_caller_fp() {
    struct S;
    impl S {
        #[track_caller]
        fn add_location(self) {}
    }

    #[track_caller]
    fn add_location() {}

    fn foo(_: fn()) {}
    fn foo2(_: fn(S)) {}
    foo(|| add_location());
    foo2(|s| s.add_location());
}

fn _late_bound_to_early_bound_regions() {
    struct Foo<'a>(&'a u32);
    impl<'a> Foo<'a> {
        fn f(x: &'a u32) -> Self {
            Foo(x)
        }
    }
    fn f(f: impl for<'a> Fn(&'a u32) -> Foo<'a>) -> Foo<'static> {
        f(&0)
    }

    let _ = f(|x| Foo::f(x));

    struct Bar;
    impl<'a> From<&'a u32> for Bar {
        fn from(x: &'a u32) -> Bar {
            Bar
        }
    }
    fn f2(f: impl for<'a> Fn(&'a u32) -> Bar) -> Bar {
        f(&0)
    }

    let _ = f2(|x| <Bar>::from(x));

    struct Baz<'a>(&'a u32);
    fn f3(f: impl Fn(&u32) -> Baz<'_>) -> Baz<'static> {
        f(&0)
    }

    let _ = f3(|x| Baz(x));
}

fn _mixed_late_bound_and_early_bound_regions() {
    fn f<T>(t: T, f: impl Fn(T, &u32) -> u32) -> u32 {
        f(t, &0)
    }
    fn f2<'a, T: 'a>(_: &'a T, y: &u32) -> u32 {
        *y
    }
    let _ = f(&0, |x, y| f2(x, y));
    //~^ redundant_closure
}

fn _closure_with_types() {
    fn f<T>(x: T) -> T {
        x
    }
    fn f2<T: Default>(f: impl Fn(T) -> T) -> T {
        f(T::default())
    }

    let _ = f2(|x: u32| f(x));
    let _ = f2(|x| -> u32 { f(x) });
}

/// https://github.com/rust-lang/rust-clippy/issues/10854
/// This is to verify that redundant_closure_for_method_calls resolves suggested paths to relative.
mod issue_10854 {
    pub mod test_mod {
        pub struct Test;

        impl Test {
            pub fn method(self) -> i32 {
                0
            }
        }

        pub fn calls_test(test: Option<Test>) -> Option<i32> {
            test.map(|t| t.method())
            //~^ redundant_closure_for_method_calls
        }

        pub fn calls_outer(test: Option<super::Outer>) -> Option<i32> {
            test.map(|t| t.method())
            //~^ redundant_closure_for_method_calls
        }
    }

    pub struct Outer;

    impl Outer {
        pub fn method(self) -> i32 {
            0
        }
    }

    pub fn calls_into_mod(test: Option<test_mod::Test>) -> Option<i32> {
        test.map(|t| t.method())
        //~^ redundant_closure_for_method_calls
    }

    mod a {
        pub mod b {
            pub mod c {
                pub fn extreme_nesting(test: Option<super::super::super::d::Test>) -> Option<i32> {
                    test.map(|t| t.method())
                    //~^ redundant_closure_for_method_calls
                }
            }
        }
    }

    mod d {
        pub struct Test;

        impl Test {
            pub fn method(self) -> i32 {
                0
            }
        }
    }
}

mod issue_12853 {
    fn f_by_value<F: Fn(u32)>(f: F) {
        let x = Box::new(|| None.map(|x| f(x)));
        //~^ redundant_closure
        x();
    }
    fn f_by_ref<F: Fn(u32)>(f: &F) {
        let x = Box::new(|| None.map(|x| f(x)));
        //~^ redundant_closure
        x();
    }
}

mod issue_13073 {
    fn get_default() -> Option<&'static str> {
        Some("foo")
    }

    pub fn foo() {
        // shouldn't lint
        let bind: Option<String> = None;
        let _field = bind.as_deref().or_else(|| get_default()).unwrap();
        let bind: Option<&'static str> = None;
        let _field = bind.as_deref().or_else(|| get_default()).unwrap();
        // should lint
        let _field = bind.or_else(|| get_default()).unwrap();
        //~^ redundant_closure
    }
}

fn issue_14789() {
    _ = Some(1u8).map(
        #[expect(clippy::redundant_closure)]
        |a| foo(a),
    );

    _ = Some("foo").map(
        #[expect(clippy::redundant_closure_for_method_calls)]
        |s| s.to_owned(),
    );

    let _: Vec<u8> = None.map_or_else(
        #[expect(clippy::redundant_closure)]
        || vec![],
        std::convert::identity,
    );
}

fn issue_15072() {
    use std::ops::Deref;

    struct Foo;
    impl Deref for Foo {
        type Target = fn() -> &'static str;

        fn deref(&self) -> &Self::Target {
            fn hello() -> &'static str {
                "Hello, world!"
            }
            &(hello as fn() -> &'static str)
        }
    }

    fn accepts_fn(f: impl Fn() -> &'static str) {
        println!("{}", f());
    }

    fn some_fn() -> &'static str {
        todo!()
    }

    let f = &Foo;
    accepts_fn(|| f());
    //~^ redundant_closure

    let g = &some_fn;
    accepts_fn(|| g());
    //~^ redundant_closure

    struct Bar(Foo);
    impl Deref for Bar {
        type Target = Foo;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    let b = &Bar(Foo);
    accepts_fn(|| b());
    //~^ redundant_closure
}

fn issue8817() {
    fn f(_: u32) -> u32 {
        todo!()
    }
    let g = |_: u32| -> u32 { todo!() };
    struct S(u32);
    enum MyError {
        A(S),
    }

    Some(5)
        .map(|n| f(n))
        //~^ redundant_closure
        //~| HELP: replace the closure with the function itself
        .map(|n| g(n))
        //~^ redundant_closure
        //~| HELP: replace the closure with the function itself
        .map(|n| S(n))
        //~^ redundant_closure
        //~| HELP: replace the closure with the tuple struct itself
        .map(|n| MyError::A(n))
        //~^ redundant_closure
        //~| HELP: replace the closure with the tuple variant itself
        .unwrap(); // just for nicer formatting
}
