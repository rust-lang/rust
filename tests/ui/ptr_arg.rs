#![feature(lint_reasons)]
#![allow(
    unused,
    clippy::many_single_char_names,
    clippy::needless_lifetimes,
    clippy::redundant_clone,
    clippy::needless_pass_by_ref_mut
)]
#![warn(clippy::ptr_arg)]

use std::borrow::Cow;
use std::path::{Path, PathBuf};

fn do_vec(x: &Vec<i64>) {
    //Nothing here
}

fn do_vec_mut(x: &mut Vec<i64>) {
    //Nothing here
}

fn do_str(x: &String) {
    //Nothing here either
}

fn do_str_mut(x: &mut String) {
    //Nothing here either
}

fn do_path(x: &PathBuf) {
    //Nothing here either
}

fn do_path_mut(x: &mut PathBuf) {
    //Nothing here either
}

fn main() {}

trait Foo {
    type Item;
    fn do_vec(x: &Vec<i64>);
    fn do_item(x: &Self::Item);
}

struct Bar;

// no error, in trait impl (#425)
impl Foo for Bar {
    type Item = Vec<u8>;
    fn do_vec(x: &Vec<i64>) {}
    fn do_item(x: &Vec<u8>) {}
}

fn cloned(x: &Vec<u8>) -> Vec<u8> {
    let e = x.clone();
    let f = e.clone(); // OK
    let g = x;
    let h = g.clone();
    let i = (e).clone();
    x.clone()
}

fn str_cloned(x: &String) -> String {
    let a = x.clone();
    let b = x.clone();
    let c = b.clone();
    let d = a.clone().clone().clone();
    x.clone()
}

fn path_cloned(x: &PathBuf) -> PathBuf {
    let a = x.clone();
    let b = x.clone();
    let c = b.clone();
    let d = a.clone().clone().clone();
    x.clone()
}

fn false_positive_capacity(x: &Vec<u8>, y: &String) {
    let a = x.capacity();
    let b = y.clone();
    let c = y.as_str();
}

fn false_positive_capacity_too(x: &String) -> String {
    if x.capacity() > 1024 {
        panic!("Too large!");
    }
    x.clone()
}

#[allow(dead_code)]
fn test_cow_with_ref(c: &Cow<[i32]>) {}

fn test_cow(c: Cow<[i32]>) {
    let _c = c;
}

trait Foo2 {
    fn do_string(&self);
}

// no error for &self references where self is of type String (#2293)
impl Foo2 for String {
    fn do_string(&self) {}
}

// Check that the allow attribute on parameters is honored
mod issue_5644 {
    use std::borrow::Cow;
    use std::path::PathBuf;

    fn allowed(
        #[allow(clippy::ptr_arg)] _v: &Vec<u32>,
        #[allow(clippy::ptr_arg)] _s: &String,
        #[allow(clippy::ptr_arg)] _p: &PathBuf,
        #[allow(clippy::ptr_arg)] _c: &Cow<[i32]>,
        #[expect(clippy::ptr_arg)] _expect: &Cow<[i32]>,
    ) {
    }

    fn some_allowed(#[allow(clippy::ptr_arg)] _v: &Vec<u32>, _s: &String) {}

    struct S;
    impl S {
        fn allowed(
            #[allow(clippy::ptr_arg)] _v: &Vec<u32>,
            #[allow(clippy::ptr_arg)] _s: &String,
            #[allow(clippy::ptr_arg)] _p: &PathBuf,
            #[allow(clippy::ptr_arg)] _c: &Cow<[i32]>,
            #[expect(clippy::ptr_arg)] _expect: &Cow<[i32]>,
        ) {
        }
    }

    trait T {
        fn allowed(
            #[allow(clippy::ptr_arg)] _v: &Vec<u32>,
            #[allow(clippy::ptr_arg)] _s: &String,
            #[allow(clippy::ptr_arg)] _p: &PathBuf,
            #[allow(clippy::ptr_arg)] _c: &Cow<[i32]>,
            #[expect(clippy::ptr_arg)] _expect: &Cow<[i32]>,
        ) {
        }
    }
}

mod issue6509 {
    use std::path::PathBuf;

    fn foo_vec(vec: &Vec<u8>) {
        let _ = vec.clone().pop();
        let _ = vec.clone().clone();
    }

    fn foo_path(path: &PathBuf) {
        let _ = path.clone().pop();
        let _ = path.clone().clone();
    }

    fn foo_str(str: &PathBuf) {
        let _ = str.clone().pop();
        let _ = str.clone().clone();
    }
}

fn mut_vec_slice_methods(v: &mut Vec<u32>) {
    v.copy_within(1..5, 10);
}

fn mut_vec_vec_methods(v: &mut Vec<u32>) {
    v.clear();
}

fn vec_contains(v: &Vec<u32>) -> bool {
    [vec![], vec![0]].as_slice().contains(v)
}

fn fn_requires_vec(v: &Vec<u32>) -> bool {
    vec_contains(v)
}

fn impl_fn_requires_vec(v: &Vec<u32>, f: impl Fn(&Vec<u32>)) {
    f(v);
}

fn dyn_fn_requires_vec(v: &Vec<u32>, f: &dyn Fn(&Vec<u32>)) {
    f(v);
}

// No error for types behind an alias (#7699)
type A = Vec<u8>;
fn aliased(a: &A) {}

// Issue #8366
pub trait Trait {
    fn f(v: &mut Vec<i32>);
    fn f2(v: &mut Vec<i32>) {}
}

// Issue #8463
fn two_vecs(a: &mut Vec<u32>, b: &mut Vec<u32>) {
    a.push(0);
    a.push(0);
    a.push(0);
    b.push(1);
}

// Issue #8495
fn cow_conditional_to_mut(a: &mut Cow<str>) {
    if a.is_empty() {
        a.to_mut().push_str("foo");
    }
}

// Issue #9542
fn dyn_trait_ok(a: &mut Vec<u32>, b: &mut String, c: &mut PathBuf) {
    trait T {}
    impl<U> T for Vec<U> {}
    impl T for String {}
    impl T for PathBuf {}
    fn takes_dyn(_: &mut dyn T) {}

    takes_dyn(a);
    takes_dyn(b);
    takes_dyn(c);
}

fn dyn_trait(a: &mut Vec<u32>, b: &mut String, c: &mut PathBuf) {
    trait T {}
    impl<U> T for Vec<U> {}
    impl<U> T for [U] {}
    impl T for String {}
    impl T for str {}
    impl T for PathBuf {}
    impl T for Path {}
    fn takes_dyn(_: &mut dyn T) {}

    takes_dyn(a);
    takes_dyn(b);
    takes_dyn(c);
}

mod issue_9218 {
    use std::borrow::Cow;

    fn cow_non_elided_lifetime<'a>(input: &Cow<'a, str>) -> &'a str {
        todo!()
    }

    // This one has an anonymous lifetime so it's not okay
    fn cow_elided_lifetime<'a>(input: &'a Cow<str>) -> &'a str {
        todo!()
    }

    // These two's return types don't use use 'a so it's not okay
    fn cow_bad_ret_ty_1<'a>(input: &'a Cow<'a, str>) -> &'static str {
        todo!()
    }
    fn cow_bad_ret_ty_2<'a, 'b>(input: &'a Cow<'a, str>) -> &'b str {
        todo!()
    }

    // Inferred to be `&'a str`, afaik.
    fn cow_good_ret_ty<'a>(input: &'a Cow<'a, str>) -> &str {
        todo!()
    }
}
