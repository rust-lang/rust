#![allow(
    unused,
    clippy::many_single_char_names,
    clippy::needless_lifetimes,
    clippy::redundant_clone,
    clippy::needless_pass_by_ref_mut
)]
#![warn(clippy::ptr_arg)]
//@no-rustfix
use std::borrow::Cow;
use std::path::{Path, PathBuf};

fn do_vec(x: &Vec<i64>) {
    //~^ ptr_arg

    //Nothing here
}

fn do_vec_mut(x: &mut Vec<i64>) {
    //~^ ptr_arg

    //Nothing here
}

fn do_vec_mut2(x: &mut Vec<i64>) {
    //~^ ptr_arg

    x.len();
    x.is_empty();
}

fn do_str(x: &String) {
    //~^ ptr_arg

    //Nothing here either
}

fn do_str_mut(x: &mut String) {
    //~^ ptr_arg

    //Nothing here either
}

fn do_path(x: &PathBuf) {
    //~^ ptr_arg

    //Nothing here either
}

fn do_path_mut(x: &mut PathBuf) {
    //~^ ptr_arg

    //Nothing here either
}

fn main() {}

trait Foo {
    type Item;
    fn do_vec(x: &Vec<i64>);
    //~^ ptr_arg

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
    //~^ ptr_arg

    let e = x.clone();
    let f = e.clone(); // OK
    let g = x;
    let h = g.clone();
    let i = (e).clone();
    x.clone()
}

fn str_cloned(x: &String) -> String {
    //~^ ptr_arg

    let a = x.clone();
    let b = x.clone();
    let c = b.clone();
    let d = a.clone().clone().clone();
    x.clone()
}

fn path_cloned(x: &PathBuf) -> PathBuf {
    //~^ ptr_arg

    let a = x.clone();
    let b = x.clone();
    let c = b.clone();
    let d = a.clone().clone().clone();
    x.clone()
}

fn false_positive_capacity(x: &Vec<u8>, y: &String) {
    //~^ ptr_arg

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
//~^ ptr_arg

fn test_cow(c: Cow<[i32]>) {
    let d = c;
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
        #[allow(clippy::ptr_arg)] v: &Vec<u32>,
        #[allow(clippy::ptr_arg)] s: &String,
        #[allow(clippy::ptr_arg)] p: &PathBuf,
        #[allow(clippy::ptr_arg)] c: &Cow<[i32]>,
        #[expect(clippy::ptr_arg)] expect: &Cow<[i32]>,
    ) {
    }

    fn some_allowed(#[allow(clippy::ptr_arg)] v: &Vec<u32>, s: &String) {}
    //~^ ptr_arg

    struct S;
    impl S {
        fn allowed(
            #[allow(clippy::ptr_arg)] v: &Vec<u32>,
            #[allow(clippy::ptr_arg)] s: &String,
            #[allow(clippy::ptr_arg)] p: &PathBuf,
            #[allow(clippy::ptr_arg)] c: &Cow<[i32]>,
            #[expect(clippy::ptr_arg)] expect: &Cow<[i32]>,
        ) {
        }
    }

    trait T {
        fn allowed(
            #[allow(clippy::ptr_arg)] v: &Vec<u32>,
            #[allow(clippy::ptr_arg)] s: &String,
            #[allow(clippy::ptr_arg)] p: &PathBuf,
            #[allow(clippy::ptr_arg)] c: &Cow<[i32]>,
            #[expect(clippy::ptr_arg)] expect: &Cow<[i32]>,
        ) {
        }
    }
}

mod issue6509 {
    use std::path::PathBuf;

    fn foo_vec(vec: &Vec<u8>) {
        //~^ ptr_arg

        let a = vec.clone().pop();
        let b = vec.clone().clone();
    }

    fn foo_path(path: &PathBuf) {
        //~^ ptr_arg

        let c = path.clone().pop();
        let d = path.clone().clone();
    }

    fn foo_str(str: &String) {
        //~^ ptr_arg

        let e = str.clone().pop();
        let f = str.clone().clone();
    }
}

fn mut_vec_slice_methods(v: &mut Vec<u32>) {
    //~^ ptr_arg

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
    //~^ ptr_arg
    //~| ptr_arg
    //~| ptr_arg

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
        //~^ ptr_arg

        todo!()
    }

    // These two's return types don't use 'a so it's not okay
    fn cow_bad_ret_ty_1<'a>(input: &'a Cow<'a, str>) -> &'static str {
        //~^ ptr_arg

        todo!()
    }
    fn cow_bad_ret_ty_2<'a, 'b>(input: &'a Cow<'a, str>) -> &'b str {
        //~^ ptr_arg

        todo!()
    }

    // Inferred to be `&'a str`, afaik.
    fn cow_good_ret_ty<'a>(input: &'a Cow<'a, str>) -> &str {
        //~^ ERROR: eliding a lifetime that's named elsewhere is confusing
        todo!()
    }
}

mod issue_11181 {
    extern "C" fn allowed(_v: &Vec<u32>) {}

    struct S;
    impl S {
        extern "C" fn allowed(_v: &Vec<u32>) {}
    }

    trait T {
        extern "C" fn allowed(_v: &Vec<u32>) {}
    }
}

mod issue_13308 {
    use std::ops::Deref;

    fn repro(source: &str, destination: &mut String) {
        source.clone_into(destination);
    }
    fn repro2(source: &str, destination: &mut String) {
        ToOwned::clone_into(source, destination);
    }

    fn h1(x: &<String as Deref>::Target) {}
    fn h2<T: Deref>(x: T, y: &T::Target) {}

    // Other cases that are still ok to lint and ideally shouldn't regress
    fn good(v1: &String, v2: &String) {
        //~^ ptr_arg
        //~| ptr_arg

        h1(v1);
        h2(String::new(), v2);
    }
}

mod issue_13489_and_13728 {
    // This is a no-lint from now on.
    fn foo(_x: &Vec<i32>) {
        todo!();
    }

    // But this still gives us a lint.
    fn foo_used(x: &Vec<i32>) {
        //~^ ptr_arg

        todo!();
    }

    // This is also a no-lint from now on.
    fn foo_local(x: &Vec<i32>) {
        let _y = x;

        todo!();
    }

    // But this still gives us a lint.
    fn foo_local_used(x: &Vec<i32>) {
        //~^ ptr_arg

        let y = x;

        todo!();
    }

    // This only lints once from now on.
    fn foofoo(_x: &Vec<i32>, y: &String) {
        //~^ ptr_arg

        todo!();
    }

    // And this is also a no-lint from now on.
    fn foofoo_local(_x: &Vec<i32>, y: &String) {
        let _z = y;

        todo!();
    }
}

mod issue_13489_and_13728_mut {
    // This is a no-lint from now on.
    fn bar(_x: &mut Vec<u32>) {
        todo!()
    }

    // But this still gives us a lint.
    fn bar_used(x: &mut Vec<u32>) {
        //~^ ptr_arg

        todo!()
    }

    // This is also a no-lint from now on.
    fn bar_local(x: &mut Vec<u32>) {
        let _y = x;

        todo!()
    }

    // But this still gives us a lint.
    fn bar_local_used(x: &mut Vec<u32>) {
        //~^ ptr_arg

        let y = x;

        todo!()
    }

    // This only lints once from now on.
    fn barbar(_x: &mut Vec<u32>, y: &mut String) {
        //~^ ptr_arg

        todo!()
    }

    // And this is also a no-lint from now on.
    fn barbar_local(_x: &mut Vec<u32>, y: &mut String) {
        let _z = y;

        todo!()
    }
}
