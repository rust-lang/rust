#![allow(
    unused,
    clippy::many_single_char_names,
    clippy::redundant_clone,
    clippy::if_then_panic
)]
#![warn(clippy::ptr_arg)]

use std::borrow::Cow;
use std::path::PathBuf;

fn do_vec(x: &Vec<i64>) {
    //Nothing here
}

fn do_vec_mut(x: &mut Vec<i64>) {
    // no error here
    //Nothing here
}

fn do_str(x: &String) {
    //Nothing here either
}

fn do_str_mut(x: &mut String) {
    // no error here
    //Nothing here either
}

fn do_path(x: &PathBuf) {
    //Nothing here either
}

fn do_path_mut(x: &mut PathBuf) {
    // no error here
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
    let h = g.clone(); // Alas, we cannot reliably detect this without following data.
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
    ) {
    }

    struct S {}
    impl S {
        fn allowed(
            #[allow(clippy::ptr_arg)] _v: &Vec<u32>,
            #[allow(clippy::ptr_arg)] _s: &String,
            #[allow(clippy::ptr_arg)] _p: &PathBuf,
            #[allow(clippy::ptr_arg)] _c: &Cow<[i32]>,
        ) {
        }
    }

    trait T {
        fn allowed(
            #[allow(clippy::ptr_arg)] _v: &Vec<u32>,
            #[allow(clippy::ptr_arg)] _s: &String,
            #[allow(clippy::ptr_arg)] _p: &PathBuf,
            #[allow(clippy::ptr_arg)] _c: &Cow<[i32]>,
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
