#![allow(unused)]

use std::ptr::NonNull;

// Should only warn for `s`.
fn foo(s: &mut Vec<u32>, b: &u32, x: &mut u32) {
    *x += *b + s.len() as u32;
}

// Should not warn.
fn foo2(s: &mut Vec<u32>) {
    s.push(8);
}

// Should not warn because we return it.
fn foo3(s: &mut Vec<u32>) -> &mut Vec<u32> {
    s
}

// Should not warn because `s` is used as mutable.
fn foo4(s: &mut Vec<u32>) {
    Vec::push(s, 4);
}

// Should not warn.
fn foo5(s: &mut Vec<u32>) {
    foo2(s);
}

// Should warn.
fn foo6(s: &mut Vec<u32>) {
    non_mut_ref(s);
}

fn non_mut_ref(_: &Vec<u32>) {}

struct Bar;

impl Bar {
    // Should not warn on `&mut self`.
    fn bar(&mut self) {}

    // Should warn about `vec`
    fn mushroom(&self, vec: &mut Vec<i32>) -> usize {
        vec.len()
    }

    // Should warn about `vec` (and not `self`).
    fn badger(&mut self, vec: &mut Vec<i32>) -> usize {
        vec.len()
    }
}

trait Babar {
    // Should not warn here since it's a trait method.
    fn method(arg: &mut u32);
}

impl Babar for Bar {
    // Should not warn here since it's a trait method.
    fn method(a: &mut u32) {}
}

// Should not warn (checking variable aliasing).
fn alias_check(s: &mut Vec<u32>) {
    let mut alias = s;
    let mut alias2 = alias;
    let mut alias3 = alias2;
    alias3.push(0);
}

// Should not warn (checking variable aliasing).
fn alias_check2(mut s: &mut Vec<u32>) {
    let mut alias = &mut s;
    alias.push(0);
}

struct Mut<T> {
    ptr: NonNull<T>,
}

impl<T> Mut<T> {
    // Should not warn because `NonNull::from` also accepts `&mut`.
    fn new(ptr: &mut T) -> Self {
        Mut {
            ptr: NonNull::from(ptr),
        }
    }
}

// Should not warn.
fn unused(_: &mut u32, _b: &mut u8) {}

fn main() {
    let mut u = 0;
    let mut v = vec![0];
    foo(&mut v, &0, &mut u);
    foo2(&mut v);
    foo3(&mut v);
    foo4(&mut v);
    foo5(&mut v);
    alias_check(&mut v);
    alias_check2(&mut v);
    println!("{u}");
}
