#![warn(unused)]
#![deny(unused_variables)]
#![deny(unused_assignments)]
#![allow(dead_code, non_camel_case_types, trivial_numeric_casts, dropping_copy_types)]
#![feature(intrinsics)]

use std::ops::AddAssign;

fn f1(x: isize) {
    //~^ ERROR unused variable: `x`
}

fn f1b(x: &mut isize) {
    //~^ ERROR unused variable: `x`
}

#[allow(unused_variables)]
fn f1c(x: isize) {}

fn f1d() {
    let x: isize;
    //~^ ERROR unused variable: `x`
}

fn f2() {
    let x = 3;
    //~^ ERROR unused variable: `x`
}

fn f3() {
    let mut x = 3;
    //~^ ERROR variable `x` is assigned to, but never used
    x += 4;
    //~^ ERROR value assigned to `x` is never read
}

fn f3b() {
    let mut z = 3;
    //~^ ERROR variable `z` is assigned to, but never used
    loop {
        z += 4;
        //~^ ERROR value assigned to `z` is never read
    }
}

#[allow(unused_variables)]
fn f3c() {
    let mut z = 3;
    loop { z += 4; }
    //~^ ERROR value assigned to `z` is never read
}

#[allow(unused_variables)]
#[allow(unused_assignments)]
fn f3d() {
    let mut x = 3;
    x += 4;
}

fn f3e() {
    let a = 13;
    let mut z = 3;
    //~^ ERROR variable `z` is assigned to, but never used
    loop {
        z += a;
        //~^ ERROR value assigned to `z` is never read
    }
}

fn f4() {
    match Some(3) {
      Some(i) => {
        //~^ ERROR unused variable: `i`
      }
      None => {}
    }
}

enum tri {
    a(isize), b(isize), c(isize)
}

fn f4b() {
    match tri::a(3) {
      tri::a(i) | tri::b(i) | tri::c(i) => {
          //~^ ERROR unused variable: `i`
      }
    }
}

fn f4c() -> isize {
    match tri::a(3) {
      tri::a(i) | tri::b(i) | tri::c(i) => {
        i
      }
    }
}

fn f4d() {
    match tri::a(3) {
      tri::a(i) | tri::b(i) | tri::c(i) if i == 0 => {}
      _ => {}
    }
}

fn f5a() {
    for x in 1..10 { }
    //~^ ERROR unused variable: `x`
}

fn f5b() {
    for (x, _) in [1, 2, 3].iter().enumerate() { }
    //~^ ERROR unused variable: `x`
}

fn f5c() {
    for (_, x) in [1, 2, 3].iter().enumerate() {
    //~^ ERROR unused variable: `x`
        continue;
        drop(*x as i32); //~ WARNING unreachable statement
    }
}

struct View<'a>(&'a mut [i32]);

impl<'a> AddAssign<i32> for View<'a> {
    fn add_assign(&mut self, rhs: i32) {
        for lhs in self.0.iter_mut() {
            *lhs += rhs;
        }
    }
}

fn f6() {
    let mut array = [1, 2, 3];
    let mut v = View(&mut array);

    // ensure an error shows up for x even if lhs of an overloaded add assign

    let x;
    //~^ ERROR variable `x` is assigned to, but never used

    *({
        x = 0;  //~ ERROR value assigned to `x` is never read
        &mut v
    }) += 1;
}


struct MutRef<'a>(&'a mut i32);

impl<'a> AddAssign<i32> for MutRef<'a> {
    fn add_assign(&mut self, rhs: i32) {
        *self.0 += rhs;
    }
}

fn f7() {
    let mut a = 1;
    {
        // `b` does not trigger unused_variables
        let mut b = MutRef(&mut a);
        b += 1;
    }
    drop(a);
}

fn f8(a: u32) {
    let _ = a;
}

fn f9() {
    let mut a = 10;
    //~^ ERROR variable `a` is assigned to, but never used
    let b = 13;
    let c = 13;
    let d = 13;
    let e = 13;
    let f = 13;
    let g = 13;
    let h = 13;

    a += b;
    //~^ ERROR value assigned to `a` is never read
    a -= c;
    //~^ ERROR value assigned to `a` is never read
    a *= d;
    //~^ ERROR value assigned to `a` is never read
    a /= e;
    //~^ ERROR value assigned to `a` is never read
    a |= f;
    //~^ ERROR value assigned to `a` is never read
    a &= g;
    //~^ ERROR value assigned to `a` is never read
    a %= h;
    //~^ ERROR value assigned to `a` is never read
}

fn f9b() {
    let mut a = 10;
    let b = 13;
    let c = 13;
    let d = 13;
    let e = 13;
    let f = 13;
    let g = 13;
    let h = 13;

    a += b;
    a -= c;
    a *= d;
    a /= e;
    a |= f;
    a &= g;
    a %= h;

    let _ = a;
}

fn f9c() {
    let mut a = 10.;
    //~^ ERROR variable `a` is assigned to, but never used
    let b = 13.;
    let c = 13.;
    let d = 13.;
    let e = 13.;
    let f = 13.;

    a += b;
    //~^ ERROR value assigned to `a` is never read
    a -= c;
    //~^ ERROR value assigned to `a` is never read
    a *= d;
    //~^ ERROR value assigned to `a` is never read
    a /= e;
    //~^ ERROR value assigned to `a` is never read
    a %= f;
    //~^ ERROR value assigned to `a` is never read
}

fn f10<T>(mut a: T, b: T) {
    //~^ ERROR variable `a` is assigned to, but never used
    a = b;
    //~^ ERROR value assigned to `a` is never read
}

fn f10b<T>(mut a: Box<T>, b: Box<T>) { //~ ERROR variable `a` is assigned to, but never used
    a = b; //~ ERROR value assigned to `a` is never read
}

// unused params warnings are not needed for intrinsic functions without bodies
#[rustc_intrinsic]
unsafe fn simd_shuffle<T, I, U>(a: T, b: T, i: I) -> U;

fn main() {
}
