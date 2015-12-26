// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(rustc_attrs)]
// aux-build:mir_external_refs.rs


extern crate mir_external_refs as ext;

struct S(u8);
#[derive(Debug, PartialEq, Eq)]
struct Unit;

impl S {
    fn hey() -> u8 { 42 }
    fn hey2(&self) -> u8 { 44 }
}

trait X {
    fn hoy(&self) -> u8 { 43 }
    fn hoy2() -> u8 { 45 }
}

trait F<U> {
    fn f(self, other: U) -> u64;
}

impl F<u32> for u32 {
    fn f(self, other: u32) -> u64 { self as u64 + other as u64 }
}

impl F<u64> for u32 {
    fn f(self, other: u64) -> u64 { self as u64 - other }
}

impl F<u64> for u64 {
    fn f(self, other: u64) -> u64 { self * other }
}

impl F<u32> for u64 {
    fn f(self, other: u32) -> u64 { self ^ other as u64 }
}

trait T<I, O> {
    fn staticmeth(i: I, o: O) -> (I, O) { (i, o) }
}

impl<I, O> T<I, O> for O {}

impl X for S {}

enum E {
    U(u8)
}

const C: u8 = 84;
const C2: [u8; 5] = [42; 5];
const C3: [u8; 3] = [42, 41, 40];

fn regular() -> u8 {
    21
}

fn parametric<T>(u: T) -> T {
    u
}

#[rustc_mir]
fn t1() -> fn()->u8 {
    regular
}

#[rustc_mir]
fn t2() -> fn(u8)->E {
    E::U
}

#[rustc_mir]
fn t3() -> fn(u8)->S {
    S
}

#[rustc_mir]
fn t4() -> fn()->u8 {
    S::hey
}

#[rustc_mir]
fn t5() -> fn(&S)-> u8 {
    <S as X>::hoy
}


#[rustc_mir]
fn t6() -> fn()->u8{
    ext::regular_fn
}

#[rustc_mir]
fn t7() -> fn(u8)->ext::E {
    ext::E::U
}

#[rustc_mir]
fn t8() -> fn(u8)->ext::S {
    ext::S
}

#[rustc_mir]
fn t9() -> fn()->u8 {
    ext::S::hey
}

#[rustc_mir]
fn t10() -> fn(&ext::S)->u8 {
    <ext::S as ext::X>::hoy
}

#[rustc_mir]
fn t11() -> fn(u8)->u8 {
    parametric
}

#[rustc_mir]
fn t12() -> u8 {
    C
}

#[rustc_mir]
fn t13() -> [u8; 5] {
    C2
}

#[rustc_mir]
fn t13_2() -> [u8; 3] {
    C3
}

#[rustc_mir]
fn t14() -> fn()-> u8 {
    <S as X>::hoy2
}

#[rustc_mir]
fn t15() -> fn(&S)-> u8 {
    S::hey2
}

#[rustc_mir]
fn t16() -> fn(u32, u32)->u64 {
    F::f
}

#[rustc_mir]
fn t17() -> fn(u32, u64)->u64 {
    F::f
}

#[rustc_mir]
fn t18() -> fn(u64, u64)->u64 {
    F::f
}

#[rustc_mir]
fn t19() -> fn(u64, u32)->u64 {
    F::f
}

#[rustc_mir]
fn t20() -> fn(u64, u32)->(u64, u32) {
    <u32 as T<_, _>>::staticmeth
}

#[rustc_mir]
fn t21() -> Unit {
    Unit
}

#[rustc_mir]
fn t22() -> Option<u8> {
    None
}

fn main(){
    unsafe {
        assert_eq!(t1()(), regular());

        assert!(::std::mem::transmute::<_, *mut ()>(t2()) ==
                ::std::mem::transmute::<_, *mut ()>(E::U));
        assert!(::std::mem::transmute::<_, *mut ()>(t3()) ==
                ::std::mem::transmute::<_, *mut ()>(S));

        assert_eq!(t4()(), S::hey());
        let s = S(42);
        assert_eq!(t5()(&s), <S as X>::hoy(&s));


        assert_eq!(t6()(), ext::regular_fn());
        assert!(::std::mem::transmute::<_, *mut ()>(t7()) ==
                ::std::mem::transmute::<_, *mut ()>(ext::E::U));
        assert!(::std::mem::transmute::<_, *mut ()>(t8()) ==
                ::std::mem::transmute::<_, *mut ()>(ext::S));

        assert_eq!(t9()(), ext::S::hey());
        let sext = ext::S(6);
        assert_eq!(t10()(&sext), <ext::S as ext::X>::hoy(&sext));

        let p = parametric::<u8>;
        assert!(::std::mem::transmute::<_, *mut ()>(t11()) ==
                ::std::mem::transmute::<_, *mut ()>(p));

        assert_eq!(t12(), C);
        assert_eq!(t13(), C2);
        assert_eq!(t13_2(), C3);

        assert_eq!(t14()(), <S as X>::hoy2());
        assert_eq!(t15()(&s), S::hey2(&s));
        assert_eq!(t16()(10u32, 20u32), F::f(10u32, 20u32));
        assert_eq!(t17()(30u32, 10u64), F::f(30u32, 10u64));
        assert_eq!(t18()(50u64, 5u64), F::f(50u64, 5u64));
        assert_eq!(t19()(322u64, 2u32), F::f(322u64, 2u32));
        assert_eq!(t20()(123u64, 38u32), <u32 as T<_, _>>::staticmeth(123, 38));
        assert_eq!(t21(), Unit);
        assert_eq!(t22(), None);
    }
}
