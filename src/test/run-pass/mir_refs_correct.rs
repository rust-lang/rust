// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

#[derive(PartialEq, Debug, Eq)]
enum CEnum {
    A = 0x321,
    B = 0x123
}

const C: u8 = 84;
const C2: [u8; 5] = [42; 5];
const C3: [u8; 3] = [42, 41, 40];
const C4: fn(u8) -> S = S;

fn regular() -> u8 {
    21
}

fn parametric<T>(u: T) -> T {
    u
}

fn t1() -> fn()->u8 {
    regular
}

fn t2() -> fn(u8)->E {
    E::U
}

fn t3() -> fn(u8)->S {
    S
}

fn t4() -> fn()->u8 {
    S::hey
}

fn t5() -> fn(&S)-> u8 {
    <S as X>::hoy
}


fn t6() -> fn()->u8{
    ext::regular_fn
}

fn t7() -> fn(u8)->ext::E {
    ext::E::U
}

fn t8() -> fn(u8)->ext::S {
    ext::S
}

fn t9() -> fn()->u8 {
    ext::S::hey
}

fn t10() -> fn(&ext::S)->u8 {
    <ext::S as ext::X>::hoy
}

fn t11() -> fn(u8)->u8 {
    parametric
}

fn t12() -> u8 {
    C
}

fn t13() -> [u8; 5] {
    C2
}

fn t13_2() -> [u8; 3] {
    C3
}

fn t14() -> fn()-> u8 {
    <S as X>::hoy2
}

fn t15() -> fn(&S)-> u8 {
    S::hey2
}

fn t16() -> fn(u32, u32)->u64 {
    F::f
}

fn t17() -> fn(u32, u64)->u64 {
    F::f
}

fn t18() -> fn(u64, u64)->u64 {
    F::f
}

fn t19() -> fn(u64, u32)->u64 {
    F::f
}

fn t20() -> fn(u64, u32)->(u64, u32) {
    <u32 as T<_, _>>::staticmeth
}

fn t21() -> Unit {
    Unit
}

fn t22() -> Option<u8> {
    None
}

fn t23() -> (CEnum, CEnum) {
    (CEnum::A, CEnum::B)
}

fn t24() -> fn(u8) -> S {
    C4
}

fn main() {
    assert_eq!(t1()(), regular());

    assert_eq!(t2() as *mut (), E::U as *mut ());
    assert_eq!(t3() as *mut (), S as *mut ());

    assert_eq!(t4()(), S::hey());
    let s = S(42);
    assert_eq!(t5()(&s), <S as X>::hoy(&s));


    assert_eq!(t6()(), ext::regular_fn());
    assert_eq!(t7() as *mut (), ext::E::U as *mut ());
    assert_eq!(t8() as *mut (), ext::S as *mut ());

    assert_eq!(t9()(), ext::S::hey());
    let sext = ext::S(6);
    assert_eq!(t10()(&sext), <ext::S as ext::X>::hoy(&sext));

    let p = parametric::<u8>;
    assert_eq!(t11() as *mut (), p as *mut ());

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
    assert_eq!(t23(), (CEnum::A, CEnum::B));
    assert_eq!(t24(), C4);
}
