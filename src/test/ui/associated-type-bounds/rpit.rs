// run-pass

#![feature(associated_type_bounds)]

use std::ops::Add;

trait Tr1 { type As1; fn mk(self) -> Self::As1; }
trait Tr2<'a> { fn tr2(self) -> &'a Self; }

fn assert_copy<T: Copy>(x: T) { let _x = x; let _x = x; }
fn assert_static<T: 'static>(_: T) {}
fn assert_forall_tr2<T: for<'a> Tr2<'a>>(_: T) {}

struct S1;
#[derive(Copy, Clone)]
struct S2;
impl Tr1 for S1 { type As1 = S2; fn mk(self) -> Self::As1 { S2 } }

fn def_et1() -> impl Tr1<As1: Copy> { S1 }
pub fn use_et1() { assert_copy(def_et1().mk()); }

fn def_et2() -> impl Tr1<As1: 'static> { S1 }
pub fn use_et2() { assert_static(def_et2().mk()); }

fn def_et3() -> impl Tr1<As1: Clone + Iterator<Item: Add<u8, Output: Into<u8>>>> {
    struct A;
    impl Tr1 for A {
        type As1 = core::ops::Range<u8>;
        fn mk(self) -> Self::As1 { 0..10 }
    };
    A
}

pub fn use_et3() {
    let _0 = def_et3().mk().clone();
    let mut s = 0u8;
    for _1 in _0 {
        let _2 = _1 + 1u8;
        s += _2.into();
    }
    assert_eq!(s, (0..10).map(|x| x + 1).sum());
}

fn def_et4() -> impl Tr1<As1: for<'a> Tr2<'a>> {
    #[derive(Copy, Clone)]
    struct A;
    impl Tr1 for A {
        type As1 = A;
        fn mk(self) -> A { A }
    }
    impl<'a> Tr2<'a> for A {
        fn tr2(self) -> &'a Self { &A }
    }
    A
}

pub fn use_et4() { assert_forall_tr2(def_et4().mk()); }

fn main() {
    let _ = use_et1();
    let _ = use_et2();
    let _ = use_et3();
    let _ = use_et4();
}
