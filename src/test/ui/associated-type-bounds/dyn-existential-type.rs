// run-pass

#![feature(associated_type_bounds)]
#![feature(existential_type)]

use std::ops::Add;

trait Tr1 { type As1; fn mk(&self) -> Self::As1; }
trait Tr2<'a> { fn tr2(self) -> &'a Self; }

fn assert_copy<T: Copy>(x: T) { let _x = x; let _x = x; }
fn assert_static<T: 'static>(_: T) {}
fn assert_forall_tr2<T: for<'a> Tr2<'a>>(_: T) {}

struct S1;
#[derive(Copy, Clone)]
struct S2;
impl Tr1 for S1 { type As1 = S2; fn mk(&self) -> Self::As1 { S2 } }

type Et1 = Box<dyn Tr1<As1: Copy>>;
fn def_et1() -> Et1 { Box::new(S1) }
pub fn use_et1() { assert_copy(def_et1().mk()); }

type Et2 = Box<dyn Tr1<As1: 'static>>;
fn def_et2() -> Et2 { Box::new(S1) }
pub fn use_et2() { assert_static(def_et2().mk()); }

type Et3 = Box<dyn Tr1<As1: Clone + Iterator<Item: Add<u8, Output: Into<u8>>>>>;
fn def_et3() -> Et3 {
    struct A;
    impl Tr1 for A {
        type As1 = core::ops::Range<u8>;
        fn mk(&self) -> Self::As1 { 0..10 }
    };
    Box::new(A)
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

type Et4 = Box<dyn Tr1<As1: for<'a> Tr2<'a>>>;
fn def_et4() -> Et4 {
    #[derive(Copy, Clone)]
    struct A;
    impl Tr1 for A {
        type As1 = A;
        fn mk(&self) -> A { A }
    }
    impl<'a> Tr2<'a> for A {
        fn tr2(self) -> &'a Self { &A }
    }
    Box::new(A)
}
pub fn use_et4() { assert_forall_tr2(def_et4().mk()); }

fn main() {
    let _ = use_et1();
    let _ = use_et2();
    let _ = use_et3();
    let _ = use_et4();
}
