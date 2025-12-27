//@ check-fail
//@ known-bug: #148283

#![feature(unsize, coerce_unsized)]
#![allow(static_mut_refs)]
#![allow(dead_code)]
use std::ops::Deref;

static mut ACTIONS: Vec<&'static str> = Vec::new();

trait Trait {
    fn self_ty(&self);

    fn complete(&self) -> Vec<&'static str> {
        self.self_ty();
        let actions = unsafe { ACTIONS.clone() };
        unsafe { ACTIONS.clear() };
        actions
    }
}

macro_rules! do_trait_impl {
    ($self:ident, $self_ty:literal) => {
        impl Trait for $self {
            fn self_ty(&self) {
                unsafe { ACTIONS.push($self_ty); }
            }
        }
    }    
}

trait Dynable: Trait {}
struct Inner;
do_trait_impl!(Inner, "self_ty Inner");
impl Dynable for Inner {}

fn assert_arms(range: std::ops::RangeInclusive<usize>, f: impl Fn(usize) -> Vec<&'static str>, arm_coercions: &[&[&'static str]]) {
    let mut coercions = vec![];
    for i in range {
        let c = f(i);
        coercions.push(c);
    }
    for (i, (arm_coercion, coercion)) in std::iter::zip(arm_coercions.iter(), coercions.into_iter()).enumerate() {
        assert_eq!(arm_coercion, &coercion, "Arm {i} didn't match expectation:\n expected {:?}\n got {:?}", arm_coercion, coercion);
    }
}

struct Wrap4<T: ?Sized>(T);

struct O;
struct P;
struct Q;

do_trait_impl!(O, "self_ty O");
do_trait_impl!(P, "self_ty P");
do_trait_impl!(Q, "self_ty Q");

impl Deref for O {
    type Target = P;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref O->P"); }
        &P
    }
}
impl Deref for P {
    type Target = Q;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref P->Q"); }
        &Q
    }
}
impl Deref for Q {
    type Target = P;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref Q->P"); }
        &P
    }
}

fn order_lub() {
    let _a = match 0 {
        0 => &O      as &O,
        1 => &P      as &P,
        2 => &Q      as &Q,
        _ => loop {},
    };
}

fn main() {
    order_lub();
}
