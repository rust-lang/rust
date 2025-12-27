//@ check-fail
//@ known-bug: #148283

#![allow(static_mut_refs)]
#![allow(dead_code)]
#![allow(unused_macros)]
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

struct Wrap2<T: ?Sized>(T);

struct E;
type F = Wrap2<[i32; 0]>;
struct G;
type H = Wrap2<[i32]>;

do_trait_impl!(E, "self_ty E");
do_trait_impl!(F, "self_ty F");
do_trait_impl!(G, "self_ty G");
do_trait_impl!(H, "self_ty H");

impl Deref for E {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref E->F"); }
        &Wrap2([])
    }
}
impl Deref for F {
    type Target = G;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref F->G"); }
        &G
    }
}
impl Deref for H {
    type Target = G;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref H->G"); }
        &G
    }
}

fn order_lub() {
    let a = match 0 {
        0 => &E          as &E,
        1 => &Wrap2([])  as &F,
        2 => &G          as &G,
        3 => &Wrap2([])  as &H,
        _ => loop {},
    };
    assert_eq!(a.complete(), vec!["deref E->F", "deref F->G", "self_ty G"]);
    let b = match 0 {
        0 => &E          as &E,
        1 => &Wrap2([])  as &F,
        3 => &Wrap2([])  as &H,
        2 => &G          as &G,
        _ => loop {},
    };
    assert_eq!(b.complete(), vec!["self_ty UnsizedArray"]);
}

fn main() {
    order_lub();
}
