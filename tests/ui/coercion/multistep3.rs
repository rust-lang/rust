//@ check-fail

#![feature(unsize, coerce_unsized)]
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

struct Wrap3<T: ?Sized>(T);

impl<'b, T: ?Sized + std::marker::Unsize<U> + std::ops::CoerceUnsized<U>, U: ?Sized>
    std::ops::CoerceUnsized<Wrap3<U>> for Wrap3<T> {}

type I = Wrap3<Inner>;
type J = Wrap3<dyn Dynable + Send>;
type K = Wrap3<dyn Dynable>;

do_trait_impl!(I, "self_ty I");
do_trait_impl!(J, "self_ty J");
do_trait_impl!(K, "self_ty K");

impl Deref for K {
    type Target = J;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref K->J"); }
        &Wrap3(Inner)
    }
}

fn order_lub() {
assert_arms(
    0..=2,
        |i| match i {
            0 => &Wrap3(Inner)      as &I,
            1 => &Wrap3(Inner)      as &J,
            2 => &Wrap3(Inner)      as &K,
            //~^ ERROR `match` arms have incompatible types
            _ => loop {},
        }.complete(),
        &[
            &["self_ty J"],
            &["self_ty J"],
            &["deref K->J", "self_ty J"],
        ],
    );
    assert_arms(
        0..=2,
        |i| match i {
            0 => &Wrap3(Inner)      as &I,
            1 => &Wrap3(Inner)      as &K,
            2 => &Wrap3(Inner)      as &J,
            //~^ ERROR `match` arms have incompatible types
            _ => loop {},
        }.complete(),
        &[
            &["self_ty K"],
            &["self_ty K"],
            &["self_ty K"],
        ],
    );
}

fn main() {
    order_lub();
}
