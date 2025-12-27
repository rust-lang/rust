//@ run-pass

#![feature(unsize, coerce_unsized)]
#![allow(static_mut_refs)]
#![allow(dead_code)]
#![allow(unused_macros)]

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

impl<'b, T: ?Sized + std::marker::Unsize<U> + std::ops::CoerceUnsized<U>, U: ?Sized>
    std::ops::CoerceUnsized<Wrap4<U>> for Wrap4<T> {}


type L = Wrap4<Inner>;
type M = Wrap4<dyn Dynable + Send>;
type N = Wrap4<dyn Dynable>;

do_trait_impl!(L, "self_ty L");
do_trait_impl!(M, "self_ty M");
do_trait_impl!(N, "self_ty N");

fn order_lub() {
    let a = match 0 {
        0 => &Wrap4(Inner)      as &L,
        1 => &Wrap4(Inner)      as &M,
        2 => &Wrap4(Inner)      as &N,
        _ => loop {},
    };
    assert_eq!(a.complete(), vec!["self_ty N"]);
    assert_arms(
        0..=2,
        |i| match i {
            0 => &Wrap4(Inner)      as &L,
            2 => &Wrap4(Inner)      as &N,
            1 => &Wrap4(Inner)      as &M,
            _ => loop {},
        }.complete(),
        &[
            &["self_ty N"],
            &["self_ty N"],
            &["self_ty N"],
        ],
    );
}

fn main() {
    order_lub();
}
