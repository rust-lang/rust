#![feature(rustc_attrs)]
#![warn(clippy::eager_transmute)]
#![allow(clippy::transmute_int_to_non_zero, clippy::missing_transmute_annotations)]

use std::num::NonZero;

#[repr(u8)]
enum Opcode {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
}

struct Data {
    foo: &'static [u8],
    bar: &'static [u8],
}

fn int_to_opcode(op: u8) -> Option<Opcode> {
    (op < 4).then_some(unsafe { std::mem::transmute(op) })
    //~^ eager_transmute
}

fn f(op: u8, op2: Data, unrelated: u8) {
    true.then_some(unsafe { std::mem::transmute::<_, Opcode>(op) });
    (unrelated < 4).then_some(unsafe { std::mem::transmute::<_, Opcode>(op) });
    (op < 4).then_some(unsafe { std::mem::transmute::<_, Opcode>(op) });
    //~^ eager_transmute
    (op > 4).then_some(unsafe { std::mem::transmute::<_, Opcode>(op) });
    //~^ eager_transmute
    (op == 0).then_some(unsafe { std::mem::transmute::<_, Opcode>(op) });
    //~^ eager_transmute

    let _: Option<Opcode> = (op > 0 && op < 10).then_some(unsafe { std::mem::transmute(op) });
    //~^ eager_transmute
    let _: Option<Opcode> = (op > 0 && op < 10 && unrelated == 0).then_some(unsafe { std::mem::transmute(op) });
    //~^ eager_transmute

    // lint even when the transmutable goes through field/array accesses
    let _: Option<Opcode> = (op2.foo[0] > 0 && op2.foo[0] < 10).then_some(unsafe { std::mem::transmute(op2.foo[0]) });
    //~^ eager_transmute

    // don't lint: wrong index used in the transmute
    let _: Option<Opcode> = (op2.foo[0] > 0 && op2.foo[0] < 10).then_some(unsafe { std::mem::transmute(op2.foo[1]) });

    // don't lint: no check for the transmutable in the condition
    let _: Option<Opcode> = (op2.foo[0] > 0 && op2.bar[1] < 10).then_some(unsafe { std::mem::transmute(op2.bar[0]) });

    // don't lint: wrong variable
    let _: Option<Opcode> = (op2.foo[0] > 0 && op2.bar[1] < 10).then_some(unsafe { std::mem::transmute(op) });

    // range contains checks
    let _: Option<Opcode> = (1..=3).contains(&op).then_some(unsafe { std::mem::transmute(op) });
    //~^ eager_transmute
    let _: Option<Opcode> = ((1..=3).contains(&op) || op == 4).then_some(unsafe { std::mem::transmute(op) });
    //~^ eager_transmute
    let _: Option<Opcode> = (1..3).contains(&op).then_some(unsafe { std::mem::transmute(op) });
    //~^ eager_transmute
    let _: Option<Opcode> = (1..).contains(&op).then_some(unsafe { std::mem::transmute(op) });
    //~^ eager_transmute
    let _: Option<Opcode> = (..3).contains(&op).then_some(unsafe { std::mem::transmute(op) });
    //~^ eager_transmute
    let _: Option<Opcode> = (..=3).contains(&op).then_some(unsafe { std::mem::transmute(op) });
    //~^ eager_transmute

    // unrelated binding in contains
    let _: Option<Opcode> = (1..=3)
        .contains(&unrelated)
        .then_some(unsafe { std::mem::transmute(op) });
}

unsafe fn f2(op: u8) {
    unsafe {
        (op < 4).then_some(std::mem::transmute::<_, Opcode>(op));
        //~^ eager_transmute
    }
}

#[rustc_layout_scalar_valid_range_end(254)]
struct NonMaxU8(u8);
#[rustc_layout_scalar_valid_range_end(254)]
#[rustc_layout_scalar_valid_range_start(1)]
struct NonZeroNonMaxU8(u8);

macro_rules! impls {
    ($($t:ty),*) => {
        $(
            impl PartialEq<u8> for $t {
                fn eq(&self, other: &u8) -> bool {
                    self.0 == *other
                }
            }
            impl PartialOrd<u8> for $t {
                fn partial_cmp(&self, other: &u8) -> Option<std::cmp::Ordering> {
                    self.0.partial_cmp(other)
                }
            }
        )*
    };
}
impls!(NonMaxU8, NonZeroNonMaxU8);

fn niche_tests(v1: u8, v2: NonZero<u8>, v3: NonZeroNonMaxU8) {
    // u8 -> NonZero<u8>, do lint
    let _: Option<NonZero<u8>> = (v1 > 0).then_some(unsafe { std::mem::transmute(v1) });
    //~^ eager_transmute

    // NonZero<u8> -> u8, don't lint, target type has no niche and therefore a higher validity range
    let _: Option<u8> = (v2 > NonZero::new(1u8).unwrap()).then_some(unsafe { std::mem::transmute(v2) });

    // NonZero<u8> -> NonMaxU8, do lint, different niche
    let _: Option<NonMaxU8> = (v2 < NonZero::new(255u8).unwrap()).then_some(unsafe { std::mem::transmute(v2) });
    //~^ eager_transmute

    // NonZeroNonMaxU8 -> NonMaxU8, don't lint, target type has more validity
    let _: Option<NonMaxU8> = (v3 < 255).then_some(unsafe { std::mem::transmute(v2) });

    // NonZero<u8> -> NonZeroNonMaxU8, do lint, target type has less validity
    let _: Option<NonZeroNonMaxU8> = (v2 < NonZero::new(255u8).unwrap()).then_some(unsafe { std::mem::transmute(v2) });
    //~^ eager_transmute
}

fn main() {}
