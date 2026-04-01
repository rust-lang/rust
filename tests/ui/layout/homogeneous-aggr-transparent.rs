#![feature(rustc_attrs)]
#![feature(transparent_unions)]
use std::marker::PhantomData;

// Regression test for #115664. We want to ensure that `repr(transparent)` wrappers do not affect
// the result of `homogeneous_aggregate`.

type Tuple = (f32, f32, f32);

struct Zst;

#[repr(transparent)]
struct Wrapper1<T>(T);
#[repr(transparent)]
struct Wrapper2<T>((), Zst, T);
#[repr(transparent)]
struct Wrapper3<T>(T, [u8; 0], PhantomData<u64>);
#[repr(transparent)]
union WrapperUnion<T: Copy> {
    nothing: (),
    something: T,
}

#[rustc_layout(homogeneous_aggregate)]
pub type Test0 = Tuple;
//~^ ERROR homogeneous_aggregate: Ok(Homogeneous(Reg { kind: Float, size: Size(4 bytes) }))

#[rustc_layout(homogeneous_aggregate)]
pub type Test1 = Wrapper1<Tuple>;
//~^ ERROR homogeneous_aggregate: Ok(Homogeneous(Reg { kind: Float, size: Size(4 bytes) }))

#[rustc_layout(homogeneous_aggregate)]
pub type Test2 = Wrapper2<Tuple>;
//~^ ERROR homogeneous_aggregate: Ok(Homogeneous(Reg { kind: Float, size: Size(4 bytes) }))

#[rustc_layout(homogeneous_aggregate)]
pub type Test3 = Wrapper3<Tuple>;
//~^ ERROR homogeneous_aggregate: Ok(Homogeneous(Reg { kind: Float, size: Size(4 bytes) }))

#[rustc_layout(homogeneous_aggregate)]
pub type Test4 = WrapperUnion<Tuple>;
//~^ ERROR homogeneous_aggregate: Ok(Homogeneous(Reg { kind: Float, size: Size(4 bytes) }))

fn main() {}
