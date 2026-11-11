//@ edition: 2021
//@ compile-flags: -Znext-solver -Arecursion_depth_exceeding_limit

#![forbid(unsafe_code)]
#![feature(field_projections)]
#![allow(incomplete_features)]

use core::field::{Field, field_of};

trait Chain {
    type A;
}

impl Chain for () {
    type A = u8;
}

struct Link<T>(T);

impl<T: Chain> Chain for Link<T> {
    type A = <T as Chain>::A;
}

type L4<T> = Link<Link<Link<Link<T>>>>;
type L16<T> = L4<L4<L4<L4<T>>>>;
type L64<T> = L16<L16<L16<L16<T>>>>;
type Deep = L64<L64<L64<()>>>; // 192 links

struct S<Y: Chain> {
    f: <Y as Chain>::A,
}

trait Tr {
    type Out;
}

impl<Y: Chain> Tr for S<Y>
where
    field_of!(S<Y>, f): Field,
{
    type Out = u8;
}

impl Tr for S<Deep> { //~ ERROR conflicting implementations of trait `Tr`
    type Out = [u8; 16];
}

fn poly<Y: Chain>() -> <S<Y> as Tr>::Out {
    0
}

fn main() {
    let v = poly::<Deep>();
    println!("{:?}", v);
}
