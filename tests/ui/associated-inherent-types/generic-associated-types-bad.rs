//@ revisions: item local region

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

#[derive(Clone, Copy)]
pub enum Ty {}

impl Ty {
    type Pr<T: Copy> = T;

    type Static<Q: 'static> = Q;
}

#[cfg(item)]
const _: Ty::Pr<String> = String::new(); //[item]~ ERROR the trait bound `String: Copy` is not satisfied
//[item]~^ ERROR the trait bound `String: Copy` is not satisfied

fn main() {
    #[cfg(local)]
    let _: Ty::Pr<Vec<()>>; //[local]~ ERROR the trait bound `Vec<()>: Copy` is not satisfied
}

fn user<'a>() {
    #[cfg(region)]
    let _: Ty::Static<&'a str> = ""; //[region]~ ERROR lifetime may not live long enough
}
