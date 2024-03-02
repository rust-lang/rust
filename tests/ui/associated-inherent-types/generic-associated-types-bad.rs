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
const _: Ty::Pr<String> = String::new(); //[item]~ ERROR trait `Copy` is not implemented for `String`
//[item]~^ ERROR trait `Copy` is not implemented for `String`

fn main() {
    #[cfg(local)]
    let _: Ty::Pr<Vec<()>>; //[local]~ ERROR trait `Copy` is not implemented for `Vec<()>`
}

fn user<'a>() {
    #[cfg(region)]
    let _: Ty::Static<&'a str> = ""; //[region]~ ERROR lifetime may not live long enough
}
