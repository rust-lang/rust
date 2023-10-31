// aux-crate:aux=assoc-inherent-unstable.rs
// edition: 2021

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

type Data = aux::Owner::Data; //~ ERROR use of unstable library feature 'data'

fn main() {}
