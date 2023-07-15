// edition:2018
// https://github.com/rust-lang/rust/issues/113021

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub async fn a(path: &[(); Abc]) {
    //~^ ERROR cannot find value `Abc` in this scope
    match path {
        [] | _ => (),
    }
}

pub async fn b(path: &[(); Abc]) {
    //~^ ERROR cannot find value `Abc` in this scope
    match path {
        &[] | _ => (),
    }
}

pub async fn c(path: &[[usize; N_ISLANDS]; PrivateStruct]) -> usize {
    //~^ ERROR cannot find value `N_ISLANDS` in this scope
    //~| ERROR cannot find value `PrivateStruct` in this scope
    match path {
        [] | _ => 0,
    }
}


fn main() {}
