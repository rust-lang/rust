// revisions: min_const_fn const_fn

#![cfg_attr(const_fn, feature(const_fn))]

enum E {
    V(i32),
}

const EXTERNAL_CONST: Option<i32> = {Some}(1);
//[min_const_fn]~^ ERROR is not yet stable as a const fn
//[const_fn]~^^ ERROR is not yet stable as a const fn
const LOCAL_CONST: E = {E::V}(1);
//[min_const_fn]~^ ERROR is not yet stable as a const fn
//[const_fn]~^^ ERROR is not yet stable as a const fn

const fn external_fn() {
    let _ = {Some}(1);
    //[min_const_fn]~^ ERROR is not yet stable as a const fn
    //[const_fn]~^^ ERROR is not yet stable as a const fn
}

const fn local_fn() {
    let _ = {E::V}(1);
    //[min_const_fn]~^ ERROR is not yet stable as a const fn
    //[const_fn]~^^ ERROR is not yet stable as a const fn
}

fn main() {}
