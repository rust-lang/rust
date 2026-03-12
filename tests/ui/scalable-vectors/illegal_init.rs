#![allow(incomplete_features, internal_features)]
#![feature(rustc_attrs)]

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
struct svint32_t(i32);
fn main() {
    let foo = svint32_t(1);
    //~^ ERROR: scalable vector types cannot be initialised using their constructor
}
