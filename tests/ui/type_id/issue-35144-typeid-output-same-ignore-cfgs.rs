// #35144: output should be the same with or without enabling the feature
//
// edition:2021
// aux-build:a.rs
// aux-build:b.rs
// run-pass
// compile-flags: --extern a --extern b --cfg feature="afirst" -A unused_imports -A dead_code
#![feature(core_intrinsics, rustc_private)]

#[cfg(afirst)]
use a;
use b;
#[cfg(not(afirst))]
use a;

use std::intrinsics::type_id;

fn main() {
    println!(
        "{:?} {:?}",
        type_id::<a::a::Arena<()>>(),
        type_id::<dyn Iterator<Item = a::a::Arena<()>>>()
    );
}
