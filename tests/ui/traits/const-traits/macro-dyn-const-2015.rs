// Ensure that the introduction of const trait bound didn't regress this code in Rust 2015.
// See also `mbe-const-trait-bound-theoretical-regression.rs`.

//@ edition: 2015
//@ check-pass

macro_rules! check {
    ($ty:ty) => { compile_error!("ty"); };
    (dyn $c:ident) => {};
}

check! { dyn const }

fn main() {}
