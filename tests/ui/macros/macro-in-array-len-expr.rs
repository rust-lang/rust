//! Regression test for <https://github.com/rust-lang/rust/issues/36816>.
//! Macro invocation in len position in arrays used to ICE.
//@ run-pass

macro_rules! m { () => { 1 } }
macro_rules! n { () => { 1 + m!() } }

fn main() {
    let _: [u32; n!()] = [0, 0];
}
