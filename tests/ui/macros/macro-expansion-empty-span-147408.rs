//@ check-pass
//@ compile-flags: -Afor_loops_over_fallibles -Warray_into_iter
//@ edition: 2015..2021

fn main() {
    macro_rules! mac {
        (iter $e:expr) => {
            $e.iter()
        };
        (into_iter $e:expr) => {
            $e.into_iter() //~ WARN this method call resolves to
            //~^ WARN this changes meaning in Rust 2021
        };
        (next $e:expr) => {
            $e.iter().next()
        };
    }

    for _ in dbg!([1, 2]).iter() {}
    for _ in dbg!([1, 2]).into_iter() {} //~ WARN this method call resolves to
    //~^ WARN this changes meaning in Rust 2021
    for _ in mac!(iter [1, 2]) {}
    for _ in mac!(into_iter [1, 2]) {}
    for _ in mac!(next [1, 2]) {}
}
