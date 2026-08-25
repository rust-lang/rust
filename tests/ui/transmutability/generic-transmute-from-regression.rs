//! Regression test for: <https://github.com/rust-lang/rust/issues/153755>
#![feature(transmutability)]

fn foo<T, U>(x: T) -> U {
    unsafe {
        std::mem::TransmuteFrom::transmute(x)
        //~^ ERROR: the trait bound `U: TransmuteFrom<T, _>` is not satisfied [E0277]
    }
}

fn main() {}
