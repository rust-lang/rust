#![feature(never_type)]

fn make_up_a_value<T>() -> T {
    unsafe {
    //~^ ERROR mismatched types
        let x: *const ! = 0 as _;
        let _: ! = *x;
        // Since `*x` "diverges" in HIR, but doesn't count as a read in MIR, this
        // is unsound since we act as if it diverges but it doesn't.
    }
}

fn main() {}
