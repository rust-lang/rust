// Regression test for issue #110856, where a borrowck error for a MIR tainted
// all promoteds within. This in turn generated a spurious "erroneous constant
// used" note when trying to evaluate a promoted.

pub fn f() -> u32 {
    let a = 0;
    a = &0 * &1 * &2 * &3;
    //~^ ERROR: cannot assign twice to immutable variable
    a
}

fn main() {}
