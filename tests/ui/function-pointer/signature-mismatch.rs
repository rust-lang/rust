//! This test used to hit an assertion instead of erroring and bailing out.

fn main() {
    let _ = [std::ops::Add::add, std::ops::Mul::mul, std::ops::Mul::mul as fn(_, &_)];
    //~^ ERROR: mismatched types
}
