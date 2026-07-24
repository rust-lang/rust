//! This test used to hit an assertion instead of erroring and bailing out.

fn main() {
    let _ = [std::ops::Add::add, std::ops::Mul::mul, std::ops::Mul::mul as fn(_, &_)]; //~ ERROR non-primitive cast: `fn(_, _) -> <_ as Mul<_>>::Output {<_ as Mul<_>>::mul}` as `for<'a> fn(_, &'a _)`

}
