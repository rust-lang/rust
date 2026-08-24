// Regression test for https://github.com/rust-lang/rust/issues/152668.
// The impl omits type Output; sub returns Self::Output,
// and DefId must carry the type_of information of fn sub

use std::ops::Sub;
trait Vector2 {}
impl Sub for dyn Vector2 {
  //~^ ERROR not all trait items implemented, missing: `Output` [E0046]
  //~| ERROR the size for values of type `dyn Vector2 + 'static` cannot be known at compilation time [E0277]
    fn sub(self, rhs: Self) -> Self::Output {}
    //~^ ERROR: the size for values of type `dyn Vector2 + 'static` cannot be known at compilation time [E0277]
    //~| ERROR: the size for values of type `dyn Vector2 + 'static` cannot be known at compilation time [E0277]
    //~| ERROR: the size for values of type `dyn Vector2 + 'static` cannot be known at compilation time [E0277]
    //~| ERROR: the size for values of type `dyn Vector2 + 'static` cannot be known at compilation time [E0277]
    //~| ERROR: mismatched types [E0308]
}

fn main() {}
