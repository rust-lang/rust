// Example cycle where a bound on `T` uses a shorthand for `T`. This
// creates a cycle because we have to know the bounds on `T` to figure
// out what trait defines `Item`, but we can't know the bounds on `T`
// without knowing how to handle `T::Item`.
//
// Note that in the future cases like this could perhaps become legal,
// if we got more fine-grained about our cycle detection or changed
// how we handle `T::Item` resolution.

use std::ops::Add;

// Preamble.
trait Trait { type Item; }

struct A<T>
    where T : Trait,
          T : Add<T::Item>
    //~^ ERROR cycle detected
{
    data: T
}

fn main() {
}
