//@ check-pass

use std::any::Any;
use std::any::TypeId;

trait Private<P, R> {
    fn call(&self, p: P, r: R);
}
pub trait Public: Private<
//~^ WARNING trait `Private<<Self as Public>::P, <Self as Public>::R>` is more private than the item `Public`
    <Self as Public>::P,
    <Self as Public>::R
> {
    type P;
    type R;

    fn call_inner(&self);
}

fn main() {}
