//@ known-bug: #137187
use std::ops::Add;
trait A where
    *const Self: Add,
{
    const fn b(c: *const Self) -> <*const Self as Add>::Output {
        c + c
    }
}
