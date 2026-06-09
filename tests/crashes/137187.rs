//@ known-bug: #137187
use std::ops::Add;

const trait A where
    *const Self: Add,
{
    fn b(c: *const Self) -> <*const Self as Add>::Output {
        c + c
    }
}
