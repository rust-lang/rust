// Test related to #22779. In this case, the impl is an inherent impl,
// so it doesn't have to match any trait, so no error results.

// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]

struct MySlice<'a, T:'a>(&'a mut [T]);

impl<'a, T> MySlice<'a, T> {
    fn renew<'b: 'a>(self) -> &'b mut [T] where 'a: 'b {
        &mut self.0[..]
    }
}


fn main() { }
