//@ check-pass

use std::marker::PhantomData;
use std::ops::Drop;

// a >= b >= c >= a implies a = b = c
struct DropMe<'a: 'b, 'b: 'c, 'c: 'a>(
    PhantomData<&'a ()>,
    PhantomData<&'b ()>,
    PhantomData<&'c ()>,
);

// a >= b, a >= c, b >= a, c >= a implies a = b = c
impl<'a: 'b + 'c, 'b: 'a, 'c: 'a> Drop for DropMe<'a, 'b, 'c> {
    fn drop(&mut self) {}
}

fn main() {}
