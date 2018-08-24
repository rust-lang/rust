// Test related to #22779, but where the `'a:'b` relation
// appears in the trait too. No error here.

#![feature(rustc_attrs)]

trait Tr<'a, T> {
    fn renew<'b: 'a>(self) -> &'b mut [T] where 'a: 'b;
}

impl<'a, T> Tr<'a, T> for &'a mut [T] {
    fn renew<'b: 'a>(self) -> &'b mut [T] where 'a: 'b {
        &mut self[..]
    }
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
