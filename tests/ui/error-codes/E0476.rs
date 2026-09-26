#![feature(coerce_unsized)]
#![feature(unsize)]

use std::marker::Unsize;
use std::ops::CoerceUnsized;

struct Wrapper<T>(T);

impl<'a, 'b, T, S> CoerceUnsized<&'a Wrapper<T>> for &'b Wrapper<S> where S: Unsize<T> {}
//~^ ERROR lifetime of the source pointer does not outlive lifetime bound of the object type [E0476]
//~^^ ERROR E0119

fn main() {}
