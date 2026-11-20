//@ known-bug: #112905
//@ check-pass

// Classified as an issue with implied bounds:
// https://github.com/rust-lang/rust/issues/112905#issuecomment-1757847998

/// Note: this is sound! It's the "type witness" pattern (here, lt witness).
mod some_lib {
    use super::T;

    /// Invariant in `'a` and `'b` for soundness.
    pub struct LtEq<'a, 'b>(::std::marker::PhantomData<*mut Self>);

    impl<'a, 'b> LtEq<'a, 'b> {
        pub fn new() -> LtEq<'a, 'a> {
            LtEq(<_>::default())
        }

        pub fn eq(&self) -> impl 'static + Fn(T<'a>) -> T<'b> {
            |a| unsafe { ::std::mem::transmute::<T<'a>, T<'b>>(a) }
        }
    }
}

use some_lib::LtEq;
use std::{any::Any, cell::Cell};

/// Feel free to choose whatever you want, here.
type T<'lt> = Cell<&'lt str>;

fn exploit<'a, 'b>(a: T<'a>) -> T<'b> {
    let f = LtEq::<'a, 'a>::new().eq();
    let any = Box::new(f) as Box<dyn Any>;

    let new_f = None.map(LtEq::<'a, 'b>::eq);

    fn downcast_a_to_type_of_new_f<F: 'static>(any: Box<dyn Any>, _: Option<F>) -> F {
        *any.downcast().unwrap_or_else(|_| unreachable!())
    }

    let f = downcast_a_to_type_of_new_f(any, new_f);

    f(a)
}

fn main() {
    let r: T<'static> = {
        let local = String::from("…");
        let a: T<'_> = Cell::new(&local[..]);
        exploit(a)
    };
    dbg!(r.get());
}
