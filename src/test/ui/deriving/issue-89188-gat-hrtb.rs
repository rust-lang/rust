// check-pass

#![feature(generic_associated_types)]

trait CallWithShim: Sized {
    type Shim<'s>
    where
        Self: 's;
}

#[derive(Clone)]
struct ShimMethod<T: CallWithShim + 'static>(pub &'static dyn for<'s> Fn(&'s mut T::Shim<'s>));

pub fn main() {}
