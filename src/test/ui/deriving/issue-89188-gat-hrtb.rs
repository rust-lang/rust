// check-pass

#![feature(generic_associated_types)]

trait CallWithShim: Sized {
    type Shim<'s>
    where
        Self: 's;
}

#[derive(Clone)]
struct ShimMethod<T: CallWithShim + 'static>(pub &'static dyn for<'s> Fn(&'s mut T::Shim<'s>));

trait CallWithShim2: Sized {
    type Shim<T>;
}

struct S<'s>(&'s ());

#[derive(Clone)]
struct ShimMethod2<T: CallWithShim2 + 'static>(pub &'static dyn for<'s> Fn(&'s mut T::Shim<S<'s>>));

trait Trait<'s, 't> {}

#[derive(Clone)]
struct ShimMethod3<T: CallWithShim2 + 'static>(
    pub &'static dyn for<'s> Fn(&'s mut T::Shim<dyn for<'t> Trait<'s, 't>>),
);

trait Trait2 {
    type As;
}

#[derive(Clone)]
struct ShimMethod4<T: Trait2 + 'static>(pub &'static dyn for<'s> Fn(&'s mut T::As));

pub fn main() {}
