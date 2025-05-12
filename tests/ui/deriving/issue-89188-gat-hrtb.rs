//@ check-pass

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

trait Trait<'s, 't, 'u> {}

#[derive(Clone)]
struct ShimMethod3<T: CallWithShim2 + 'static>(
    pub  &'static dyn for<'s> Fn(
        &'s mut T::Shim<dyn for<'t> Fn(&'s mut T::Shim<dyn for<'u> Trait<'s, 't, 'u>>)>,
    ),
);

trait Trait2 {
    type As;
}

#[derive(Clone)]
struct ShimMethod4<T: Trait2 + 'static>(pub &'static dyn for<'s> Fn(&'s mut T::As));

pub fn main() {}
