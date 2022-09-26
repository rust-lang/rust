// check-fail
// known-bug

// We almost certaintly want this to pass, but
// it's particularly difficult currently, because we need a way of specifying
// that `<Self::Base as Functor>::With<T> = Self` without using that when we have
// a `U`. See `https://github.com/rust-lang/rust/pull/92728` for a (hacky)
// solution. This might be better to just wait for Chalk.

pub trait Functor {
    type With<T>;

    fn fmap<T, U>(this: Self::With<T>) -> Self::With<U>;
}

pub trait FunctorExt<T>: Sized {
    type Base: Functor<With<T> = Self>;

    fn fmap<U>(self) {
        let arg: <Self::Base as Functor>::With<T>;
        let ret: <Self::Base as Functor>::With<U>;

        arg = self;
        ret = <Self::Base as Functor>::fmap(arg);
        //~^ type annotations needed
    }
}

fn main() {}
