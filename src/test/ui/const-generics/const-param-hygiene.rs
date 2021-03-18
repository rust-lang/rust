// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

macro_rules! bar {
    ($($t:tt)*) => { impl<const N: usize> $($t)* };
}

macro_rules! baz {
    ($t:tt) => { fn test<const M: usize>(&self) -> usize { $t } };
}

struct Foo<const N: usize>;

bar!(Foo<N> { baz!{ M } });

fn main() {
    assert_eq!(Foo::<7>.test::<3>(), 3);
}
