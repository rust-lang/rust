#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Foo<const N: usize, const M: usize = { N + 1 }>;
fn should_unify<const N: usize>() -> Foo<N> where [(); { N + 1 }]: {
    Foo::<N, { N + 1 }>
}
pub fn shouldnt_unify<const N: usize>() -> Foo<N>
where
    [(); { N + 1 }]:,
    [(); { N + 2 }]:, {
    Foo::<N, { N + 2 }>
    //~^ error: mismatched types
}

fn main() {}
