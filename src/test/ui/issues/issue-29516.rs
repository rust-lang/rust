// check-pass
#![feature(optin_builtin_traits)]
#![feature(negative_impls)]

auto trait NotSame {}

impl<A> !NotSame for (A, A) {}

trait OneOfEach {}

impl<A> OneOfEach for (A,) {}

impl<A, B> OneOfEach for (A, B)
where
    (B,): OneOfEach,
    (A, B): NotSame,
{
}

fn main() {}
