// check-pass
#![feature(rustc_attrs)]
#![feature(negative_impls)]

#[rustc_auto_trait]
trait NotSame {}

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
