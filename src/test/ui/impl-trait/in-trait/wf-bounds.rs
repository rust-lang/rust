// issue #101663

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

trait Wf<T> {}

trait Uwu {
    fn nya() -> impl Wf<Vec<[u8]>>;
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

    fn nya2() -> impl Wf<[u8]>;
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

fn main() {}
