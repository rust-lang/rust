#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn consts<const N: i32>() -> i32 {
        N
    }
    pub fn late<'a>(x: &'a u8) -> u8 {
        *x
    }
    pub fn bounds<T: Clone>(_: T) {}
}

// FIXME(fn_delegation): this is supposed to work eventually
reuse to_reuse::consts;
//~^ ERROR  type annotations needed
reuse to_reuse::late;
reuse to_reuse::bounds;

fn main() {
    late::<'static>(&0u8);
    //~^ ERROR cannot specify lifetime arguments explicitly if late bound lifetime parameters are present

    struct S;
    bounds(S);
    //~^ ERROR the trait bound `S: Clone` is not satisfied
}
