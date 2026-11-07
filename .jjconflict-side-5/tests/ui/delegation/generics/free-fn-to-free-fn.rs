#![feature(fn_delegation)]

mod to_reuse {
    pub fn consts<const N: i32>() -> i32 {
        N
    }
    pub fn late<'a>(x: &'a u8) -> u8 {
        *x
    }
    pub fn bounds<T: Clone>(_: T) {}
}

reuse to_reuse::consts;
reuse to_reuse::late;
reuse to_reuse::bounds;

fn main() {
    late::<'static>(&0u8);
    //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied

    struct S;
    bounds(S);
    //~^ ERROR the trait bound `S: Clone` is not satisfied
}
