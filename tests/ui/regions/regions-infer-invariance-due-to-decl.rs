use std::marker;

struct Invariant<'a> {
    marker: marker::PhantomData<*mut &'a()>
}

fn to_same_lifetime<'r>(b_isize: Invariant<'r>) {
    let bj: Invariant<'r> = b_isize;
}

fn to_longer_lifetime<'r>(b_isize: Invariant<'r>) -> Invariant<'static> {
    b_isize
    //~^ ERROR lifetime may not live long enough
}

fn main() {
}
