use std::marker;

struct invariant<'a> {
    marker: marker::PhantomData<*mut &'a()>
}

fn to_same_lifetime<'r>(b_isize: invariant<'r>) {
    let bj: invariant<'r> = b_isize;
}

fn to_longer_lifetime<'r>(b_isize: invariant<'r>) -> invariant<'static> {
    b_isize //~ ERROR mismatched types
}

fn main() {
}
