// run-pass

struct S<T> {
    t : T,
    s : Box<S<fn(u : T)>>
}

fn f(x : S<u32>) {}

fn main () {}
