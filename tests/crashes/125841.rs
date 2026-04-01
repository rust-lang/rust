//@ known-bug: rust-lang/rust#125841
#![feature(non_lifetime_binders)]
fn take(id: impl for<T> Fn(T) -> T) {
    id(0);
    id("");
}

fn take2() -> impl for<T> Fn(T) -> T {
    |x| x
}

fn main() {
    take(|x| take2)
}
