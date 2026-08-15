//! regression test for <https://github.com/rust-lang/rust/issues/19601>
//@ check-pass

trait A<T> {}
struct B<T>
where
    B<T>: A<B<T>>,
{
    t: T,
}

fn main() {}
