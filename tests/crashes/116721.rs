//@ known-bug: #116721
//@ compile-flags: -Zmir-opt-level=3 --emit=mir
fn hey<T>(it: &[T])
where
    [T]: Clone,
{
}

fn main() {}
