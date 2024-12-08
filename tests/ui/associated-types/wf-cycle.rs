//@ check-pass

trait A {
    type U: Copy;
}

trait B where
    <Self::V as A>::U: Copy,
{
    type V: A;
}

fn main() {}
