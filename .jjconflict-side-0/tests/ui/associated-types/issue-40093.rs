//@ check-pass

pub trait Test {
    type Item;
    type Bundle: From<Self::Item>;
}

fn fails<T>()
where
    T: Test<Item = String>,
{
}

fn main() {}
