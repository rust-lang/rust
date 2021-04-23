// check-pass

trait Foo {
    type Item;
}

trait Bar
where
    Self: Foo,
{
}

#[allow(dead_code)]
fn foo<M>(_m: M)
where
    M: Bar,
    M::Item: Send,
{
}

fn main() {}
