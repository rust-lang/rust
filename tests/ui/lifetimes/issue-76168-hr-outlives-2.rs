//@ edition:2018
//@ check-pass

trait Trait<Input> {
    type Output;
}

async fn walk<F>(filter: F)
where
    for<'a> F: Trait<&'a u32> + 'a,
    for<'a> <F as Trait<&'a u32>>::Output: 'a,
{
}

async fn walk2<F: 'static>(filter: F)
where
    for<'a> F: Trait<&'a u32> + 'a,
    for<'a> <F as Trait<&'a u32>>::Output: 'a,
{
}

fn main() {}
