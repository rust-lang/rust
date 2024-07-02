#[allow(dead_code)]
fn foo<M>(_m: M)
where
    M::Item: Temp,
    //~^ ERROR cannot find trait `Temp`
    //~| ERROR associated type `Item` not found for `M` [E0220]
{
}

fn main() {}
