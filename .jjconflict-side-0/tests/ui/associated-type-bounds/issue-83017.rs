//@ check-pass

trait TraitA<'a> {
    type AsA;
}

trait TraitB<'a, 'b> {
    type AsB;
}

trait TraitC<'a, 'b, 'c> {}

struct X;

impl<'a, 'b, 'c> TraitC<'a, 'b, 'c> for X {}

struct Y;

impl<'a, 'b> TraitB<'a, 'b> for Y {
    type AsB = X;
}

struct Z;

impl<'a> TraitA<'a> for Z {
    type AsA = Y;
}

fn foo<T>()
where
    for<'a> T: TraitA<'a, AsA: for<'b> TraitB<'a, 'b, AsB: for<'c> TraitC<'a, 'b, 'c>>>,
{
}

fn main() {
    foo::<Z>();
}
