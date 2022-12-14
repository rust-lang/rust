// compile-flags: --document-private-items --document-hidden-items

// This is a regression test for #98009.

// @has "$.index[*][?(@.name=='this_compiles')]"
// @has "$.index[*][?(@.name=='this_does_not')]"
// @has "$.index[*][?(@.name=='Events')]"
// @has "$.index[*][?(@.name=='Other')]"
// @has "$.index[*][?(@.name=='Trait')]"

struct Events<R>(R);

struct Other;

pub trait Trait<T> {
    fn handle(value: T) -> Self;
}

impl<T, U> Trait<U> for T where T: From<U> {
    fn handle(_: U) -> Self { unimplemented!() }
}

impl<'a, R> Trait<&'a mut Events<R>> for Other {
    fn handle(_: &'a mut Events<R>) -> Self { unimplemented!() }
}

fn this_compiles<'a, R>(value: &'a mut Events<R>) {
    for _ in 0..3 {
        Other::handle(&mut *value);
    }
}

fn this_does_not<'a, R>(value: &'a mut Events<R>) {
    for _ in 0..3 {
        Other::handle(value);
    }
}
