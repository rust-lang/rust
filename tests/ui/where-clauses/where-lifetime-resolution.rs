trait Trait1<'a> {}
trait Trait2<'a, 'b> {}

fn f() where
    for<'a> dyn Trait1<'a>: Trait1<'a>, // OK
    (dyn for<'a> Trait1<'a>): Trait1<'a>,
    //~^ ERROR use of undeclared lifetime name `'a`
    for<'a> dyn for<'b> Trait2<'a, 'b>: Trait2<'a, 'b>,
    //~^ ERROR use of undeclared lifetime name `'b`
{}

fn main() {}
