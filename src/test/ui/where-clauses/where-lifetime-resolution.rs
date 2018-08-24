trait Trait1<'a> {}
trait Trait2<'a, 'b> {}

fn f() where
    for<'a> Trait1<'a>: Trait1<'a>, // OK
    (for<'a> Trait1<'a>): Trait1<'a>,
    //~^ ERROR use of undeclared lifetime name `'a`
    for<'a> for<'b> Trait2<'a, 'b>: Trait2<'a, 'b>,
    //~^ ERROR use of undeclared lifetime name `'b`
    //~| ERROR nested quantification of lifetimes
{}

fn main() {}
