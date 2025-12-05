// issue: 114146

trait Foo {
    fn bar<'other: 'a>() -> impl Sized + 'a {}
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| ERROR use of undeclared lifetime name `'a`
}

fn main() {}
