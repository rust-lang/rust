//@ edition: 2021
fn a(_: dyn Trait + 'r#fn) { //~ ERROR use of undeclared lifetime name `'fn` [E0261]

}

trait Trait {}

#[derive(Eq, PartialEq)]
struct Test {
    a: &'r#fn str, 
    //~^ ERROR use of undeclared lifetime name `'fn` [E0261]
    //~| ERROR use of undeclared lifetime name `'fn` [E0261]
}

trait Trait1<T>
  where T: for<'a> Trait1<T> + 'r#fn { } //~ ERROR use of undeclared lifetime name `'fn` [E0261]



fn main() {}
