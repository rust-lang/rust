// Check that we properly suggest `r#fn` if we use it undeclared.
// https://github.com/rust-lang/rust/issues/143150
//
//@ edition: 2021

fn a(_: dyn Trait + 'r#fn) {
    //~^ ERROR use of undeclared lifetime name `'r#fn` [E0261]
}

trait Trait {}

struct Test {
    a: &'r#fn str,
    //~^ ERROR use of undeclared lifetime name `'r#fn` [E0261]
}

trait Trait1<T>
  where T: for<'a> Trait1<T> + 'r#fn { }
//~^ ERROR use of undeclared lifetime name `'r#fn` [E0261]

fn main() {}
