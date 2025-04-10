trait Trait {}
impl Trait for () {}

fn foo<'a: 'a>() {
    let _x: impl Trait = ();
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
}

fn main() {}
