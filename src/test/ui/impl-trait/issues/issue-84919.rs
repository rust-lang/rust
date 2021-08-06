trait Trait {}
impl Trait for () {}

fn foo<'a: 'a>() {
    let _x: impl Trait = ();
    //~^ `impl Trait` not allowed outside of function and method return types
}

fn main() {}
