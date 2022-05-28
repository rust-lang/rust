trait Trait {}
impl Trait for () {}

fn foo<'a: 'a>() {
    let _x: impl Trait = ();
    //~^ `impl Trait` only allowed in function and inherent method return types
}

fn main() {}
