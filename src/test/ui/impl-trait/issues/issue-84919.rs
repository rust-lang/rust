trait Trait {}
impl Trait for () {}

fn foo<'a: 'a>() {
    let _x: impl Trait = ();
    //~^ `impl Trait` not allowed within variable binding [E0562]
}

fn main() {}
