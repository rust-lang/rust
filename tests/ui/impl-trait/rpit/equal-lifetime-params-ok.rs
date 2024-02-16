//@ check-pass

// related to #113916, check that using RPITs in functions with lifetime params
// which are constrained to be equal compiles.

trait Trait<'a, 'b> {}
impl Trait<'_, '_> for () {}
fn pass<'a: 'b, 'b: 'a>() -> impl Trait<'a, 'b> {
    (|| {})()
}

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    fn bar<'b: 'a>(&'b self) -> impl Trait<'a, 'b> {
        let _: &'a &'b &'a ();
    }
}

fn main() {}
