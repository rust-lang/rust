trait Trait {
    fn baz(&self, _: Self) {}
    fn bat(&self) -> Self {}
}

fn bar(x: &dyn Trait) {} //~ ERROR the trait `Trait` cannot be made into an object

trait Other: Sized {}

fn foo(x: &dyn Other) {} //~ ERROR the trait `Other` cannot be made into an object

fn main() {}
