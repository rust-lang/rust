//@ check-pass

#![feature(trait_alias)]

trait SomeTrait {
    fn map(&self) {}
}

impl<T> SomeTrait for Option<T> {}

trait SomeAlias = SomeTrait;

fn main() {
    let x = Some(123);
    // This should resolve to the trait impl for Option
    Option::map(x, |z| z);
    // This should resolve to the trait impl for SomeTrait
    SomeTrait::map(&x);
}
