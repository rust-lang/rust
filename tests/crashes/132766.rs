//@ known-bug: #132766

trait Trait {}
impl<'a> Trait for () {
    fn pass2<'a>() -> impl Trait2 {}
}

trait Trait2 {}
impl Trait2 for () {}
