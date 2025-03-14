//@ known-bug: #137895
trait A {
    fn b() -> impl ?Sized + 'a;
}

impl A for dyn A {}
