//@ known-bug: #137706
//@ needs-rustc-debug-assertions
trait A {
    fn b() -> impl IntoIterator<Item = ()>;
}

impl A<()> for dyn A {}
