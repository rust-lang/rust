This error indicates that there is a mismatch between generic parameters and
impl Trait parameters in a trait declaration versus its impl.

```compile_fail,E0643
trait Foo {
    fn foo(&self, _: &impl Iterator);
}
impl Foo for () {
    fn foo<U: Iterator>(&self, _: &U) { } // error method `foo` has incompatible
                                          // signature for trait
}
```
