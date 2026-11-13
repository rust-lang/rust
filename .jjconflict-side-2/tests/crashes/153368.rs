//@ known-bug: #153368
//@ compile-flags: -Znext-solver=globally
trait Foo: Bar<T> + Bar<u32> {}
trait Bar<T> {
    fn bar(self) -> T;
}

fn test_infer_version(x: &dyn Foo) {
    x.bar()
}
