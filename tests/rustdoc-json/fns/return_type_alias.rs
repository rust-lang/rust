// Regression test for <https://github.com/rust-lang/rust/issues/104851>

//@ set foo = "$.index[?(@.name=='Foo')].id"
pub type Foo = i32;

//@ is "$.index[?(@.name=='demo')].inner.function.sig.output.resolved_path.id" $foo
pub fn demo() -> Foo {
    42
}
