// Regression test for <https://github.com/rust-lang/rust/issues/104851>

/// @set foo = "$.index[*][?(@.name=='Foo')].id"
pub type Foo = i32;

// @is "$.index[*][?(@.name=='demo')].inner.decl.output.kind" '"resolved_path"'
// @is "$.index[*][?(@.name=='demo')].inner.decl.output.inner.id" $foo
pub fn demo() -> Foo {
    42
}
