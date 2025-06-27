// Regression test for <https://github.com/rust-lang/rust/issues/104851>

//@ arg foo .index[] | select(.name == "Foo").id
pub type Foo = i32;

//@ jq .index[] | select(.name == "demo").inner.function.sig?.output.resolved_path?.id == $foo
pub fn demo() -> Foo {
    42
}
