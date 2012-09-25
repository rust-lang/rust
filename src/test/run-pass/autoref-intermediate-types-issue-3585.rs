trait Foo {
    fn foo(&self) -> ~str;
}

impl<T: Foo> @T: Foo {
    fn foo(&self) -> ~str {
        fmt!("@%s", (**self).foo())
    }
}

impl uint: Foo {
    fn foo(&self) -> ~str {
        fmt!("%u", *self)
    }
}

fn main() {
    let x = @3u;
    assert x.foo() == ~"@3";
}