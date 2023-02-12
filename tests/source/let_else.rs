fn main() {
    let Some(x) = opt else { return };

    let Some(x) = opt else { return; };

    let Some(x) = opt else {
        // nope
        return;
    };

    let Some(x) = y.foo("abc", fairly_long_identifier, "def", "123456", "string", "cheese") else { bar() };

    let Some(x) = abcdef().foo("abc", some_really_really_really_long_ident, "ident", "123456").bar().baz().qux("fffffffffffffffff") else { foo_bar() };
}
