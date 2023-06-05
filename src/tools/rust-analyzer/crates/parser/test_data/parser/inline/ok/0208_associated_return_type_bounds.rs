fn foo<T: Foo<foo(): Send, bar(i32): Send, baz(i32, i32): Send>>() {}
