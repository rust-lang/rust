This error occurs when you defined methods or associated functions with same
name.

Erroneous code example:

```compile_fail,E0592
struct Foo;

impl Foo {
    fn bar() {} // previous definition here
}

impl Foo {
    fn bar() {} // duplicate definition here
}
```

A similar error is E0201. The difference is whether there is one declaration
block or not. To avoid this error, you must give each `fn` a unique name.

```
struct Foo;

impl Foo {
    fn bar() {}
}

impl Foo {
    fn baz() {} // define with different name
}
```
