An undeclared lifetime was used.

Erroneous code example:

```compile_fail,E0261
// error, use of undeclared lifetime name `'a`
fn foo(x: &'a str) { }

struct Foo {
    // error, use of undeclared lifetime name `'a`
    x: &'a str,
}
```

These can be fixed by declaring lifetime parameters:

```
struct Foo<'a> {
    x: &'a str,
}

fn foo<'a>(x: &'a str) {}
```

Impl blocks declare lifetime parameters separately. You need to add lifetime
parameters to an impl block if you're implementing a type that has a lifetime
parameter of its own.
For example:

```compile_fail,E0261
struct Foo<'a> {
    x: &'a str,
}

// error,  use of undeclared lifetime name `'a`
impl Foo<'a> {
    fn foo<'a>(x: &'a str) {}
}
```

This is fixed by declaring the impl block like this:

```
struct Foo<'a> {
    x: &'a str,
}

// correct
impl<'a> Foo<'a> {
    fn foo(x: &'a str) {}
}
```
