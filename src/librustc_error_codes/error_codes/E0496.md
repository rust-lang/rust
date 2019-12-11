A lifetime name is shadowing another lifetime name.

Erroneous code example:

```compile_fail,E0496
struct Foo<'a> {
    a: &'a i32,
}

impl<'a> Foo<'a> {
    fn f<'a>(x: &'a i32) { // error: lifetime name `'a` shadows a lifetime
                           //        name that is already in scope
    }
}
```

Please change the name of one of the lifetimes to remove this error. Example:

```
struct Foo<'a> {
    a: &'a i32,
}

impl<'a> Foo<'a> {
    fn f<'b>(x: &'b i32) { // ok!
    }
}

fn main() {
}
```
