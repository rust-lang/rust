The lifetime parameters of the method do not match the trait declaration.

Erroneous code example:

```compile_fail,E0195
trait Trait {
    fn bar<'a,'b:'a>(x: &'a str, y: &'b str);
}

struct Foo;

impl Trait for Foo {
    fn bar<'a,'b>(x: &'a str, y: &'b str) {
    // error: lifetime parameters or bounds on method `bar`
    // do not match the trait declaration
    }
}
```

The lifetime constraint `'b` for `bar()` implementation does not match the
trait declaration. Ensure lifetime declarations match exactly in both trait
declaration and implementation. Example:

```
trait Trait {
    fn t<'a,'b:'a>(x: &'a str, y: &'b str);
}

struct Foo;

impl Trait for Foo {
    fn t<'a,'b:'a>(x: &'a str, y: &'b str) { // ok!
    }
}
```
