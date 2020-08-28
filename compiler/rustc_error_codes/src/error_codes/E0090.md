#### Note: this error code is no longer emitted by the compiler.

You gave too few lifetime arguments. Example:

```compile_fail,E0107
fn foo<'a: 'b, 'b: 'a>() {}

fn main() {
    foo::<'static>(); // error: wrong number of lifetime arguments:
                      //        expected 2, found 1
}
```

Please check you give the right number of lifetime arguments. Example:

```
fn foo<'a: 'b, 'b: 'a>() {}

fn main() {
    foo::<'static, 'static>();
}
```
