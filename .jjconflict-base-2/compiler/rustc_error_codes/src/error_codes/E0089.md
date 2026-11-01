#### Note: this error code is no longer emitted by the compiler.

Too few type arguments were supplied for a function. For example:

```compile_fail,E0107
fn foo<T, U>() {}

fn main() {
    foo::<f64>(); // error: wrong number of type arguments: expected 2, found 1
}
```

Note that if a function takes multiple type arguments but you want the compiler
to infer some of them, you can use type placeholders:

```compile_fail,E0107
fn foo<T, U>(x: T) {}

fn main() {
    let x: bool = true;
    foo::<f64>(x);    // error: wrong number of type arguments:
                      //        expected 2, found 1
    foo::<_, f64>(x); // same as `foo::<bool, f64>(x)`
}
```
