#### Note: this error code is no longer emitted by the compiler.

Too many type arguments were supplied for a function. For example:

```compile_fail,E0107
fn foo<T>() {}

fn main() {
    foo::<f64, bool>(); // error: wrong number of type arguments:
                        //        expected 1, found 2
}
```

The number of supplied arguments must exactly match the number of defined type
parameters.
