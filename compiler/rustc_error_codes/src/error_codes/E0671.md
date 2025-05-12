#### Note: this error code is no longer emitted by the compiler.

Const parameters cannot depend on type parameters.
The following is therefore invalid:

```compile_fail,E0770
fn const_id<T, const N: T>() -> T { // error
    N
}
```
