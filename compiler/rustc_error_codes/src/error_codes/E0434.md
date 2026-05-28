A variable used inside an inner function comes from a dynamic environment.

Erroneous code example:

```compile_fail,E0434
fn foo() {
    let y = 5;
    fn bar() -> u32 {
        y // error: can't capture dynamic environment in a fn item; use the
          //        || { ... } closure form instead.
    }
}
```

Inner functions do not have access to their containing environment. To fix this
error, you can replace the function with a closure:

```
fn foo() {
    let y = 5;
    let bar = || {
        y
    };
}
```

Or replace the captured variable with a constant or a static item:

```
fn foo() {
    static mut X: u32 = 4;
    const Y: u32 = 5;
    fn bar() -> u32 {
        unsafe {
            X = 3;
        }
        Y
    }
}
```
