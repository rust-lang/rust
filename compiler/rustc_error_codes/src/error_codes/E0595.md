#### Note: this error code is no longer emitted by the compiler.

Closures cannot mutate immutable captured variables.

Erroneous code example:

```compile_fail,E0594
let x = 3; // error: closure cannot assign to immutable local variable `x`
let mut c = || { x += 1 };
```

Make the variable binding mutable:

```
let mut x = 3; // ok!
let mut c = || { x += 1 };
```
