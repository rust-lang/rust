A number in a non-decimal base has no digits.

Erroneous code example:

```compile_fail,E0768
let s: i32 = 0b; // error!
```

To fix this error, add the missing digits:

```
let s: i32 = 0b1; // ok!
```
