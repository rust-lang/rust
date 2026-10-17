It is not allowed to cast to a bool.

Erroneous code example:

```compile_fail,E0054
let x = 5;

// Not allowed, won't compile
let x_is_nonzero = x as bool;
```

If you are trying to cast a numeric type to a bool, you can compare it with
zero instead:

```
let x = 5;

// Ok
let x_is_nonzero = x != 0;
```
