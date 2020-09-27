An `if` expression is missing an `else` block.

Erroneous code example:

```compile_fail,E0317
let x = 5;
let a = if x == 5 {
    1
};
```

This error occurs when an `if` expression without an `else` block is used in a
context where a type other than `()` is expected. In the previous code example,
the `let` expression was expecting a value but since there was no `else`, no
value was returned.

An `if` expression without an `else` block has the type `()`, so this is a type
error. To resolve it, add an `else` block having the same type as the `if`
block.

So to fix the previous code example:

```
let x = 5;
let a = if x == 5 {
    1
} else {
    2
};
```
