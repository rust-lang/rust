Expected type did not match the received type.

Erroneous code examples:

```compile_fail,E0308
fn plus_one(x: i32) -> i32 {
    x + 1
}

plus_one("Not a number");
//       ^^^^^^^^^^^^^^ expected `i32`, found `&str`

if "Not a bool" {
// ^^^^^^^^^^^^ expected `bool`, found `&str`
}

let x: f32 = "Not a float";
//     ---   ^^^^^^^^^^^^^ expected `f32`, found `&str`
//     |
//     expected due to this
```

This error occurs when an expression was used in a place where the compiler
expected an expression of a different type. It can occur in several cases, the
most common being when calling a function and passing an argument which has a
different type than the matching type in the function declaration.
