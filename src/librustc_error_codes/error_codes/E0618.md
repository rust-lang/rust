Attempted to call something which isn't a function nor a method.

Erroneous code examples:

```compile_fail,E0618
enum X {
    Entry,
}

X::Entry(); // error: expected function, tuple struct or tuple variant,
            // found `X::Entry`

// Or even simpler:
let x = 0i32;
x(); // error: expected function, tuple struct or tuple variant, found `i32`
```

Only functions and methods can be called using `()`. Example:

```
// We declare a function:
fn i_am_a_function() {}

// And we call it:
i_am_a_function();
```
