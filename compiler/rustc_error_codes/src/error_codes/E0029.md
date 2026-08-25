Something other than numbers and characters has been used for a range.

Erroneous code example:

```compile_fail,E0029
let string = "salutations !";

// The ordering relation for strings cannot be evaluated at compile time,
// so this doesn't work:
match string {
    "hello" ..= "world" => {}
    _ => {}
}

// This is a more general version, using a guard:
match string {
    s if s >= "hello" && s <= "world" => {}
    _ => {}
}
```

In a match expression, only numbers and characters can be matched against a
range. This is because the compiler checks that the range is non-empty at
compile-time, and is unable to evaluate arbitrary comparison functions. If you
want to capture values of an orderable type between two end-points, you can use
a guard.
