The maximum value of an enum was reached, so it cannot be automatically
set in the next enum value.

Erroneous code example:

```compile_fail,E0370
#[repr(i64)]
enum Foo {
    X = 0x7fffffffffffffff,
    Y, // error: enum discriminant overflowed on value after
       //        9223372036854775807: i64; set explicitly via
       //        Y = -9223372036854775808 if that is desired outcome
}
```

To fix this, please set manually the next enum value or put the enum variant
with the maximum value at the end of the enum. Examples:

```
#[repr(i64)]
enum Foo {
    X = 0x7fffffffffffffff,
    Y = 0, // ok!
}
```

Or:

```
#[repr(i64)]
enum Foo {
    Y = 0, // ok!
    X = 0x7fffffffffffffff,
}
```
