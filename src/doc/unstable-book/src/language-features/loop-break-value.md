# `loop_break_value`

The tracking issue for this feature is: [#37339]

[#37339]: https://github.com/rust-lang/rust/issues/37339

Documentation to be appended to section 3.6 of the book: Loops (after "Loop Labels", or before if
the "Break" section is moved). If this is deemed too complex a feature this early in the book, it
could also be moved to a new section (please advise). This would allow examples breaking with
non-primitive types, references, and discussion of coercion (probably unnecessary however).

------------------------

### Loops as expressions

Like everything else in Rust, loops are expressions; for example, the following is perfectly legal,
if rather useless:

```rust
let result = for n in 1..4 {
    println!("Hello, {}", n);
};
assert_eq!(result, ());
```

Until now, all the loops you have seen evaluate to either `()` or `!`, the latter being special
syntax for "no value", meaning the loop never exits. A `loop` can instead evaluate to
a useful value via *break with value*:

```rust
// Find the first square number over 1000:
let mut n = 1;
let square = loop {
    if n * n > 1000 {
        break n * n;
    }
    n += 1;
};
```

The evaluation type may be specified externally:

```rust
// Declare that value returned is unsigned 64-bit:
let n: u64 = loop {
    break 1;
};
```

It is an error if types do not agree, either between a "break" value and an external requirement,
or between multiple "break" values:

```rust
loop {
    if random_bool() {
        break 1u32;
    } else {
        break 0u8;  // error: types do not agree
    }
};

let n: i32 = loop {
    break 0u32; // error: type does not agree with external requirement
};
```

For now, breaking with a value is only possible with `loop`; the same functionality may
some day be added to `for` and `while` (this would require some new syntax like
`while f() { break 1; } default { break 0; }`).

#### Break: label, value

Four forms of `break` are available, where EXPR is some expression which evaluates to a value:

1.  `break;`
2.  `break 'label;`
3.  `break EXPR;`
4.  `break 'label EXPR;`

When no value is given, the value `()` is assumed, thus `break;` is equivalent to `break ();`.

Using a label allows returning a value from an inner loop:

```rust
let result = 'outer: loop {
    for n in 1..10 {
        if n > 4 {
            break 'outer n;
        }
    }
};
```
