# `loop_break_value`

The tracking issue for this feature is: [#37339]

[#37339]: https://github.com/rust-lang/rust/issues/37339

Documentation to be appended to section G of the book.

------------------------

### Loops as expressions

Like most things in Rust, loops are expressions, and have a value; normally `()` unless the loop
never exits.
A `loop` can instead evaluate to a useful value via *break with value*:

```rust
#![feature(loop_break_value)]

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
#![feature(loop_break_value)]

// Declare that value returned is unsigned 64-bit:
let n: u64 = loop {
    break 1;
};
```

It is an error if types do not agree, either between a "break" value and an external requirement,
or between multiple "break" values:

```no_compile
#![feature(loop_break_value)]

loop {
    if true {
        break 1u32;
    } else {
        break 0u8;  // error: types do not agree
    }
};

let n: i32 = loop {
    break 0u32; // error: type does not agree with external requirement
};
```

#### Break: label, value

Four forms of `break` are available, where EXPR is some expression which evaluates to a value:

1.  `break;`
2.  `break 'label;`
3.  `break EXPR;`
4.  `break 'label EXPR;`

When no value is given, the value `()` is assumed, thus `break;` is equivalent to `break ();`.

Using a label allows returning a value from an inner loop:

```rust
#![feature(loop_break_value)]

let result = 'outer: loop {
    for n in 1..10 {
        if n > 4 {
            break 'outer n;
        }
    }
};
```
