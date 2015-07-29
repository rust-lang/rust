% infinite loops

The infinite `loop` is the simplest form of `loop` available in Rust. Using the keyword `loop`, Rust provides a way to loop until a `break` or `return` is issued. Rust's infinite `loop`s look like this:

```rust
loop {
    println!("Loop forever!");
}
```

Leaving a infinite `loop` can be achieved using a break statement as follows:

```rust
let mut i = 0;
loop {
    if i == 10 {
        break;
    }
    println!("Loop number {}", i);
    i = i + 1;
}
```

## Loop labels

Labels can be assigned to `loop`s to so that, in the case of nested `loop`s, an outer `loop` may be left early when certain criteria are met in an inner `loop`.

```rust
let mut i = 0;
'outer: loop {
    'inner: loop {
        if i == 10 {
            break 'outer;
        }
        i = i + 1;
    }
}
```

In the above example, the inner `loop` is able to cause the outer `loop` to stop.
