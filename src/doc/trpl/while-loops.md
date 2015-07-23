% while Loops

Rust also has a `while` loop. It looks like this:

```rust
let mut x = 5; // mut x: i32
let mut done = false; // mut done: bool

while !done {
    x += x - 3;

    println!("{}", x);

    if x % 5 == 0 {
        done = true;
    }
}
```

`while` loops are the correct choice when you’re not sure how many times
you need to loop.

If you need an infinite loop, you may be tempted to write this:

```rust,ignore
while true {
```

However, Rust has a dedicated keyword, `loop`, to handle this case:

```rust,ignore
loop {
```

Rust’s control-flow analysis treats this construct differently than a `while
true`, since we know that it will always loop. In general, the more information
we can give to the compiler, the better it can do with safety and code
generation, so you should always prefer `loop` when you plan to loop
infinitely.

## Ending iteration early

Let’s take a look at that `while` loop we had earlier:

```rust
let mut x = 5;
let mut done = false;

while !done {
    x += x - 3;

    println!("{}", x);

    if x % 5 == 0 {
        done = true;
    }
}
```

We had to keep a dedicated `mut` boolean variable binding, `done`, to know
when we should exit out of the loop. Rust has two keywords to help us with
modifying iteration: `break` and `continue`.

In this case, we can write the loop in a better way with `break`:

```rust
let mut x = 5;

loop {
    x += x - 3;

    println!("{}", x);

    if x % 5 == 0 { break; }
}
```

We now loop forever with `loop` and use `break` to break out early.

`continue` is similar, but instead of ending the loop, goes to the next
iteration. This will only print the odd numbers:

```rust
for x in 0..10 {
    if x % 2 == 0 { continue; }

    println!("{}", x);
}
```

You may also encounter situations where you have nested loops and need to 
specify which one your `break` or `continue` statement is for. Like most 
other languages, by default a `break` or `continue` will apply to innermost 
loop. In a sitation where you would like to a `break` or `continue` for one 
of the outer loops, you can use labels to specify which loop the `break` or
 `continue` statement applies to. This will only print when both `x` and `y` are
 odd:

```rust
'outer: for x in 0..10 {
    'inner: for y in 0..10 {
        if x % 2 == 0 { continue 'outer; } // continues the loop over x
        if y % 2 == 0 { continue 'inner; } // continues the loop over y
        println!("x: {}, y: {}", x, y);
    }
}
```

Both `continue` and `break` are valid in both `while` loops and [`for` loops][for].

[for]: for-loops.html
