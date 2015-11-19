% if

Rust’s take on `if` is not particularly complex, but it’s much more like the
`if` you’ll find in a dynamically typed language than in a more traditional
systems language. So let’s talk about it, to make sure you grasp the nuances.

`if` is a specific form of a more general concept, the ‘branch’. The name comes
from a branch in a tree: a decision point, where depending on a choice,
multiple paths can be taken.

In the case of `if`, there is one choice that leads down two paths:

```rust
let x = 5;

if x == 5 {
    println!("x is five!");
}
```

If we changed the value of `x` to something else, this line would not print.
More specifically, if the expression after the `if` evaluates to `true`, then
the block is executed. If it’s `false`, then it is not.

If you want something to happen in the `false` case, use an `else`:

```rust
let x = 5;

if x == 5 {
    println!("x is five!");
} else {
    println!("x is not five :(");
}
```

If there is more than one case, use an `else if`:

```rust
let x = 5;

if x == 5 {
    println!("x is five!");
} else if x == 6 {
    println!("x is six!");
} else {
    println!("x is not five or six :(");
}
```

This is all pretty standard. However, you can also do this:

```rust
let x = 5;

let y = if x == 5 {
    10
} else {
    15
}; // y: i32
```

Which we can (and probably should) write like this:

```rust
let x = 5;

let y = if x == 5 { 10 } else { 15 }; // y: i32
```

This works because `if` is an expression. The value of the expression is the
value of the last expression in whichever branch was chosen. An `if` without an
`else` always results in `()` as the value.
