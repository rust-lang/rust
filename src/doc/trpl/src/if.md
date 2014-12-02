% If

Rust's take on `if` is not particularly complex, but it's much more like the
`if` you'll find in a dynamically typed language than in a more traditional
systems language. So let's talk about it, to make sure you grasp the nuances.

`if` is a specific form of a more general concept, the 'branch.' The name comes
from a branch in a tree: a decision point, where depending on a choice,
multiple paths can be taken.

In the case of `if`, there is one choice that leads down two paths:

```rust
let x = 5i;

if x == 5i {
    println!("x is five!");
}
```

If we changed the value of `x` to something else, this line would not print.
More specifically, if the expression after the `if` evaluates to `true`, then
the block is executed. If it's `false`, then it is not.

If you want something to happen in the `false` case, use an `else`:

```{rust}
let x = 5i;

if x == 5i {
    println!("x is five!");
} else {
    println!("x is not five :(");
}
```

This is all pretty standard. However, you can also do this:


```{rust}
let x = 5i;

let y = if x == 5i {
    10i
} else {
    15i
};
```

Which we can (and probably should) write like this:

```{rust}
let x = 5i;

let y = if x == 5i { 10i } else { 15i };
```

This reveals two interesting things about Rust: it is an expression-based
language, and semicolons are different than in other 'curly brace and
semicolon'-based languages. These two things are related.

