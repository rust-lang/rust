% Looping

Looping is the last basic construct that we haven't learned yet in Rust. Rust has
two main looping constructs: `for` and `while`.

## `for`

The `for` loop is used to loop a particular number of times. Rust's `for` loops
work a bit differently than in other systems languages, however. Rust's `for`
loop doesn't look like this "C-style" `for` loop:

```{c}
for (x = 0; x < 10; x++) {
    printf( "%d\n", x );
}
```

Instead, it looks like this:

```{rust}
for x in range(0, 10) {
    println!("{}", x); // x: i32
}
```

In slightly more abstract terms,

```{ignore}
for var in expression {
    code
}
```

The expression is an iterator, which we will discuss in more depth later in the
guide. The iterator gives back a series of elements. Each element is one
iteration of the loop. That value is then bound to the name `var`, which is
valid for the loop body. Once the body is over, the next value is fetched from
the iterator, and we loop another time. When there are no more values, the
`for` loop is over.

In our example, `range` is a function that takes a start and an end position,
and gives an iterator over those values. The upper bound is exclusive, though,
so our loop will print `0` through `9`, not `10`.

Rust does not have the "C-style" `for` loop on purpose. Manually controlling
each element of the loop is complicated and error prone, even for experienced C
developers.

We'll talk more about `for` when we cover **iterator**s, later in the Guide.

## `while`

The other kind of looping construct in Rust is the `while` loop. It looks like
this:

```{rust}
let mut x = 5u;       // mut x: uint
let mut done = false; // mut done: bool

while !done {
    x += x - 3;
    println!("{}", x);
    if x % 5 == 0 { done = true; }
}
```

`while` loops are the correct choice when you're not sure how many times
you need to loop.

If you need an infinite loop, you may be tempted to write this:

```{rust,ignore}
while true {
```

However, Rust has a dedicated keyword, `loop`, to handle this case:

```{rust,ignore}
loop {
```

Rust's control-flow analysis treats this construct differently than a
`while true`, since we know that it will always loop. The details of what
that _means_ aren't super important to understand at this stage, but in
general, the more information we can give to the compiler, the better it
can do with safety and code generation, so you should always prefer
`loop` when you plan to loop infinitely.

## Ending iteration early

Let's take a look at that `while` loop we had earlier:

```{rust}
let mut x = 5u;
let mut done = false;

while !done {
    x += x - 3;
    println!("{}", x);
    if x % 5 == 0 { done = true; }
}
```

We had to keep a dedicated `mut` boolean variable binding, `done`, to know
when we should exit out of the loop. Rust has two keywords to help us with
modifying iteration: `break` and `continue`.

In this case, we can write the loop in a better way with `break`:

```{rust}
let mut x = 5u;

loop {
    x += x - 3;
    println!("{}", x);
    if x % 5 == 0 { break; }
}
```

We now loop forever with `loop` and use `break` to break out early.

`continue` is similar, but instead of ending the loop, goes to the next
iteration. This will only print the odd numbers:

```{rust}
for x in range(0, 10) {
    if x % 2 == 0 { continue; }

    println!("{}", x);
}
```

Both `continue` and `break` are valid in both kinds of loops.
