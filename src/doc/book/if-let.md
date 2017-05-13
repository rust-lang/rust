% if let

`if let` allows you to combine `if` and `let` together to reduce the overhead
of certain kinds of pattern matches.

For example, let’s say we have some sort of `Option<T>`. We want to call a function
on it if it’s `Some<T>`, but do nothing if it’s `None`. That looks like this:

```rust
# let option = Some(5);
# fn foo(x: i32) { }
match option {
    Some(x) => { foo(x) },
    None => {},
}
```

We don’t have to use `match` here, for example, we could use `if`:

```rust
# let option = Some(5);
# fn foo(x: i32) { }
if option.is_some() {
    let x = option.unwrap();
    foo(x);
}
```

Neither of these options is particularly appealing. We can use `if let` to
do the same thing in a nicer way:

```rust
# let option = Some(5);
# fn foo(x: i32) { }
if let Some(x) = option {
    foo(x);
}
```

If a [pattern][patterns] matches successfully, it binds any appropriate parts of
the value to the identifiers in the pattern, then evaluates the expression. If
the pattern doesn’t match, nothing happens.

If you want to do something else when the pattern does not match, you can
use `else`:

```rust
# let option = Some(5);
# fn foo(x: i32) { }
# fn bar() { }
if let Some(x) = option {
    foo(x);
} else {
    bar();
}
```

## `while let`

In a similar fashion, `while let` can be used when you want to conditionally
loop  as long as a value matches a certain pattern. It turns code like this:

```rust
let mut v = vec![1, 3, 5, 7, 11];
loop {
    match v.pop() {
        Some(x) =>  println!("{}", x),
        None => break,
    }
}
```

Into code like this:

```rust
let mut v = vec![1, 3, 5, 7, 11];
while let Some(x) = v.pop() {
    println!("{}", x);
}
```

[patterns]: patterns.html
