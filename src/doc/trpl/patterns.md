% Patterns

Patterns are quite common in Rust. We use them in [variable
bindings][bindings], [match statements][match], and other places, too. Let’s go
on a whirlwind tour of all of the things patterns can do!

[bindings]: variable-bindings.html
[match]: match.html

A quick refresher: you can match against literals directly, and `_` acts as an
‘any’ case:

```rust
let x = 1;

match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    _ => println!("anything"),
}
```

This prints `one`.

# Multiple patterns

You can match multiple patterns with `|`:

```rust
let x = 1;

match x {
    1 | 2 => println!("one or two"),
    3 => println!("three"),
    _ => println!("anything"),
}
```

This prints `one or two`.

# Ranges

You can match a range of values with `...`:

```rust
let x = 1;

match x {
    1 ... 5 => println!("one through five"),
    _ => println!("anything"),
}
```

This prints `one through five`.

Ranges are mostly used with integers and `char`s:

```rust
let x = '💅';

match x {
    'a' ... 'j' => println!("early letter"),
    'k' ... 'z' => println!("late letter"),
    _ => println!("something else"),
}
```

This prints `something else`

# Bindings

You can bind values to names with `@`:

```rust
let x = 1;

match x {
    e @ 1 ... 5 => println!("got a range element {}", e),
    _ => println!("anything"),
}
```

This prints `got a range element 1`. This is useful when you want to
do a complicated match of part of a data structure:

```rust
#[derive(Debug)]
struct Person {
    name: Option<String>,
}

let name = "Steve".to_string();
let mut x: Option<Person> = Some(Person { name: Some(name) });
match x {
    Some(Person { name: ref a @ Some(_), .. }) => println!("{:?}", a),
    _ => {}
}
```

This prints `Some("Steve")`: We’ve bound the inner `name` to `a`.

If you use `@` with `|`, you need to make sure the name is bound in each part
of the pattern:

```rust
let x = 5;

match x {
    e @ 1 ... 5 | e @ 8 ... 10 => println!("got a range element {}", e),
    _ => println!("anything"),
}
```

# Ignoring variants

If you’re matching on an enum which has variants, you can use `..` to
ignore the value and type in the variant:

```rust
enum OptionalInt {
    Value(i32),
    Missing,
}

let x = OptionalInt::Value(5);

match x {
    OptionalInt::Value(..) => println!("Got an int!"),
    OptionalInt::Missing => println!("No such luck."),
}
```

This prints `Got an int!`.

# Guards

You can introduce ‘match guards’ with `if`:

```rust
enum OptionalInt {
    Value(i32),
    Missing,
}

let x = OptionalInt::Value(5);

match x {
    OptionalInt::Value(i) if i > 5 => println!("Got an int bigger than five!"),
    OptionalInt::Value(..) => println!("Got an int!"),
    OptionalInt::Missing => println!("No such luck."),
}
```

This prints `Got an int!`

# ref and ref mut

If you want to get a [reference][ref], use the `ref` keyword:

```rust
let x = 5;

match x {
    ref r => println!("Got a reference to {}", r),
}
```

This prints `Got a reference to 5`.

[ref]: references-and-borrowing.html

Here, the `r` inside the `match` has the type `&i32`. In other words, the `ref`
keyword _creates_ a reference, for use in the pattern. If you need a mutable
reference, `ref mut` will work in the same way:

```rust
let mut x = 5;

match x {
    ref mut mr => println!("Got a mutable reference to {}", mr),
}
```

# Destructuring

If you have a compound data type, like a [`struct`][struct], you can destructure it
inside of a pattern:

```rust
struct Point {
    x: i32,
    y: i32,
}

let origin = Point { x: 0, y: 0 };

match origin {
    Point { x: x, y: y } => println!("({},{})", x, y),
}
```

[struct]: structs.html

If we only care about some of the values, we don’t have to give them all names:

```rust
struct Point {
    x: i32,
    y: i32,
}

let origin = Point { x: 0, y: 0 };

match origin {
    Point { x: x, .. } => println!("x is {}", x),
}
```

This prints `x is 0`.

You can do this kind of match on any member, not just the first:

```rust
struct Point {
    x: i32,
    y: i32,
}

let origin = Point { x: 0, y: 0 };

match origin {
    Point { y: y, .. } => println!("y is {}", y),
}
```

This prints `y is 0`.

This ‘destructuring’ behavior works on any compound data type, like
[tuples][tuples] or [enums][enums].

[tuples]: primitive-types.html#tuples
[enums]: enums.html

# Mix and Match

Whew! That’s a lot of different ways to match things, and they can all be
mixed and matched, depending on what you’re doing:

```rust,ignore
match x {
    Foo { x: Some(ref name), y: None } => ...
}
```

Patterns are very powerful. Make good use of them.
