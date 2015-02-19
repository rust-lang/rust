% Patterns

We've made use of patterns a few times in the guide: first with `let` bindings,
then with `match` statements. Let's go on a whirlwind tour of all of the things
patterns can do!

A quick refresher: you can match against literals directly, and `_` acts as an
*any* case:

```{rust}
let x = 1;

match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    _ => println!("anything"),
}
```

You can match multiple patterns with `|`:

```{rust}
let x = 1;

match x {
    1 | 2 => println!("one or two"),
    3 => println!("three"),
    _ => println!("anything"),
}
```

You can match a range of values with `...`:

```{rust}
let x = 1;

match x {
    1 ... 5 => println!("one through five"),
    _ => println!("anything"),
}
```

Ranges are mostly used with integers and single characters.

If you're matching multiple things, via a `|` or a `...`, you can bind
the value to a name with `@`:

```{rust}
let x = 1;

match x {
    e @ 1 ... 5 => println!("got a range element {}", e),
    _ => println!("anything"),
}
```

If you're matching on an enum which has variants, you can use `..` to
ignore the value and type in the variant:

```{rust}
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

You can introduce *match guards* with `if`:

```{rust}
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

If you're matching on a pointer, you can use the same syntax as you declared it
with. First, `&`:

```{rust}
let x = &5;

match x {
    &val => println!("Got a value: {}", val),
}
```

Here, the `val` inside the `match` has type `i32`. In other words, the left-hand
side of the pattern destructures the value. If we have `&5`, then in `&val`, `val`
would be `5`.

If you want to get a reference, use the `ref` keyword:

```{rust}
let x = 5;

match x {
    ref r => println!("Got a reference to {}", r),
}
```

Here, the `r` inside the `match` has the type `&i32`. In other words, the `ref`
keyword _creates_ a reference, for use in the pattern. If you need a mutable
reference, `ref mut` will work in the same way:

```{rust}
let mut x = 5;

match x {
    ref mut mr => println!("Got a mutable reference to {}", mr),
}
```

If you have a struct, you can destructure it inside of a pattern:

```{rust}
# #![allow(non_shorthand_field_patterns)]
struct Point {
    x: i32,
    y: i32,
}

let origin = Point { x: 0, y: 0 };

match origin {
    Point { x: x, y: y } => println!("({},{})", x, y),
}
```

If we only care about some of the values, we don't have to give them all names:

```{rust}
# #![allow(non_shorthand_field_patterns)]
struct Point {
    x: i32,
    y: i32,
}

let origin = Point { x: 0, y: 0 };

match origin {
    Point { x: x, .. } => println!("x is {}", x),
}
```

You can do this kind of match on any member, not just the first:

```{rust}
# #![allow(non_shorthand_field_patterns)]
struct Point {
    x: i32,
    y: i32,
}

let origin = Point { x: 0, y: 0 };

match origin {
    Point { y: y, .. } => println!("y is {}", y),
}
```

If you want to match against a slice or array, you can use `&`:

```{rust}
fn main() {
    let v = vec!["match_this", "1"];

    match &v[..] {
        ["match_this", second] => println!("The second element is {}", second),
        _ => {},
    }
}
```

Whew! That's a lot of different ways to match things, and they can all be
mixed and matched, depending on what you're doing:

```{rust,ignore}
match x {
    Foo { x: Some(ref name), y: None } => ...
}
```

Patterns are very powerful.  Make good use of them.
