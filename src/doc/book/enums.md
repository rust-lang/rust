% Enums

An `enum` in Rust is a type that represents data that could be one of
several possible variants:

```rust
enum Message {
    Quit,
    ChangeColor(i32, i32, i32),
    Move { x: i32, y: i32 },
    Write(String),
}
```

Each variant can optionally have data associated with it. The syntax for
defining variants resembles the syntaxes used to define structs: you can
have variants with no data (like unit-like structs), variants with named
data, and variants with unnamed data (like tuple structs). Unlike
separate struct definitions, however, an `enum` is a single type. A
value of the enum can match any of the variants. For this reason, an
enum is sometimes called a ‘sum type’: the set of possible values of the
enum is the sum of the sets of possible values for each variant.

We use the `::` syntax to use the name of each variant: they’re scoped by the name
of the `enum` itself. This allows both of these to work:

```rust
# enum Message {
#     Move { x: i32, y: i32 },
# }
let x: Message = Message::Move { x: 3, y: 4 };

enum BoardGameTurn {
    Move { squares: i32 },
    Pass,
}

let y: BoardGameTurn = BoardGameTurn::Move { squares: 1 };
```

Both variants are named `Move`, but since they’re scoped to the name of
the enum, they can both be used without conflict.

A value of an enum type contains information about which variant it is,
in addition to any data associated with that variant. This is sometimes
referred to as a ‘tagged union’, since the data includes a ‘tag’
indicating what type it is. The compiler uses this information to
enforce that you’re accessing the data in the enum safely. For instance,
you can’t simply try to destructure a value as if it were one of the
possible variants:

```rust,ignore
fn process_color_change(msg: Message) {
    let Message::ChangeColor(r, g, b) = msg; // compile-time error
}
```

Not supporting these operations may seem rather limiting, but it’s a limitation
which we can overcome. There are two ways: by implementing equality ourselves,
or by pattern matching variants with [`match`][match] expressions, which you’ll
learn in the next section. We don’t know enough about Rust to implement
equality yet, but we’ll find out in the [`traits`][traits] section.

[match]: match.html
[if-let]: if-let.html
[traits]: traits.html

# Constructors as functions

An enum’s constructors can also be used like functions. For example:

```rust
# enum Message {
# Write(String),
# }
let m = Message::Write("Hello, world".to_string());
```

Is the same as

```rust
# enum Message {
# Write(String),
# }
fn foo(x: String) -> Message {
    Message::Write(x)
}

let x = foo("Hello, world".to_string());
```

This is not immediately useful to us, but when we get to
[`closures`][closures], we’ll talk about passing functions as arguments to
other functions. For example, with [`iterators`][iterators], we can do this
to convert a vector of `String`s into a vector of `Message::Write`s:

```rust
# enum Message {
# Write(String),
# }

let v = vec!["Hello".to_string(), "World".to_string()];

let v1: Vec<Message> = v.into_iter().map(Message::Write).collect();
```

[closures]: closures.html
[iterators]: iterators.html
