% Structs

`struct`s are a way of creating more complex data types. For example, if we were
doing calculations involving coordinates in 2D space, we would need both an `x`
and a `y` value:

```rust
let origin_x = 0;
let origin_y = 0;
```

A `struct` lets us combine these two into a single, unified datatype with `x`
and `y` as field labels:

```rust
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let origin = Point { x: 0, y: 0 }; // origin: Point

    println!("The origin is at ({}, {})", origin.x, origin.y);
}
```

There’s a lot going on here, so let’s break it down. We declare a `struct` with
the `struct` keyword, and then with a name. By convention, `struct`s begin with
a capital letter and are camel cased: `PointInSpace`, not `Point_In_Space`.

We can create an instance of our `struct` via `let`, as usual, but we use a `key:
value` style syntax to set each field. The order doesn’t need to be the same as
in the original declaration.

Finally, because fields have names, we can access them through dot
notation: `origin.x`.

The values in `struct`s are immutable by default, like other bindings in Rust.
Use `mut` to make them mutable:

```rust
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut point = Point { x: 0, y: 0 };

    point.x = 5;

    println!("The point is at ({}, {})", point.x, point.y);
}
```

This will print `The point is at (5, 0)`.

Rust does not support field mutability at the language level, so you cannot
write something like this:

```rust,ignore
struct Point {
    mut x: i32, // This causes an error.
    y: i32,
}
```

Mutability is a property of the binding, not of the structure itself. If you’re
used to field-level mutability, this may seem strange at first, but it
significantly simplifies things. It even lets you make things mutable on a temporary
basis:

```rust,ignore
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut point = Point { x: 0, y: 0 };

    point.x = 5;

    let point = point; // `point` is now immutable.

    point.y = 6; // This causes an error.
}
```

Your structure can still contain `&mut` pointers, which will let
you do some kinds of mutation:

```rust
struct Point {
    x: i32,
    y: i32,
}

struct PointRef<'a> {
    x: &'a mut i32,
    y: &'a mut i32,
}

fn main() {
    let mut point = Point { x: 0, y: 0 };

    {
        let r = PointRef { x: &mut point.x, y: &mut point.y };

        *r.x = 5;
        *r.y = 6;
    }

    assert_eq!(5, point.x);
    assert_eq!(6, point.y);
}
```

# Update syntax

A `struct` can include `..` to indicate that you want to use a copy of some
other `struct` for some of the values. For example:

```rust
struct Point3d {
    x: i32,
    y: i32,
    z: i32,
}

let mut point = Point3d { x: 0, y: 0, z: 0 };
point = Point3d { y: 1, .. point };
```

This gives `point` a new `y`, but keeps the old `x` and `z` values. It doesn’t
have to be the same `struct` either, you can use this syntax when making new
ones, and it will copy the values you don’t specify:

```rust
# struct Point3d {
#     x: i32,
#     y: i32,
#     z: i32,
# }
let origin = Point3d { x: 0, y: 0, z: 0 };
let point = Point3d { z: 1, x: 2, .. origin };
```

# Tuple structs

Rust has another data type that’s like a hybrid between a [tuple][tuple] and a
`struct`, called a ‘tuple struct’. Tuple structs have a name, but their fields
don't. They are declared with the `struct` keyword, and then with a name
followed by a tuple:

[tuple]: primitive-types.html#tuples

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

let black = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

Here, `black` and `origin` are not the same type, even though they contain the
same values.

The members of a tuple struct may be accessed by dot notation or destructuring
`let`, just like regular tuples:

```rust
# struct Color(i32, i32, i32);
# struct Point(i32, i32, i32);
# let black = Color(0, 0, 0);
# let origin = Point(0, 0, 0);
let black_r = black.0;
let Point(_, origin_y, origin_z) = origin;
```

Patterns like `Point(_, origin_y, origin_z)` are also used in
[match expressions][match].

One case when a tuple struct is very useful is when it has only one element.
We call this the ‘newtype’ pattern, because it allows you to create a new type
that is distinct from its contained value and also expresses its own semantic
meaning:

```rust
struct Inches(i32);

let length = Inches(10);

let Inches(integer_length) = length;
println!("length is {} inches", integer_length);
```

As above, you can extract the inner integer type through a destructuring `let`.
In this case, the `let Inches(integer_length)` assigns `10` to `integer_length`.
We could have used dot notation to do the same thing:

```rust
# struct Inches(i32);
# let length = Inches(10);
let integer_length = length.0;
```

It's always possible to use a `struct` instead of a tuple struct, and can be
clearer. We could write `Color` and `Point` like this instead:

```rust
struct Color {
    red: i32,
    blue: i32,
    green: i32,
}

struct Point {
    x: i32,
    y: i32,
    z: i32,
}
```

Good names are important, and while values in a tuple struct can be
referenced with dot notation as well, a `struct` gives us actual names,
rather than positions.

[match]: match.html

# Unit-like structs

You can define a `struct` with no members at all:

```rust
struct Electron {} // Use empty braces...
struct Proton;     // ...or just a semicolon.

// Whether you declared the struct with braces or not, do the same when creating one.
let x = Electron {};
let y = Proton;
```

Such a `struct` is called ‘unit-like’ because it resembles the empty
tuple, `()`, sometimes called ‘unit’. Like a tuple struct, it defines a
new type.

This is rarely useful on its own (although sometimes it can serve as a
marker type), but in combination with other features, it can become
useful. For instance, a library may ask you to create a structure that
implements a certain [trait][trait] to handle events. If you don’t have
any data you need to store in the structure, you can create a
unit-like `struct`.

[trait]: traits.html
