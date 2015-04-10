% Structs

Structs are a way of creating more complex datatypes. For example, if we were
doing calculations involving coordinates in 2D space, we would need both an `x`
and a `y` value:

```rust
let origin_x = 0;
let origin_y = 0;
```

A struct lets us combine these two into a single, unified datatype:

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

There’s a lot going on here, so let’s break it down. We declare a struct with
the `struct` keyword, and then with a name. By convention, structs begin with a
capital letter and are also camel cased: `PointInSpace`, not `Point_In_Space`.

We can create an instance of our struct via `let`, as usual, but we use a `key:
value` style syntax to set each field. The order doesn't need to be the same as
in the original declaration.

Finally, because fields have names, we can access the field through dot
notation: `origin.x`.

The values in structs are immutable by default, like other bindings in Rust.
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

Rust does not support mutability at the field level, so you cannot write
something like this:

```rust,ignore
struct Point {
    mut x: i32,
    y: i32,
}
```

Mutability is a property of the binding, not of the structure itself. If you’re
used to field-level mutability, this may seem strange at first, but it
significantly simplifies things. It even lets you make things mutable for a short
time only:


```rust,ignore
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut point = Point { x: 0, y: 0 };

    point.x = 5;

    let point = point; // this new binding is immutable

    point.y = 6; // this causes an error, because `point` is immutable!
}
```
