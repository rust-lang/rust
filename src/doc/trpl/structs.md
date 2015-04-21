% Structs

Structs are a way of creating more complex data types. For example, if we were
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

There’s a lot going on here, so let’s break it down. We declare a `struct` with
the `struct` keyword, and then with a name. By convention, `struct`s begin with
a capital letter and are camel cased: `PointInSpace`, not `Point_In_Space`.

We can create an instance of our struct via `let`, as usual, but we use a `key:
value` style syntax to set each field. The order doesn’t need to be the same as
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

Rust does not support field mutability at the language level, so you cannot
write something like this:

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

    let point = point; // this new binding can’t change now

    point.y = 6; // this causes an error
}
```

# Update syntax

A `struct` can include `..` to indicate that you want to use a copy of some
other struct for some of the values. For example:

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
