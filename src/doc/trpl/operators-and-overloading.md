% Operators and Overloading

Rust allows for a limited form of operator overloading. There are certain
operators that are able to be overloaded. To support a particular operator
between types, there’s a specific trait that you can implement, which then
overloads the operator.

For example, the `+` operator can be overloaded with the `Add` trait:

```rust
use std::ops::Add;

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point { x: self.x + other.x, y: self.y + other.y }
    }
}

fn main() {
    let p1 = Point { x: 1, y: 0 };
    let p2 = Point { x: 2, y: 3 };

    let p3 = p1 + p2;

    println!("{:?}", p3);
}
```

In `main`, we can use `+` on our two `Point`s, since we’ve implemented
`Add<Output=Point>` for `Point`.

There are a number of operators that can be overloaded this way, and all of
their associated traits live in the [`std::ops`][stdops] module. Check out its
documentation for the full list.

[stdops]: ../std/ops/index.html

Implementing these traits follows a pattern. Let’s look at [`Add`][add] in more
detail:

```rust
# mod foo {
pub trait Add<RHS = Self> {
    type Output;

    fn add(self, rhs: RHS) -> Self::Output;
}
# }
```

[add]: ../std/ops/trait.Add.html

There’s three types in total involved here: the type you `impl Add` for, `RHS`,
which defaults to `Self`, and `Output`. For an expression `let z = x + y`, `x`
is the `Self` type, `y` is the RHS, and `z` is the `Self::Output` type.

```rust
# struct Point;
# use std::ops::Add;
impl Add<i32> for Point {
    type Output = f64;

    fn add(self, rhs: i32) -> f64 {
        // add an i32 to a Point and get an f64
# 1.0
    }
}
```

will let you do this:

```rust,ignore
let p: Point = // ...
let x: f64 = p + 2i32;
```
