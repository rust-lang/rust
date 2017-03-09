# Method Syntax

Functions are great, but if you want to call a bunch of them on some data, it
can be awkward. Consider this code:

```rust,ignore
baz(bar(foo));
```

We would read this left-to-right, and so we see ‘baz bar foo’. But this isn’t the
order that the functions would get called in, that’s inside-out: ‘foo bar baz’.
Wouldn’t it be nice if we could do this instead?

```rust,ignore
foo.bar().baz();
```

Luckily, as you may have guessed with the leading question, you can! Rust provides
the ability to use this ‘method call syntax’ via the `impl` keyword.

# Method calls

Here’s how it works:

```rust
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
    }
}

fn main() {
    let c = Circle { x: 0.0, y: 0.0, radius: 2.0 };
    println!("{}", c.area());
}
```

This will print `12.566371`.

We’ve made a `struct` that represents a circle. We then write an `impl` block,
and inside it, define a method, `area`.

Methods take a special first parameter, of which there are three variants:
`self`, `&self`, and `&mut self`. You can think of this first parameter as
being the `foo` in `foo.bar()`. The three variants correspond to the three
kinds of things `foo` could be: `self` if it’s a value on the stack,
`&self` if it’s a reference, and `&mut self` if it’s a mutable reference.
Because we took the `&self` parameter to `area`, we can use it like any
other parameter. Because we know it’s a `Circle`, we can access the `radius`
like we would with any other `struct`.

We should default to using `&self`, as you should prefer borrowing over taking
ownership, as well as taking immutable references over mutable ones. Here’s an
example of all three variants:

```rust
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Circle {
    fn reference(&self) {
       println!("taking self by reference!");
    }

    fn mutable_reference(&mut self) {
       println!("taking self by mutable reference!");
    }

    fn takes_ownership(self) {
       println!("taking ownership of self!");
    }
}
```

You can use as many `impl` blocks as you’d like. The previous example could
have also been written like this:

```rust
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Circle {
    fn reference(&self) {
       println!("taking self by reference!");
    }
}

impl Circle {
    fn mutable_reference(&mut self) {
       println!("taking self by mutable reference!");
    }
}

impl Circle {
    fn takes_ownership(self) {
       println!("taking ownership of self!");
    }
}
```

# Chaining method calls

So, now we know how to call a method, such as `foo.bar()`. But what about our
original example, `foo.bar().baz()`? This is called ‘method chaining’. Let’s
look at an example:

```rust
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
    }

    fn grow(&self, increment: f64) -> Circle {
        Circle { x: self.x, y: self.y, radius: self.radius + increment }
    }
}

fn main() {
    let c = Circle { x: 0.0, y: 0.0, radius: 2.0 };
    println!("{}", c.area());

    let d = c.grow(2.0).area();
    println!("{}", d);
}
```

Check the return type:

```rust
# struct Circle;
# impl Circle {
fn grow(&self, increment: f64) -> Circle {
# Circle } }
```

We say we’re returning a `Circle`. With this method, we can grow a new
`Circle` to any arbitrary size.

# Associated functions

You can also define associated functions that do not take a `self` parameter.
Here’s a pattern that’s very common in Rust code:

```rust
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Circle {
    fn new(x: f64, y: f64, radius: f64) -> Circle {
        Circle {
            x: x,
            y: y,
            radius: radius,
        }
    }
}

fn main() {
    let c = Circle::new(0.0, 0.0, 2.0);
}
```

This ‘associated function’ builds a new `Circle` for us. Note that associated
functions are called with the `Struct::function()` syntax, rather than the
`ref.method()` syntax. Some other languages call associated functions ‘static
methods’.

# Builder Pattern

Let’s say that we want our users to be able to create `Circle`s, but we will
allow them to only set the properties they care about. Otherwise, the `x`
and `y` attributes will be `0.0`, and the `radius` will be `1.0`. Rust doesn’t
have method overloading, named arguments, or variable arguments. We employ
the builder pattern instead. It looks like this:

```rust
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
    }
}

struct CircleBuilder {
    x: f64,
    y: f64,
    radius: f64,
}

impl CircleBuilder {
    fn new() -> CircleBuilder {
        CircleBuilder { x: 0.0, y: 0.0, radius: 1.0, }
    }

    fn x(&mut self, coordinate: f64) -> &mut CircleBuilder {
        self.x = coordinate;
        self
    }

    fn y(&mut self, coordinate: f64) -> &mut CircleBuilder {
        self.y = coordinate;
        self
    }

    fn radius(&mut self, radius: f64) -> &mut CircleBuilder {
        self.radius = radius;
        self
    }

    fn finalize(&self) -> Circle {
        Circle { x: self.x, y: self.y, radius: self.radius }
    }
}

fn main() {
    let c = CircleBuilder::new()
                .x(1.0)
                .y(2.0)
                .radius(2.0)
                .finalize();

    println!("area: {}", c.area());
    println!("x: {}", c.x);
    println!("y: {}", c.y);
}
```

What we’ve done here is make another `struct`, `CircleBuilder`. We’ve defined our
builder methods on it. We’ve also defined our `area()` method on `Circle`. We
also made one more method on `CircleBuilder`: `finalize()`. This method creates
our final `Circle` from the builder. Now, we’ve used the type system to enforce
our concerns: we can use the methods on `CircleBuilder` to constrain making
`Circle`s in any way we choose.
