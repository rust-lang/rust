% Method Syntax

Functions are great, but if you want to call a bunch of them on some data, it
can be awkward. Consider this code:

```{rust,ignore}
baz(bar(foo(x)));
```

We would read this left-to right, and so we see "baz bar foo." But this isn't the
order that the functions would get called in, that's inside-out: "foo bar baz."
Wouldn't it be nice if we could do this instead?

```{rust,ignore}
x.foo().bar().baz();
```

Luckily, as you may have guessed with the leading question, you can! Rust provides
the ability to use this *method call syntax* via the `impl` keyword.

## Method calls

Here's how it works:

```{rust}
# #![feature(core)]
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

We've made a struct that represents a circle. We then write an `impl` block,
and inside it, define a method, `area`. Methods take a  special first
parameter, `&self`. There are three variants: `self`, `&self`, and `&mut self`.
You can think of this first parameter as being the `x` in `x.foo()`. The three
variants correspond to the three kinds of thing `x` could be: `self` if it's
just a value on the stack, `&self` if it's a reference, and `&mut self` if it's
a mutable reference. We should default to using `&self`, as it's the most
common. Here's an example of all three variants:

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

Finally, as you may remember, the value of the area of a circle is `π*r²`.
Because we took the `&self` parameter to `area`, we can use it just like any
other parameter. Because we know it's a `Circle`, we can access the `radius`
just like we would with any other struct. An import of π and some
multiplications later, and we have our area.

## Chaining method calls

So, now we know how to call a method, such as `foo.bar()`. But what about our
original example, `foo.bar().baz()`? This is called 'method chaining', and we
can do it by returning `self`.

```
# #![feature(core)]
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
    }

    fn grow(&self) -> Circle {
        Circle { x: self.x, y: self.y, radius: (self.radius * 10.0) }
    }
}

fn main() {
    let c = Circle { x: 0.0, y: 0.0, radius: 2.0 };
    println!("{}", c.area());

    let d = c.grow().area();
    println!("{}", d);
}
```

Check the return type:

```
# struct Circle;
# impl Circle {
fn grow(&self) -> Circle {
# Circle } }
```

We just say we're returning a `Circle`. With this method, we can grow a new
circle with an area that's 100 times larger than the old one.

## Static methods

You can also define methods that do not take a `self` parameter. Here's a
pattern that's very common in Rust code:

```
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

This *static method* builds a new `Circle` for us. Note that static methods
are called with the `Struct::method()` syntax, rather than the `ref.method()`
syntax.

## Builder Pattern

Let's say that we want our users to be able to create Circles, but we will
allow them to only set the properties they care about. Otherwise, the `x`
and `y` attributes will be `0.0`, and the `radius` will be `1.0`. Rust doesn't
have method overloading, named arguments, or variable arguments. We employ
the builder pattern instead. It looks like this:

```
# #![feature(core)]
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl Circle {
    fn new() -> Circle {
        Circle { x: 0.0, y: 0.0, radius: 0.0, }
    }

    fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
    }

    fn set_x(&mut self, coordinate: f64) -> &mut Circle {
        self.x = coordinate;
        self
    }

    fn set_y(&mut self, coordinate: f64) -> &mut Circle {
        self.y = coordinate;
        self
    }

    fn set_radius(&mut self, radius: f64) -> &mut Circle {
        self.radius = radius;
        self
    }

    fn finalize(&self) -> Circle {
        Circle { x: self.x, y: self.y, radius: self.radius }
    }
}

fn main() {
    let c = Circle::new()
                .set_x(1.0)
                .set_y(2.0)
                .set_radius(2.0)
                .finalize();


    println!("area: {}", c.area());
}
```

What we've done here is define our builder methods on our struct. We
also added a method: `finalize()`. This method creates
our final `Circle` from the builder. Now, we've used the type system to enforce
our concerns: we can use these methods to constrain making
`Circle`s in any way we choose.
