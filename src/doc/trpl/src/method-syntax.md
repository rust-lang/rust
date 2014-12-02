% Method Syntax

Functions are great, but if you want to call a bunch of them on some data, it
can be awkward. Consider this code:

```{rust,ignore}
baz(bar(foo(x)));
```

We would read this left-to right, and so we see 'baz bar foo.' But this isn't the
order that the functions would get called in, that's inside-out: 'foo bar baz.'
Wouldn't it be nice if we could do this instead?

```{rust,ignore}
x.foo().bar().baz();
```

Luckily, as you may have guessed with the leading question, you can! Rust provides
the ability to use this **method call syntax** via the `impl` keyword.

Here's how it works:

```{rust}
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
common.

Finally, as you may remember, the value of the area of a circle is `π*r²`.
Because we took the `&self` parameter to `area`, we can use it just like any
other parameter. Because we know it's a `Circle`, we can access the `radius`
just like we would with any other struct. An import of π and some
multiplications later, and we have our area.

You can also define methods that do not take a `self` parameter. Here's a
pattern that's very common in Rust code:

```{rust}
# #![allow(non_shorthand_field_patterns)]
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

This **static method** builds a new `Circle` for us. Note that static methods
are called with the `Struct::method()` syntax, rather than the `ref.method()`
syntax.

