% Traits

Do you remember the `impl` keyword, used to call a function with method
syntax?

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
```

Traits are similar, except that we define a trait with just the method
signature, then implement the trait for that struct. Like this:

```{rust}
# #![feature(core)]
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

trait HasArea {
    fn area(&self) -> f64;
}

impl HasArea for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
    }
}
```

As you can see, the `trait` block looks very similar to the `impl` block,
but we don't define a body, just a type signature. When we `impl` a trait,
we use `impl Trait for Item`, rather than just `impl Item`.

So what's the big deal? Remember the error we were getting with our generic
`inverse` function?

```text
error: binary operation `==` cannot be applied to type `T`
```

We can use traits to constrain our generics. Consider this function, which
does not compile, and gives us a similar error:

```{rust,ignore}
fn print_area<T>(shape: T) {
    println!("This shape has an area of {}", shape.area());
}
```

Rust complains:

```text
error: type `T` does not implement any method in scope named `area`
```

Because `T` can be any type, we can't be sure that it implements the `area`
method. But we can add a *trait constraint* to our generic `T`, ensuring
that it does:

```{rust}
# trait HasArea {
#     fn area(&self) -> f64;
# }
fn print_area<T: HasArea>(shape: T) {
    println!("This shape has an area of {}", shape.area());
}
```

The syntax `<T: HasArea>` means `any type that implements the HasArea trait`.
Because traits define function type signatures, we can be sure that any type
which implements `HasArea` will have an `.area()` method.

Here's an extended example of how this works:

```{rust}
# #![feature(core)]
trait HasArea {
    fn area(&self) -> f64;
}

struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

impl HasArea for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
    }
}

struct Square {
    x: f64,
    y: f64,
    side: f64,
}

impl HasArea for Square {
    fn area(&self) -> f64 {
        self.side * self.side
    }
}

fn print_area<T: HasArea>(shape: T) {
    println!("This shape has an area of {}", shape.area());
}

fn main() {
    let c = Circle {
        x: 0.0f64,
        y: 0.0f64,
        radius: 1.0f64,
    };

    let s = Square {
        x: 0.0f64,
        y: 0.0f64,
        side: 1.0f64,
    };

    print_area(c);
    print_area(s);
}
```

This program outputs:

```text
This shape has an area of 3.141593
This shape has an area of 1
```

As you can see, `print_area` is now generic, but also ensures that we
have passed in the correct types. If we pass in an incorrect type:

```{rust,ignore}
print_area(5);
```

We get a compile-time error:

```text
error: failed to find an implementation of trait main::HasArea for int
```

So far, we've only added trait implementations to structs, but you can
implement a trait for any type. So technically, we _could_ implement
`HasArea` for `i32`:

```{rust}
trait HasArea {
    fn area(&self) -> f64;
}

impl HasArea for i32 {
    fn area(&self) -> f64 {
        println!("this is silly");

        *self as f64
    }
}

5.area();
```

It is considered poor style to implement methods on such primitive types, even
though it is possible.

This may seem like the Wild West, but there are two other restrictions around
implementing traits that prevent this from getting out of hand. First, traits
must be `use`d in any scope where you wish to use the trait's method. So for
example, this does not work:

```{rust,ignore}
mod shapes {
    use std::f64::consts;

    trait HasArea {
        fn area(&self) -> f64;
    }

    struct Circle {
        x: f64,
        y: f64,
        radius: f64,
    }

    impl HasArea for Circle {
        fn area(&self) -> f64 {
            consts::PI * (self.radius * self.radius)
        }
    }
}

fn main() {
    let c = shapes::Circle {
        x: 0.0f64,
        y: 0.0f64,
        radius: 1.0f64,
    };

    println!("{}", c.area());
}
```

Now that we've moved the structs and traits into their own module, we get an
error:

```text
error: type `shapes::Circle` does not implement any method in scope named `area`
```

If we add a `use` line right above `main` and make the right things public,
everything is fine:

```{rust}
# #![feature(core)]
mod shapes {
    use std::f64::consts;

    pub trait HasArea {
        fn area(&self) -> f64;
    }

    pub struct Circle {
        pub x: f64,
        pub y: f64,
        pub radius: f64,
    }

    impl HasArea for Circle {
        fn area(&self) -> f64 {
            consts::PI * (self.radius * self.radius)
        }
    }
}

use shapes::HasArea;

fn main() {
    let c = shapes::Circle {
        x: 0.0f64,
        y: 0.0f64,
        radius: 1.0f64,
    };

    println!("{}", c.area());
}
```

This means that even if someone does something bad like add methods to `int`,
it won't affect you, unless you `use` that trait.

There's one more restriction on implementing traits. Either the trait or the
type you're writing the `impl` for must be inside your crate. So, we could
implement the `HasArea` type for `i32`, because `HasArea` is in our crate.  But
if we tried to implement `Float`, a trait provided by Rust, for `i32`, we could
not, because both the trait and the type aren't in our crate.

One last thing about traits: generic functions with a trait bound use
*monomorphization* (*mono*: one, *morph*: form), so they are statically
dispatched. What's that mean? Check out the chapter on [trait
objects](trait-objects.html) for more.

## Multiple trait bounds

You’ve seen that you can bound a generic type parameter with a trait:

```rust
fn foo<T: Clone>(x: T) {
    x.clone();
}
```

If you need more than one bound, you can use `+`:

```rust
use std::fmt::Debug;

fn foo<T: Clone + Debug>(x: T) {
    x.clone();
    println!("{:?}", x);
}
```

`T` now needs to be both `Clone` as well as `Debug`.

## Where clause

Writing functions with only a few generic types and a small number of trait
bounds isn't too bad, but as the number increases, the syntax gets increasingly
awkward:

```
use std::fmt::Debug;

fn foo<T: Clone, K: Clone + Debug>(x: T, y: K) {
    x.clone();
    y.clone();
    println!("{:?}", y);
}
```

The name of the function is on the far left, and the parameter list is on the
far right. The bounds are getting in the way.

Rust has a solution, and it's called a '`where` clause':

```
use std::fmt::Debug;

fn foo<T: Clone, K: Clone + Debug>(x: T, y: K) {
    x.clone();
    y.clone();
    println!("{:?}", y);
}

fn bar<T, K>(x: T, y: K) where T: Clone, K: Clone + Debug {
    x.clone();
    y.clone();
    println!("{:?}", y);
}

fn main() {
    foo("Hello", "world");
    bar("Hello", "workd");
}
```

`foo()` uses the syntax we showed earlier, and `bar()` uses a `where` clause.
All you need to do is leave off the bounds when defining your type parameters,
and then add `where` after the parameter list. For longer lists, whitespace can
be added:

```
use std::fmt::Debug;

fn bar<T, K>(x: T, y: K)
    where T: Clone,
          K: Clone + Debug {

    x.clone();
    y.clone();
    println!("{:?}", y);
}
```

This flexibility can add clarity in complex situations.

`where` is also more powerful than the simpler syntax. For example:

```
trait ConvertTo<Output> {
    fn convert(&self) -> Output;
}

impl ConvertTo<i64> for i32 {
    fn convert(&self) -> i64 { *self as i64 }
}

// can be called with T == i32
fn normal<T: ConvertTo<i64>>(x: &T) -> i64 {
    x.convert()
}

// can be called with T == i64
fn inverse<T>() -> T
        // this is using ConvertTo as if it were "ConvertFrom<i32>"
        where i32: ConvertTo<T> {
    1i32.convert()
}
```

This shows off the additional feature of `where` clauses: they allow bounds
where the left-hand side is an arbitrary type (`i32` in this case), not just a
plain type parameter (like `T`).

## Our `inverse` Example

Back in [Generics](generics.html), we were trying to write code like this:

```{rust,ignore}
fn inverse<T>(x: T) -> Result<T, String> {
    if x == 0.0 { return Err("x cannot be zero!".to_string()); }

    Ok(1.0 / x)
}
```

If we try to compile it, we get this error:

```text
error: binary operation `==` cannot be applied to type `T`
```

This is because `T` is too generic: we don't know if a random `T` can be
compared. For that, we can use trait bounds. It doesn't quite work, but try
this:

```{rust,ignore}
fn inverse<T: PartialEq>(x: T) -> Result<T, String> {
    if x == 0.0 { return Err("x cannot be zero!".to_string()); }

    Ok(1.0 / x)
}
```

You should get this error:

```text
error: mismatched types:
 expected `T`,
    found `_`
(expected type parameter,
    found floating-point variable)
```

So this won't work. While our `T` is `PartialEq`, we expected to have another `T`,
but instead, we found a floating-point variable. We need a different bound. `Float`
to the rescue:

```
# #![feature(std_misc)]
use std::num::Float;

fn inverse<T: Float>(x: T) -> Result<T, String> {
    if x == Float::zero() { return Err("x cannot be zero!".to_string()) }

    let one: T = Float::one();
    Ok(one / x)
}
```

We've had to replace our generic `0.0` and `1.0` with the appropriate methods
from the `Float` trait. Both `f32` and `f64` implement `Float`, so our function
works just fine:

```
# #![feature(std_misc)]
# use std::num::Float;
# fn inverse<T: Float>(x: T) -> Result<T, String> {
#     if x == Float::zero() { return Err("x cannot be zero!".to_string()) }
#     let one: T = Float::one();
#     Ok(one / x)
# }
println!("the inverse of {} is {:?}", 2.0f32, inverse(2.0f32));
println!("the inverse of {} is {:?}", 2.0f64, inverse(2.0f64));

println!("the inverse of {} is {:?}", 0.0f32, inverse(0.0f32));
println!("the inverse of {} is {:?}", 0.0f64, inverse(0.0f64));
```

## Default methods

There's one last feature of traits we should cover: default methods. It's
easiest just to show an example:

```rust
trait Foo {
    fn bar(&self);

    fn baz(&self) { println!("We called baz."); }
}
```

Implementors of the `Foo` trait need to implement `bar()`, but they don't
need to implement `baz()`. They'll get this default behavior. They can
override the default if they so choose:

```rust
# trait Foo {
# fn bar(&self);
# fn baz(&self) { println!("We called baz."); }
# }
struct UseDefault;

impl Foo for UseDefault {
    fn bar(&self) { println!("We called bar."); }
}

struct OverrideDefault;

impl Foo for OverrideDefault {
    fn bar(&self) { println!("We called bar."); }

    fn baz(&self) { println!("Override baz!"); }
}

let default = UseDefault;
default.baz(); // prints "We called baz."

let over = OverrideDefault;
over.baz(); // prints "Override baz!"
```
