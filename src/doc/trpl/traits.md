% Traits

A trait is a language feature that tells the Rust compiler about
functionality a type must provide.

Do you remember the `impl` keyword, used to call a function with [method
syntax][methodsyntax]?

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
```

[methodsyntax]: method-syntax.html

Traits are similar, except that we define a trait with just the method
signature, then implement the trait for that struct. Like this:

```rust
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
but we don’t define a body, just a type signature. When we `impl` a trait,
we use `impl Trait for Item`, rather than just `impl Item`.

## Traits bounds for generic functions

Traits are useful because they allow a type to make certain promises about its
behavior. Generic functions can exploit this to constrain the types they
accept. Consider this function, which does not compile:

```rust,ignore
fn print_area<T>(shape: T) {
    println!("This shape has an area of {}", shape.area());
}
```

Rust complains:

```text
error: no method named `area` found for type `T` in the current scope
```

Because `T` can be any type, we can’t be sure that it implements the `area`
method. But we can add a ‘trait constraint’ to our generic `T`, ensuring
that it does:

```rust
# trait HasArea {
#     fn area(&self) -> f64;
# }
fn print_area<T: HasArea>(shape: T) {
    println!("This shape has an area of {}", shape.area());
}
```

The syntax `<T: HasArea>` means “any type that implements the `HasArea` trait.”
Because traits define function type signatures, we can be sure that any type
which implements `HasArea` will have an `.area()` method.

Here’s an extended example of how this works:

```rust
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

As you can see, `print_area` is now generic, but also ensures that we have
passed in the correct types. If we pass in an incorrect type:

```rust,ignore
print_area(5);
```

We get a compile-time error:

```text
error: the trait `HasArea` is not implemented for the type `_` [E0277]
```

## Traits bounds for generic structs

Your generic structs can also benefit from trait constraints. All you need to
do is append the constraint when you declare type parameters. Here is a new
type `Rectangle<T>` and its operation `is_square()`:

```rust
struct Rectangle<T> {
    x: T,
    y: T,
    width: T,
    height: T,
}

impl<T: PartialEq> Rectangle<T> {
    fn is_square(&self) -> bool {
        self.width == self.height
    }
}

fn main() {
    let mut r = Rectangle {
        x: 0,
        y: 0,
        width: 47,
        height: 47,
    };

    assert!(r.is_square());

    r.height = 42;
    assert!(!r.is_square());
}
```

`is_square()` needs to check that the sides are equal, so the sides must be of
a type that implements the [`core::cmp::PartialEq`][PartialEq] trait:

```ignore
impl<T: PartialEq> Rectangle<T> { ... }
```

Now, a rectangle can be defined in terms of any type that can be compared for
equality.

[PartialEq]: ../core/cmp/trait.PartialEq.html

Here we defined a new struct `Rectangle` that accepts numbers of any
precision—really, objects of pretty much any type—as long as they can be
compared for equality. Could we do the same for our `HasArea` structs, `Square`
and `Circle`? Yes, but they need multiplication, and to work with that we need
to know more about [operator traits][operators-and-overloading].

[operators-and-overloading]: operators-and-overloading.html

# Rules for implementing traits

So far, we’ve only added trait implementations to structs, but you can
implement a trait for any type. So technically, we _could_ implement `HasArea`
for `i32`:

```rust
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

This may seem like the Wild West, but there are two restrictions around
implementing traits that prevent this from getting out of hand. The first is
that if the trait isn’t defined in your scope, it doesn’t apply. Here’s an
example: the standard library provides a [`Write`][write] trait which adds
extra functionality to `File`s, for doing file I/O. By default, a `File`
won’t have its methods:

[write]: ../std/io/trait.Write.html

```rust,ignore
let mut f = std::fs::File::open("foo.txt").ok().expect("Couldn’t open foo.txt");
let buf = b"whatever"; // byte string literal. buf: &[u8; 8]
let result = f.write(buf);
# result.unwrap(); // ignore the error
```

Here’s the error:

```text
error: type `std::fs::File` does not implement any method in scope named `write`
let result = f.write(buf);
               ^~~~~~~~~~
```

We need to `use` the `Write` trait first:

```rust,ignore
use std::io::Write;

let mut f = std::fs::File::open("foo.txt").ok().expect("Couldn’t open foo.txt");
let buf = b"whatever";
let result = f.write(buf);
# result.unwrap(); // ignore the error
```

This will compile without error.

This means that even if someone does something bad like add methods to `i32`,
it won’t affect you, unless you `use` that trait.

There’s one more restriction on implementing traits: either the trait, or the
type you’re writing the `impl` for, must be defined by you. So, we could
implement the `HasArea` type for `i32`, because `HasArea` is in our code. But
if we tried to implement `ToString`, a trait provided by Rust, for `i32`, we could
not, because neither the trait nor the type are in our code.

One last thing about traits: generic functions with a trait bound use
‘monomorphization’ (mono: one, morph: form), so they are statically dispatched.
What’s that mean? Check out the chapter on [trait objects][to] for more details.

[to]: trait-objects.html

# Multiple trait bounds

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

# Where clause

Writing functions with only a few generic types and a small number of trait
bounds isn’t too bad, but as the number increases, the syntax gets increasingly
awkward:

```rust
use std::fmt::Debug;

fn foo<T: Clone, K: Clone + Debug>(x: T, y: K) {
    x.clone();
    y.clone();
    println!("{:?}", y);
}
```

The name of the function is on the far left, and the parameter list is on the
far right. The bounds are getting in the way.

Rust has a solution, and it’s called a ‘`where` clause’:

```rust
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
    bar("Hello", "world");
}
```

`foo()` uses the syntax we showed earlier, and `bar()` uses a `where` clause.
All you need to do is leave off the bounds when defining your type parameters,
and then add `where` after the parameter list. For longer lists, whitespace can
be added:

```rust
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

```rust
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
        // this is using ConvertTo as if it were "ConvertTo<i64>"
        where i32: ConvertTo<T> {
    42.convert()
}
```

This shows off the additional feature of `where` clauses: they allow bounds
where the left-hand side is an arbitrary type (`i32` in this case), not just a
plain type parameter (like `T`).

# Default methods

If you already know how a typical implementor will define a method, you can
let your trait supply a default:

```rust
trait Foo {
    fn is_valid(&self) -> bool;

    fn is_invalid(&self) -> bool { !self.is_valid() }
}
```

Implementors of the `Foo` trait need to implement `is_valid()`, but they don’t
need to implement `is_invalid()`. They’ll get this default behavior. They can
override the default if they so choose:

```rust
# trait Foo {
#     fn is_valid(&self) -> bool;
#
#     fn is_invalid(&self) -> bool { !self.is_valid() }
# }
struct UseDefault;

impl Foo for UseDefault {
    fn is_valid(&self) -> bool {
        println!("Called UseDefault.is_valid.");
        true
    }
}

struct OverrideDefault;

impl Foo for OverrideDefault {
    fn is_valid(&self) -> bool {
        println!("Called OverrideDefault.is_valid.");
        true
    }

    fn is_invalid(&self) -> bool {
        println!("Called OverrideDefault.is_invalid!");
        true // this implementation is a self-contradiction!
    }
}

let default = UseDefault;
assert!(!default.is_invalid()); // prints "Called UseDefault.is_valid."

let over = OverrideDefault;
assert!(over.is_invalid()); // prints "Called OverrideDefault.is_invalid!"
```

# Inheritance

Sometimes, implementing a trait requires implementing another trait:

```rust
trait Foo {
    fn foo(&self);
}

trait FooBar : Foo {
    fn foobar(&self);
}
```

Implementors of `FooBar` must also implement `Foo`, like this:

```rust
# trait Foo {
#     fn foo(&self);
# }
# trait FooBar : Foo {
#     fn foobar(&self);
# }
struct Baz;

impl Foo for Baz {
    fn foo(&self) { println!("foo"); }
}

impl FooBar for Baz {
    fn foobar(&self) { println!("foobar"); }
}
```

If we forget to implement `Foo`, Rust will tell us:

```text
error: the trait `main::Foo` is not implemented for the type `main::Baz` [E0277]
```

# Deriving

Implementing traits like `Debug` and `Default` over and over again can become
quite tedious. For that reason, Rust provides an [attribute][attributes] that
allows you to let Rust automatically implement traits for you:

```rust
#[derive(Debug)]
struct Foo;

fn main() {
    println!("{:?}", Foo);
}
```

[attributes]: attributes.html

However, deriving is limited to a certain set of traits:

- [`Clone`](../core/clone/trait.Clone.html)
- [`Copy`](../core/marker/trait.Copy.html)
- [`Debug`](../core/fmt/trait.Debug.html)
- [`Default`](../core/default/trait.Default.html)
- [`Eq`](../core/cmp/trait.Eq.html)
- [`Hash`](../core/hash/trait.Hash.html)
- [`Ord`](../core/cmp/trait.Ord.html)
- [`PartialEq`](../core/cmp/trait.PartialEq.html)
- [`PartialOrd`](../core/cmp/trait.PartialOrd.html)
