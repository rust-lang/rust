% Closures

So far, we've made lots of functions in Rust, but we've given them all names.
Rust also allows us to create anonymous functions. Rust's anonymous
functions are called *closures*. By themselves, closures aren't all that
interesting, but when you combine them with functions that take closures as
arguments, really powerful things are possible.

Let's make a closure:

```{rust}
let add_one = |x| { 1 + x };

println!("The sum of 5 plus 1 is {}.", add_one(5));
```

We create a closure using the `|...| { ... }` syntax, and then we create a
binding so we can use it later. Note that we call the function using the
binding name and two parentheses, just like we would for a named function.

Let's compare syntax. The two are pretty close:

```{rust}
let add_one = |x: i32| -> i32 { 1 + x };
fn  add_one   (x: i32) -> i32 { 1 + x }
```

As you may have noticed, closures infer their argument and return types, so you
don't need to declare one. This is different from named functions, which
default to returning unit (`()`).

There's one big difference between a closure and named functions, and it's in
the name: a closure "closes over its environment." What does that mean? It means
this:

```{rust}
fn main() {
    let x: i32 = 5;

    let printer = || { println!("x is: {}", x); };

    printer(); // prints "x is: 5"
}
```

The `||` syntax means this is an anonymous closure that takes no arguments.
Without it, we'd just have a block of code in `{}`s.

In other words, a closure has access to variables in the scope where it's
defined. The closure borrows any variables it uses, so this will error:

```{rust,ignore}
fn main() {
    let mut x: i32 = 5;

    let printer = || { println!("x is: {}", x); };

    x = 6; // error: cannot assign to `x` because it is borrowed
}
```

## Moving closures

Rust has a second type of closure, called a *moving closure*. Moving
closures are indicated using the `move` keyword (e.g., `move || x *
x`). The difference between a moving closure and an ordinary closure
is that a moving closure always takes ownership of all variables that
it uses. Ordinary closures, in contrast, just create a reference into
the enclosing stack frame. Moving closures are most useful with Rust's
concurrency features, and so we'll just leave it at this for
now. We'll talk about them more in the "Concurrency" chapter of the book.

## Accepting closures as arguments

Closures are most useful as an argument to another function. Here's an example:

```{rust}
fn twice<F: Fn(i32) -> i32>(x: i32, f: F) -> i32 {
    f(x) + f(x)
}

fn main() {
    let square = |x: i32| { x * x };

    twice(5, square); // evaluates to 50
}
```

Let's break the example down, starting with `main`:

```{rust}
let square = |x: i32| { x * x };
```

We've seen this before. We make a closure that takes an integer, and returns
its square.

```{rust}
# fn twice<F: Fn(i32) -> i32>(x: i32, f: F) -> i32 { f(x) + f(x) }
# let square = |x: i32| { x * x };
twice(5, square); // evaluates to 50
```

This line is more interesting. Here, we call our function, `twice`, and we pass
it two arguments: an integer, `5`, and our closure, `square`. This is just like
passing any other two variable bindings to a function, but if you've never
worked with closures before, it can seem a little complex. Just think: "I'm
passing two variables: one is an i32, and one is a function."

Next, let's look at how `twice` is defined:

```{rust,ignore}
fn twice<F: Fn(i32) -> i32>(x: i32, f: F) -> i32 {
```

`twice` takes two arguments, `x` and `f`. That's why we called it with two
arguments. `x` is an `i32`, we've done that a ton of times. `f` is a function,
though, and that function takes an `i32` and returns an `i32`. This is
what the requirement `Fn(i32) -> i32` for the type parameter `F` says.
Now `F` represents *any* function that takes an `i32` and returns an `i32`.

This is the most complicated function signature we've seen yet! Give it a read
a few times until you can see how it works. It takes a teeny bit of practice, and
then it's easy. The good news is that this kind of passing a closure around
can be very efficient. With all the type information available at compile-time
the compiler can do wonders.

Finally, `twice` returns an `i32` as well.

Okay, let's look at the body of `twice`:

```{rust}
fn twice<F: Fn(i32) -> i32>(x: i32, f: F) -> i32 {
  f(x) + f(x)
}
```

Since our closure is named `f`, we can call it just like we called our closures
before, and we pass in our `x` argument to each one, hence the name `twice`.

If you do the math, `(5 * 5) + (5 * 5) == 50`, so that's the output we get.

Play around with this concept until you're comfortable with it. Rust's standard
library uses lots of closures where appropriate, so you'll be using
this technique a lot.

If we didn't want to give `square` a name, we could just define it inline.
This example is the same as the previous one:

```{rust}
fn twice<F: Fn(i32) -> i32>(x: i32, f: F) -> i32 {
    f(x) + f(x)
}

fn main() {
    twice(5, |x: i32| { x * x }); // evaluates to 50
}
```

A named function's name can be used wherever you'd use a closure. Another
way of writing the previous example:

```{rust}
fn twice<F: Fn(i32) -> i32>(x: i32, f: F) -> i32 {
    f(x) + f(x)
}

fn square(x: i32) -> i32 { x * x }

fn main() {
    twice(5, square); // evaluates to 50
}
```

Doing this is not particularly common, but it's useful every once in a while.

Before we move on, let us look at a function that accepts two closures.

```{rust}
fn compose<F, G>(x: i32, f: F, g: G) -> i32
    where F: Fn(i32) -> i32, G: Fn(i32) -> i32 {
    g(f(x))
}

fn main() {
    compose(5,
            |n: i32| { n + 42 },
            |n: i32| { n * 2 }); // evaluates to 94
}
```

You might ask yourself: why do we need to introduce two type
parameters `F` and `G` here?  Evidently, both `f` and `g` have the
same signature: `Fn(i32) -> i32`.

That is because in Rust each closure has its own unique type.
So, not only do closures with different signatures have different types,
but different closures with the *same* signature have *different*
types, as well!

You can think of it this way: the behavior of a closure is part of its
type.  Therefore, using a single type parameter for both closures
will accept the first of them, rejecting the second. The distinct
type of the second closure does not allow it to be represented by the
same type parameter as that of the first.  We acknowledge this, and
use two different type parameters `F` and `G`.

This also introduces the `where` clause, which lets us describe type
parameters in a more flexible manner.

That's all you need to get the hang of closures! Closures are a little bit
strange at first, but once you're used to them, you'll miss them
in other languages. Passing functions to other functions is
incredibly powerful, as you will see in the following chapter about iterators.
