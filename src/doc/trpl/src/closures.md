% Closures

So far, we've made lots of functions in Rust, but we've given them all names.
Rust also allows us to create anonymous functions. Rust's anonymous
functions are called **closure**s. By themselves, closures aren't all that
interesting, but when you combine them with functions that take closures as
arguments, really powerful things are possible.

Let's make a closure:

```{rust}
let add_one = |x| { 1i + x };

println!("The sum of 5 plus 1 is {}.", add_one(5i));
```

We create a closure using the `|...| { ... }` syntax, and then we create a
binding so we can use it later. Note that we call the function using the
binding name and two parentheses, just like we would for a named function.

Let's compare syntax. The two are pretty close:

```{rust}
let add_one = |x: int| -> int { 1i + x };
fn  add_one   (x: int) -> int { 1i + x }
```

As you may have noticed, closures infer their argument and return types, so you
don't need to declare one. This is different from named functions, which
default to returning unit (`()`).

There's one big difference between a closure and named functions, and it's in
the name: a closure "closes over its environment." What does that mean? It means
this:

```{rust}
fn main() {
    let x = 5i;

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
    let mut x = 5i;

    let printer = || { println!("x is: {}", x); };

    x = 6i; // error: cannot assign to `x` because it is borrowed
}
```

## Procs

Rust has a second type of closure, called a **proc**. Procs are created
with the `proc` keyword:

```{rust}
let x = 5i;

let p = proc() { x * x };
println!("{}", p()); // prints 25
```

There is a big difference between procs and closures: procs may only be called once. This
will error when we try to compile:

```{rust,ignore}
let x = 5i;

let p = proc() { x * x };
println!("{}", p());
println!("{}", p()); // error: use of moved value `p`
```

This restriction is important. Procs are allowed to consume values that they
capture, and thus have to be restricted to being called once for soundness
reasons: any value consumed would be invalid on a second call.

Procs are most useful with Rust's concurrency features, and so we'll just leave
it at this for now. We'll talk about them more in the "Tasks" section of the
guide.

## Accepting closures as arguments

Closures are most useful as an argument to another function. Here's an example:

```{rust}
fn twice(x: int, f: |int| -> int) -> int {
    f(x) + f(x)
}

fn main() {
    let square = |x: int| { x * x };

    twice(5i, square); // evaluates to 50
}
```

Let's break the example down, starting with `main`:

```{rust}
let square = |x: int| { x * x };
```

We've seen this before. We make a closure that takes an integer, and returns
its square.

```{rust,ignore}
twice(5i, square); // evaluates to 50
```

This line is more interesting. Here, we call our function, `twice`, and we pass
it two arguments: an integer, `5`, and our closure, `square`. This is just like
passing any other two variable bindings to a function, but if you've never
worked with closures before, it can seem a little complex. Just think: "I'm
passing two variables, one is an int, and one is a function."

Next, let's look at how `twice` is defined:

```{rust,ignore}
fn twice(x: int, f: |int| -> int) -> int {
```

`twice` takes two arguments, `x` and `f`. That's why we called it with two
arguments. `x` is an `int`, we've done that a ton of times. `f` is a function,
though, and that function takes an `int` and returns an `int`. Notice
how the `|int| -> int` syntax looks a lot like our definition of `square`
above, if we added the return type in:

```{rust}
let square = |x: int| -> int { x * x };
//           |int|    -> int
```

This function takes an `int` and returns an `int`.

This is the most complicated function signature we've seen yet! Give it a read
a few times until you can see how it works. It takes a teeny bit of practice, and
then it's easy.

Finally, `twice` returns an `int` as well.

Okay, let's look at the body of `twice`:

```{rust}
fn twice(x: int, f: |int| -> int) -> int {
  f(x) + f(x)
}
```

Since our closure is named `f`, we can call it just like we called our closures
before. And we pass in our `x` argument to each one. Hence 'twice.'

If you do the math, `(5 * 5) + (5 * 5) == 50`, so that's the output we get.

Play around with this concept until you're comfortable with it. Rust's standard
library uses lots of closures where appropriate, so you'll be using
this technique a lot.

If we didn't want to give `square` a name, we could just define it inline.
This example is the same as the previous one:

```{rust}
fn twice(x: int, f: |int| -> int) -> int {
    f(x) + f(x)
}

fn main() {
    twice(5i, |x: int| { x * x }); // evaluates to 50
}
```

A named function's name can be used wherever you'd use a closure. Another
way of writing the previous example:

```{rust}
fn twice(x: int, f: |int| -> int) -> int {
    f(x) + f(x)
}

fn square(x: int) -> int { x * x }

fn main() {
    twice(5i, square); // evaluates to 50
}
```

Doing this is not particularly common, but it's useful every once in a while.

That's all you need to get the hang of closures! Closures are a little bit
strange at first, but once you're used to them, you'll miss them
in other languages. Passing functions to other functions is
incredibly powerful, as you will see in the following chapter about iterators.

