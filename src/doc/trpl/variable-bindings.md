% Variable bindings

The first thing we'll learn about are 'variable bindings.' They look like this:

```{rust}
fn main() {
    let x = 5;
}
```

Putting `fn main() {` in each example is a bit tedious, so we'll leave that out
in the future. If you're following along, make sure to edit your `main()`
function, rather than leaving it off. Otherwise, you'll get an error.

In many languages, this is called a 'variable.' But Rust's variable bindings
have a few tricks up their sleeves. Rust has a very powerful feature called
'pattern matching' that we'll get into detail with later, but the left
hand side of a `let` expression is a full pattern, not just a variable name.
This means we can do things like:

```{rust}
let (x, y) = (1, 2);
```

After this expression is evaluated, `x` will be one, and `y` will be two.
Patterns are really powerful, but this is about all we can do with them so far.
So let's just keep this in the back of our minds as we go forward.

Rust is a statically typed language, which means that we specify our types up
front. So why does our first example compile? Well, Rust has this thing called
"type inference." If it can figure out what the type of something is, Rust
doesn't require you to actually type it out.

We can add the type if we want to, though. Types come after a colon (`:`):

```{rust}
let x: i32 = 5;
```

If I asked you to read this out loud to the rest of the class, you'd say "`x`
is a binding with the type `i32` and the value `five`."

In future examples, we may annotate the type in a comment. The examples will
look like this:

```{rust}
fn main() {
    let x = 5; // x: i32
}
```

Note the similarities between this annotation and the syntax you use with `let`.
Including these kinds of comments is not idiomatic Rust, but we'll occasionally
include them to help you understand what the types that Rust infers are.

By default, bindings are **immutable**. This code will not compile:

```{ignore}
let x = 5;
x = 10;
```

It will give you this error:

```text
error: re-assignment of immutable variable `x`
     x = 10;
     ^~~~~~~
```

If you want a binding to be mutable, you can use `mut`:

```{rust}
let mut x = 5; // mut x: i32
x = 10;
```

There is no single reason that bindings are immutable by default, but we can
think about it through one of Rust's primary focuses: safety. If you forget to
say `mut`, the compiler will catch it, and let you know that you have mutated
something you may not have intended to mutate. If bindings were mutable by
default, the compiler would not be able to tell you this. If you _did_ intend
mutation, then the solution is quite easy: add `mut`.

There are other good reasons to avoid mutable state when possible, but they're
out of the scope of this guide. In general, you can often avoid explicit
mutation, and so it is preferable in Rust. That said, sometimes, mutation is
what you need, so it's not verboten.

Let's get back to bindings. Rust variable bindings have one more aspect that
differs from other languages: bindings are required to be initialized with a
value before you're allowed to use them. If we try...

```{ignore}
let x;
```

...we'll get an error:

```text
src/main.rs:2:9: 2:10 error: cannot determine a type for this local variable: unconstrained type
src/main.rs:2     let x;
                      ^
```

Giving it a type will compile, though:

```{rust}
let x: i32;
```

Let's try it out. Change your `src/main.rs` file to look like this:

```{rust}
fn main() {
    let x: i32;

    println!("Hello world!");
}
```

You can use `cargo build` on the command line to build it. You'll get a warning,
but it will still print "Hello, world!":

```text
   Compiling hello_world v0.0.1 (file:///home/you/projects/hello_world)
src/main.rs:2:9: 2:10 warning: unused variable: `x`, #[warn(unused_variable)] on by default
src/main.rs:2     let x: i32;
                      ^
```

Rust warns us that we never use the variable binding, but since we never use it,
no harm, no foul. Things change if we try to actually use this `x`, however. Let's
do that. Change your program to look like this:

```{rust,ignore}
fn main() {
    let x: i32;

    println!("The value of x is: {}", x);
}
```

And try to build it. You'll get an error:

```{bash}
$ cargo build
   Compiling hello_world v0.0.1 (file:///home/you/projects/hello_world)
src/main.rs:4:39: 4:40 error: use of possibly uninitialized variable: `x`
src/main.rs:4     println!("The value of x is: {}", x);
                                                    ^
note: in expansion of format_args!
<std macros>:2:23: 2:77 note: expansion site
<std macros>:1:1: 3:2 note: in expansion of println!
src/main.rs:4:5: 4:42 note: expansion site
error: aborting due to previous error
Could not compile `hello_world`.
```

Rust will not let us use a value that has not been initialized. Next, let's
talk about this stuff we've added to `println!`.

If you include two curly braces (`{}`, some call them moustaches...) in your
string to print, Rust will interpret this as a request to interpolate some sort
of value. **String interpolation** is a computer science term that means "stick
in the middle of a string." We add a comma, and then `x`, to indicate that we
want `x` to be the value we're interpolating. The comma is used to separate
arguments we pass to functions and macros, if you're passing more than one.

When you just use the curly braces, Rust will attempt to display the
value in a meaningful way by checking out its type. If you want to specify the
format in a more detailed manner, there are a [wide number of options
available](../std/fmt/index.html). For now, we'll just stick to the default:
integers aren't very complicated to print.
