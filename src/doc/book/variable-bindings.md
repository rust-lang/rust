% Variable Bindings

Virtually every non-'Hello World’ Rust program uses *variable bindings*. They
bind some value to a name, so it can be used later. `let` is
used to introduce a binding, just like this:

```rust
fn main() {
    let x = 5;
}
```

Putting `fn main() {` in each example is a bit tedious, so we’ll leave that out
in the future. If you’re following along, make sure to edit your `main()`
function, rather than leaving it off. Otherwise, you’ll get an error.

# Patterns

In many languages, a variable binding would be called a *variable*, but Rust’s
variable bindings have a few tricks up their sleeves. For example the
left-hand side of a `let` expression is a ‘[pattern][pattern]’, not just a
variable name. This means we can do things like:

```rust
let (x, y) = (1, 2);
```

After this expression is evaluated, `x` will be one, and `y` will be two.
Patterns are really powerful, and have [their own section][pattern] in the
book. We don’t need those features for now, so we’ll just keep this in the back
of our minds as we go forward.

[pattern]: patterns.html

# Type annotations

Rust is a statically typed language, which means that we specify our types up
front, and they’re checked at compile time. So why does our first example
compile? Well, Rust has this thing called ‘type inference’. If it can figure
out what the type of something is, Rust doesn’t require you to actually type it
out.

We can add the type if we want to, though. Types come after a colon (`:`):

```rust
let x: i32 = 5;
```

If I asked you to read this out loud to the rest of the class, you’d say “`x`
is a binding with the type `i32` and the value `five`.”

In this case we chose to represent `x` as a 32-bit signed integer. Rust has
many different primitive integer types. They begin with `i` for signed integers
and `u` for unsigned integers. The possible integer sizes are 8, 16, 32, and 64
bits.

In future examples, we may annotate the type in a comment. The examples will
look like this:

```rust
fn main() {
    let x = 5; // x: i32
}
```

Note the similarities between this annotation and the syntax you use with
`let`. Including these kinds of comments is not idiomatic Rust, but we'll
occasionally include them to help you understand what the types that Rust
infers are.

# Mutability

By default, bindings are *immutable*. This code will not compile:

```rust,ignore
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

```rust
let mut x = 5; // mut x: i32
x = 10;
```

There is no single reason that bindings are immutable by default, but we can
think about it through one of Rust’s primary focuses: safety. If you forget to
say `mut`, the compiler will catch it, and let you know that you have mutated
something you may not have intended to mutate. If bindings were mutable by
default, the compiler would not be able to tell you this. If you _did_ intend
mutation, then the solution is quite easy: add `mut`.

There are other good reasons to avoid mutable state when possible, but they’re
out of the scope of this guide. In general, you can often avoid explicit
mutation, and so it is preferable in Rust. That said, sometimes, mutation is
what you need, so it’s not verboten.

# Initializing bindings

Rust variable bindings have one more aspect that differs from other languages:
bindings are required to be initialized with a value before you're allowed to
use them.

Let’s try it out. Change your `src/main.rs` file to look like this:

```rust
fn main() {
    let x: i32;

    println!("Hello world!");
}
```

You can use `cargo build` on the command line to build it. You’ll get a
warning, but it will still print "Hello, world!":

```text
   Compiling hello_world v0.0.1 (file:///home/you/projects/hello_world)
src/main.rs:2:9: 2:10 warning: unused variable: `x`, #[warn(unused_variable)]
   on by default
src/main.rs:2     let x: i32;
                      ^
```

Rust warns us that we never use the variable binding, but since we never use
it, no harm, no foul. Things change if we try to actually use this `x`,
however. Let’s do that. Change your program to look like this:

```rust,ignore
fn main() {
    let x: i32;

    println!("The value of x is: {}", x);
}
```

And try to build it. You’ll get an error:

```bash
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

Rust will not let us use a value that has not been initialized. Next, let’s
talk about this stuff we've added to `println!`.

If you include two curly braces (`{}`, some call them moustaches...) in your
string to print, Rust will interpret this as a request to interpolate some sort
of value. *String interpolation* is a computer science term that means "stick
in the middle of a string." We add a comma, and then `x`, to indicate that we
want `x` to be the value we’re interpolating. The comma is used to separate
arguments we pass to functions and macros, if you’re passing more than one.

When you just use the curly braces, Rust will attempt to display the value in a
meaningful way by checking out its type. If you want to specify the format in a
more detailed manner, there are a [wide number of options available][format].
For now, we'll just stick to the default: integers aren't very complicated to
print.

[format]: ../std/fmt/index.html

# Scope and shadowing

Let’s get back to bindings. Variable bindings have a scope - they are
constrained to live in a block they were defined in. A block is a collection
of statements enclosed by `{` and `}`. Function definitions are also blocks!
In the following example we define two variable bindings, `x` and `y`, which
live in different blocks. `x` can be accessed from inside the `fn main() {}`
block, while `y` can be accessed only from inside the inner block:

```rust,ignore
fn main() {
    let x: i32 = 17;
    {
        let y: i32 = 3;
        println!("The value of x is {} and value of y is {}", x, y);
    }
    println!("The value of x is {} and value of y is {}", x, y); // This won't work
}
```

The first `println!` would print "The value of x is 17 and the value of y is
3", but this example cannot be compiled successfully, because the second
`println!` cannot access the value of `y`, since it is not in scope anymore.
Instead we get this error:

```bash
$ cargo build
   Compiling hello v0.1.0 (file:///home/you/projects/hello_world)
main.rs:7:62: 7:63 error: unresolved name `y`. Did you mean `x`? [E0425]
main.rs:7     println!("The value of x is {} and value of y is {}", x, y); // This won't work
                                                                       ^
note: in expansion of format_args!
<std macros>:2:25: 2:56 note: expansion site
<std macros>:1:1: 2:62 note: in expansion of print!
<std macros>:3:1: 3:54 note: expansion site
<std macros>:1:1: 3:58 note: in expansion of println!
main.rs:7:5: 7:65 note: expansion site
main.rs:7:62: 7:63 help: run `rustc --explain E0425` to see a detailed explanation
error: aborting due to previous error
Could not compile `hello`.

To learn more, run the command again with --verbose.
```

Additionally, variable bindings can be shadowed. This means that a later
variable binding with the same name as another binding, that's currently in
scope, will override the previous binding.

```rust
let x: i32 = 8;
{
    println!("{}", x); // Prints "8"
    let x = 12;
    println!("{}", x); // Prints "12"
}
println!("{}", x); // Prints "8"
let x =  42;
println!("{}", x); // Prints "42"
```

Shadowing and mutable bindings may appear as two sides of the same coin, but
they are two distinct concepts that can't always be used interchangeably. For
one, shadowing enables us to rebind a name to a value of a different type. It
is also possible to change the mutability of a binding.

```rust
let mut x: i32 = 1;
x = 7;
let x = x; // x is now immutable and is bound to 7

let y = 4;
let y = "I can also be bound to text!"; // y is now of a different type
```
