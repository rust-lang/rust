# Functions

Every Rust program has at least one function, the `main` function:

```rust
fn main() {
}
```

This is the simplest possible function declaration. As we mentioned before,
`fn` says ‘this is a function’, followed by the name, some parentheses because
this function takes no arguments, and then some curly braces to indicate the
body. Here’s a function named `foo`:

```rust
fn foo() {
}
```

So, what about taking arguments? Here’s a function that prints a number:

```rust
fn print_number(x: i32) {
    println!("x is: {}", x);
}
```

Here’s a complete program that uses `print_number`:

```rust
fn main() {
    print_number(5);
}

fn print_number(x: i32) {
    println!("x is: {}", x);
}
```

As you can see, function arguments work very similar to `let` declarations:
you add a type to the argument name, after a colon.

Here’s a complete program that adds two numbers together and prints them:

```rust
fn main() {
    print_sum(5, 6);
}

fn print_sum(x: i32, y: i32) {
    println!("sum is: {}", x + y);
}
```

You separate arguments with a comma, both when you call the function, as well
as when you declare it.

Unlike `let`, you _must_ declare the types of function arguments. This does
not work:

```rust,ignore
fn print_sum(x, y) {
    println!("sum is: {}", x + y);
}
```

You get this error:

```text
expected one of `!`, `:`, or `@`, found `)`
fn print_sum(x, y) {
```

This is a deliberate design decision. While full-program inference is possible,
languages which have it, like Haskell, often suggest that documenting your
types explicitly is a best-practice. We agree that forcing functions to declare
types while allowing for inference inside of function bodies is a wonderful
sweet spot between full inference and no inference.

What about returning a value? Here’s a function that adds one to an integer:

```rust
fn add_one(x: i32) -> i32 {
    x + 1
}
```

Rust functions return exactly one value, and you declare the type after an
‘arrow’, which is a dash (`-`) followed by a greater-than sign (`>`). The last
line of a function determines what it returns. You’ll note the lack of a
semicolon here. If we added it in:

```rust,ignore
fn add_one(x: i32) -> i32 {
    x + 1;
}
```

We would get an error:

```text
error: not all control paths return a value
fn add_one(x: i32) -> i32 {
     x + 1;
}

help: consider removing this semicolon:
     x + 1;
          ^
```

This reveals two interesting things about Rust: it is an expression-based
language, and semicolons are different from semicolons in other ‘curly brace
and semicolon’-based languages. These two things are related.

## Expressions vs. Statements

Rust is primarily an expression-based language. There are only two kinds of
statements, and everything else is an expression.

So what's the difference? Expressions return a value, and statements do not.
That’s why we end up with ‘not all control paths return a value’ here: the
statement `x + 1;` doesn’t return a value. There are two kinds of statements in
Rust: ‘declaration statements’ and ‘expression statements’. Everything else is
an expression. Let’s talk about declaration statements first.

In some languages, variable bindings can be written as expressions, not
statements. Like Ruby:

```ruby
x = y = 5
```

In Rust, however, using `let` to introduce a binding is _not_ an expression. The
following will produce a compile-time error:

```rust,ignore
let x = (let y = 5); // Expected identifier, found keyword `let`.
```

The compiler is telling us here that it was expecting to see the beginning of
an expression, and a `let` can only begin a statement, not an expression.

Note that assigning to an already-bound variable (e.g. `y = 5`) is still an
expression, although its value is not particularly useful. Unlike other
languages where an assignment evaluates to the assigned value (e.g. `5` in the
previous example), in Rust the value of an assignment is an empty tuple `()`
because the assigned value can have [only one owner](ownership.html), and any
other returned value would be too surprising:

```rust
let mut y = 5;

let x = (y = 6);  // `x` has the value `()`, not `6`.
```

The second kind of statement in Rust is the *expression statement*. Its
purpose is to turn any expression into a statement. In practical terms, Rust's
grammar expects statements to follow other statements. This means that you use
semicolons to separate expressions from each other. This means that Rust
looks a lot like most other languages that require you to use semicolons
at the end of every line, and you will see semicolons at the end of almost
every line of Rust code you see.

What is this exception that makes us say "almost"? You saw it already, in this
code:

```rust
fn add_one(x: i32) -> i32 {
    x + 1
}
```

Our function claims to return an `i32`, but with a semicolon, it would return
`()` instead. Rust realizes this probably isn’t what we want, and suggests
removing the semicolon in the error we saw before.

## Early returns

But what about early returns? Rust does have a keyword for that, `return`:

```rust
fn foo(x: i32) -> i32 {
    return x;

    // We never run this code!
    x + 1
}
```

Using a `return` as the last line of a function works, but is considered poor
style:

```rust
fn foo(x: i32) -> i32 {
    return x + 1;
}
```

The previous definition without `return` may look a bit strange if you haven’t
worked in an expression-based language before, but it becomes intuitive over
time.

## Diverging functions

Rust has some special syntax for ‘diverging functions’, which are functions that
do not return:

```rust
fn diverges() -> ! {
    panic!("This function never returns!");
}
```

`panic!` is a macro, similar to `println!()` that we’ve already seen. Unlike
`println!()`, `panic!()` causes the current thread of execution to crash with
the given message. Because this function will cause a crash, it will never
return, and so it has the type ‘`!`’, which is read ‘diverges’.

If you add a main function that calls `diverges()` and run it, you’ll get
some output that looks like this:

```text
thread ‘main’ panicked at ‘This function never returns!’, hello.rs:2
```

If you want more information, you can get a backtrace by setting the
`RUST_BACKTRACE` environment variable:

```text
$ RUST_BACKTRACE=1 ./diverges
thread 'main' panicked at 'This function never returns!', hello.rs:2
Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
stack backtrace:
  hello::diverges
        at ./hello.rs:2
  hello::main
        at ./hello.rs:6
```

If you want the complete backtrace and filenames:

```text
$ RUST_BACKTRACE=full ./diverges
thread 'main' panicked at 'This function never returns!', hello.rs:2
stack backtrace:
   1:     0x7f402773a829 - sys::backtrace::write::h0942de78b6c02817K8r
   2:     0x7f402773d7fc - panicking::on_panic::h3f23f9d0b5f4c91bu9w
   3:     0x7f402773960e - rt::unwind::begin_unwind_inner::h2844b8c5e81e79558Bw
   4:     0x7f4027738893 - rt::unwind::begin_unwind::h4375279447423903650
   5:     0x7f4027738809 - diverges::h2266b4c4b850236beaa
   6:     0x7f40277389e5 - main::h19bb1149c2f00ecfBaa
   7:     0x7f402773f514 - rt::unwind::try::try_fn::h13186883479104382231
   8:     0x7f402773d1d8 - __rust_try
   9:     0x7f402773f201 - rt::lang_start::ha172a3ce74bb453aK5w
  10:     0x7f4027738a19 - main
  11:     0x7f402694ab44 - __libc_start_main
  12:     0x7f40277386c8 - <unknown>
  13:                0x0 - <unknown>
```

If you need to override an already set `RUST_BACKTRACE`, 
in cases when you cannot just unset the variable, 
then set it to `0` to avoid getting a backtrace. 
Any other value (even no value at all) turns on backtrace.

```text
$ export RUST_BACKTRACE=1
...
$ RUST_BACKTRACE=0 ./diverges 
thread 'main' panicked at 'This function never returns!', hello.rs:2
note: Run with `RUST_BACKTRACE=1` for a backtrace.
```

`RUST_BACKTRACE` also works with Cargo’s `run` command:

```text
$ RUST_BACKTRACE=full cargo run
     Running `target/debug/diverges`
thread 'main' panicked at 'This function never returns!', hello.rs:2
stack backtrace:
   1:     0x7f402773a829 - sys::backtrace::write::h0942de78b6c02817K8r
   2:     0x7f402773d7fc - panicking::on_panic::h3f23f9d0b5f4c91bu9w
   3:     0x7f402773960e - rt::unwind::begin_unwind_inner::h2844b8c5e81e79558Bw
   4:     0x7f4027738893 - rt::unwind::begin_unwind::h4375279447423903650
   5:     0x7f4027738809 - diverges::h2266b4c4b850236beaa
   6:     0x7f40277389e5 - main::h19bb1149c2f00ecfBaa
   7:     0x7f402773f514 - rt::unwind::try::try_fn::h13186883479104382231
   8:     0x7f402773d1d8 - __rust_try
   9:     0x7f402773f201 - rt::lang_start::ha172a3ce74bb453aK5w
  10:     0x7f4027738a19 - main
  11:     0x7f402694ab44 - __libc_start_main
  12:     0x7f40277386c8 - <unknown>
  13:                0x0 - <unknown>
```

A diverging function can be used as any type:

```rust,should_panic
# fn diverges() -> ! {
#    panic!("This function never returns!");
# }
let x: i32 = diverges();
let x: String = diverges();
```

## Function pointers

We can also create variable bindings which point to functions:

```rust
let f: fn(i32) -> i32;
```

`f` is a variable binding which points to a function that takes an `i32` as
an argument and returns an `i32`. For example:

```rust
fn plus_one(i: i32) -> i32 {
    i + 1
}

// Without type inference:
let f: fn(i32) -> i32 = plus_one;

// With type inference:
let f = plus_one;
```

We can then use `f` to call the function:

```rust
# fn plus_one(i: i32) -> i32 { i + 1 }
# let f = plus_one;
let six = f(5);
```
