% Functions

You've already seen one function so far, the `main` function:

```rust
fn main() {
}
```

This is the simplest possible function declaration. As we mentioned before,
`fn` says "this is a function," followed by the name, some parentheses because
this function takes no arguments, and then some curly braces to indicate the
body. Here's a function named `foo`:

```rust
fn foo() {
}
```

So, what about taking arguments? Here's a function that prints a number:

```rust
fn print_number(x: i32) {
    println!("x is: {}", x);
}
```

Here's a complete program that uses `print_number`:

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

Here's a complete program that adds two numbers together and prints them:

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

```{rust,ignore}
fn print_sum(x, y) {
    println!("sum is: {}", x + y);
}
```

You get this error:

```text
hello.rs:5:18: 5:19 expected one of `!`, `:`, or `@`, found `)`
hello.rs:5 fn print_number(x, y) {
```

This is a deliberate design decision. While full-program inference is possible,
languages which have it, like Haskell, often suggest that documenting your
types explicitly is a best-practice. We agree that forcing functions to declare
types while allowing for inference inside of function bodies is a wonderful
sweet spot between full inference and no inference.

What about returning a value? Here's a function that adds one to an integer:

```rust
fn add_one(x: i32) -> i32 {
    x + 1
}
```

Rust functions return exactly one value, and you declare the type after an
"arrow," which is a dash (`-`) followed by a greater-than sign (`>`).

You'll note the lack of a semicolon here. If we added it in:

```{rust,ignore}
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

Remember our earlier discussions about semicolons and `()`? Our function claims
to return an `i32`, but with a semicolon, it would return `()` instead. Rust
realizes this probably isn't what we want, and suggests removing the semicolon.

This is very much like our `if` statement before: the result of the block
(`{}`) is the value of the expression. Other expression-oriented languages,
such as Ruby, work like this, but it's a bit unusual in the systems programming
world. When people first learn about this, they usually assume that it
introduces bugs. But because Rust's type system is so strong, and because unit
is its own unique type, we have never seen an issue where adding or removing a
semicolon in a return position would cause a bug.

But what about early returns? Rust does have a keyword for that, `return`:

```rust
fn foo(x: i32) -> i32 {
    if x < 5 { return x; }

    x + 1
}
```

Using a `return` as the last line of a function works, but is considered poor
style:

```rust
fn foo(x: i32) -> i32 {
    if x < 5 { return x; }

    return x + 1;
}
```

The previous definition without `return` may look a bit strange if you haven't
worked in an expression-based language before, but it becomes intuitive over
time. If this were production code, we wouldn't write it in that way anyway,
we'd write this:

```rust
fn foo(x: i32) -> i32 {
    if x < 5 {
        x
    } else {
        x + 1
    }
}
```

Because `if` is an expression, and it's the only expression in this function,
the value will be the result of the `if`.

## Diverging functions

Rust has some special syntax for 'diverging functions', which are functions that
do not return:

```
fn diverges() -> ! {
    panic!("This function never returns!");
}
```

`panic!` is a macro, similar to `println!()` that we've already seen. Unlike
`println!()`, `panic!()` causes the current thread of execution to crash with
the given message.

Because this function will cause a crash, it will never return, and so it has
the type '`!`', which is read "diverges." A diverging function can be used
as any type:

```should_panic
# fn diverges() -> ! {
#    panic!("This function never returns!");
# }

let x: i32 = diverges();
let x: String = diverges();
```

We don't have a good use for diverging functions yet, because they're used in
conjunction with other Rust features. But when you see `-> !` later, you'll
know what it's called.
