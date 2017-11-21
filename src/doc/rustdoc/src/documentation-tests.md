# Documentation tests

`rustdoc` supports executing your documentation examples as tests. This makes sure
that your tests are up to date and working.

The basic idea is this:

```ignore
/// # Examples
///
/// ```
/// let x = 5;
/// ```
```

The triple backticks start and end code blocks. If this were in a file named `foo.rs`,
running `rustdoc --test foo.rs` will extract this example, and then run it as a test.

Please note that by default, if no language is set for the block code, `rustdoc`
assumes it is `Rust` code. So the following:

```rust
let x = 5;
```

is strictly equivalent to:

```
let x = 5;
```

There's some subtlety though! Read on for more details.

## Pre-processing examples

In the example above, you'll note something strange: there's no `main`
function! Forcing you to write `main` for every example, no matter how small,
adds friction. So `rustdoc` processes your examples slightly before
running them. Here's the full algorithm rustdoc uses to preprocess examples:

1. Some common `allow` attributes are inserted, including
   `unused_variables`, `unused_assignments`, `unused_mut`,
   `unused_attributes`, and `dead_code`. Small examples often trigger
   these lints.
2. Any attributes specified with `#![doc(test(attr(...)))]` are added.
3. Any leading `#![foo]` attributes are left intact as crate attributes.
4. If the example does not contain `extern crate`, and
   `#![doc(test(no_crate_inject))]` was not specified, then `extern crate
   <mycrate>;` is inserted (note the lack of `#[macro_use]`).
5. Finally, if the example does not contain `fn main`, the remainder of the
   text is wrapped in `fn main() { your_code }`.

For more about that caveat in rule 4, see "Documenting Macros" below.

## Hiding portions of the example

Sometimes, you need some setup code, or other things that would distract
from your example, but are important to make the tests work. Consider
an example block that looks like this:

```text
/// Some documentation.
# fn foo() {}
```

It will render like this:

```rust
/// Some documentation.
# fn foo() {}
```

Yes, that's right: you can add lines that start with `# `, and they will
be hidden from the output, but will be used when compiling your code. You
can use this to your advantage. In this case, documentation comments need
to apply to some kind of function, so if I want to show you just a
documentation comment, I need to add a little function definition below
it. At the same time, it's only there to satisfy the compiler, so hiding
it makes the example more clear. You can use this technique to explain
longer examples in detail, while still preserving the testability of your
documentation.

For example, imagine that we wanted to document this code:

```rust
let x = 5;
let y = 6;
println!("{}", x + y);
```

We might want the documentation to end up looking like this:

> First, we set `x` to five:
>
> ```rust
> let x = 5;
> # let y = 6;
> # println!("{}", x + y);
> ```
>
> Next, we set `y` to six:
>
> ```rust
> # let x = 5;
> let y = 6;
> # println!("{}", x + y);
> ```
>
> Finally, we print the sum of `x` and `y`:
>
> ```rust
> # let x = 5;
> # let y = 6;
> println!("{}", x + y);
> ```

To keep each code block testable, we want the whole program in each block, but
we don't want the reader to see every line every time.  Here's what we put in
our source code:

```text
    First, we set `x` to five:

    ```
    let x = 5;
    # let y = 6;
    # println!("{}", x + y);
    ```

    Next, we set `y` to six:

    ```
    # let x = 5;
    let y = 6;
    # println!("{}", x + y);
    ```

    Finally, we print the sum of `x` and `y`:

    ```
    # let x = 5;
    # let y = 6;
    println!("{}", x + y);
    ```
```

By repeating all parts of the example, you can ensure that your example still
compiles, while only showing the parts that are relevant to that part of your
explanation.

Another case where the use of `#` is handy is when you want to ignore
error handling. Lets say you want the following,

```ignore
/// use std::io;
/// let mut input = String::new();
/// io::stdin().read_line(&mut input)?;
```

The problem is that `?` returns a `Result<T, E>` and test functions
don't return anything so this will give a mismatched types error.

```ignore
/// A doc test using ?
///
/// ```
/// use std::io;
/// # fn foo() -> io::Result<()> {
/// let mut input = String::new();
/// io::stdin().read_line(&mut input)?;
/// # Ok(())
/// # }
/// ```
# fn foo() {}
```

You can get around this by wrapping the code in a function. This catches
and swallows the `Result<T, E>` when running tests on the docs. This
pattern appears regularly in the standard library.

### Documenting macros

Here’s an example of documenting a macro:

```rust
/// Panic with a given message unless an expression evaluates to true.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate foo;
/// # fn main() {
/// panic_unless!(1 + 1 == 2, “Math is broken.”);
/// # }
/// ```
///
/// ```should_panic
/// # #[macro_use] extern crate foo;
/// # fn main() {
/// panic_unless!(true == false, “I’m broken.”);
/// # }
/// ```
#[macro_export]
macro_rules! panic_unless {
    ($condition:expr, $($rest:expr),+) => ({ if ! $condition { panic!($($rest),+); } });
}
# fn main() {}
```

You’ll note three things: we need to add our own `extern crate` line, so that
we can add the `#[macro_use]` attribute. Second, we’ll need to add our own
`main()` as well (for reasons discussed above). Finally, a judicious use of
`#` to comment out those two things, so they don’t show up in the output.

## Attributes

There are a few annotations that are useful to help `rustdoc` do the right
thing when testing your code:

```rust
/// ```ignore
/// fn foo() {
/// ```
# fn foo() {}
```

The `ignore` directive tells Rust to ignore your code. This is almost never
what you want, as it's the most generic. Instead, consider annotating it
with `text` if it's not code, or using `#`s to get a working example that
only shows the part you care about.

```rust
/// ```should_panic
/// assert!(false);
/// ```
# fn foo() {}
```

`should_panic` tells `rustdoc` that the code should compile correctly, but
not actually pass as a test.

```text
/// ```no_run
/// loop {
///     println!("Hello, world");
/// }
/// ```
# fn foo() {}
```

`compile_fail` tells `rustdoc` that the compilation should fail. If it
compiles, then the test will fail. However please note that code failing
with the current Rust release may work in a future release, as new features
are added.

```text
/// ```compile_fail
/// let x = 5;
/// x += 2; // shouldn't compile!
/// ```
```

The `no_run` attribute will compile your code, but not run it. This is
important for examples such as "Here's how to retrieve a web page,"
which you would want to ensure compiles, but might be run in a test
environment that has no network access.
