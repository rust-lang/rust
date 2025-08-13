# Documentation tests

`rustdoc` supports executing your documentation examples as tests. This makes sure
that examples within your documentation are up to date and working.

The basic idea is this:

```rust,no_run
/// # Examples
///
/// ```
/// let x = 5;
/// ```
# fn f() {}
```

The triple backticks start and end code blocks. If this were in a file named `foo.rs`,
running `rustdoc --test foo.rs` will extract this example, and then run it as a test.

Please note that by default, if no language is set for the block code, rustdoc
assumes it is Rust code. So the following:

``````markdown
```rust
let x = 5;
```
``````

is strictly equivalent to:

``````markdown
```
let x = 5;
```
``````

There's some subtlety though! Read on for more details.

## Passing or failing a doctest

Like regular unit tests, regular doctests are considered to "pass"
if they compile and run without panicking.
So if you want to demonstrate that some computation gives a certain result,
the `assert!` family of macros works the same as other Rust code:

```rust
let foo = "foo";
assert_eq!(foo, "foo");
```

This way, if the computation ever returns something different,
the code panics and the doctest fails.

## Pre-processing examples

In the example above, you'll note something strange: there's no `main`
function! Forcing you to write `main` for every example, no matter how small,
adds friction and clutters the output. So `rustdoc` processes your examples
slightly before running them. Here's the full algorithm `rustdoc` uses to
preprocess examples:

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

```rust,no_run
/// ```
/// /// Some documentation.
/// # fn foo() {} // this function will be hidden
/// println!("Hello, World!");
/// ```
# fn f() {}
```

It will render like this:

```rust
/// Some documentation.
# fn foo() {}
println!("Hello, World!");
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

``````markdown
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
``````

By repeating all parts of the example, you can ensure that your example still
compiles, while only showing the parts that are relevant to that part of your
explanation.

The `#`-hiding of lines can be prevented by using two consecutive hashes
`##`. This only needs to be done with the first `#` which would've
otherwise caused hiding. If we have a string literal like the following,
which has a line that starts with a `#`:

```rust
let s = "foo
## bar # baz";
```

We can document it by escaping the initial `#`:

```text
/// let s = "foo
/// ## bar # baz";
```

Here is an example with a macro rule which matches on tokens starting with `#`:

`````rust,no_run
/// ```
/// macro_rules! ignore { (##tag) => {}; }
/// ignore! {
///     ###tag
/// }
/// ```
# fn f() {}
`````

As you can see, the rule is expecting two `#`, so when calling it, we need to add an extra `#`
because the first one is used as escape.

## Using `?` in doc tests

When writing an example, it is rarely useful to include a complete error
handling, as it would add significant amounts of boilerplate code. Instead, you
may want the following:

```rust,no_run
/// ```
/// use std::io;
/// let mut input = String::new();
/// io::stdin().read_line(&mut input)?;
/// ```
# fn f() {}
```

The problem is that `?` returns a `Result<T, E>` and test functions don't
return anything, so this will give a mismatched types error.

You can get around this limitation by manually adding a `main` that returns
`Result<T, E>`, because `Result<T, E>` implements the `Termination` trait:

```rust,no_run
/// A doc test using ?
///
/// ```
/// use std::io;
///
/// fn main() -> io::Result<()> {
///     let mut input = String::new();
///     io::stdin().read_line(&mut input)?;
///     Ok(())
/// }
/// ```
# fn f() {}
```

Together with the `# ` from the section above, you arrive at a solution that
appears to the reader as the initial idea but works with doc tests:

```rust,no_run
/// ```
/// use std::io;
/// # fn main() -> io::Result<()> {
/// let mut input = String::new();
/// io::stdin().read_line(&mut input)?;
/// # Ok(())
/// # }
/// ```
# fn f() {}
```

As of version 1.34.0, one can also omit the `fn main()`, but you will have to
disambiguate the error type:

```rust,no_run
/// ```
/// use std::io;
/// let mut input = String::new();
/// io::stdin().read_line(&mut input)?;
/// # Ok::<(), io::Error>(())
/// ```
# fn f() {}
```

This is an unfortunate consequence of the `?` operator adding an implicit
conversion, so type inference fails because the type is not unique. Please note
that you must write the `(())` in one sequence without intermediate whitespace
so that `rustdoc` understands you want an implicit `Result`-returning function.

## Showing warnings in doctests

You can show warnings in doctests by running `rustdoc --test --test-args=--show-output`
(or, if you're using cargo, `cargo test --doc -- --show-output`).
By default, this will still hide `unused` warnings, since so many examples use private functions;
you can add `#![warn(unused)]` to the top of your example if you want to see unused variables or dead code warnings.
You can also use [`#![doc(test(attr(warn(unused))))]`][test-attr] in the crate root to enable warnings globally.

[test-attr]: the-doc-attribute.md#testattr

## Documenting macros

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

Code blocks can be annotated with attributes that help `rustdoc` do the right
thing when testing your code:

The `ignore` attribute tells Rust to ignore your code. This is almost never
what you want as it's the most generic. Instead, consider annotating it
with `text` if it's not code or using `#`s to get a working example that
only shows the part you care about.

```rust
/// ```ignore
/// fn foo() {
/// ```
# fn foo() {}
```

`should_panic` tells `rustdoc` that the code should compile correctly but
panic during execution. If the code doesn't panic, the test will fail.

```rust
/// ```should_panic
/// assert!(false);
/// ```
# fn foo() {}
```

The `no_run` attribute will compile your code but not run it. This is
important for examples such as "Here's how to retrieve a web page,"
which you would want to ensure compiles, but might be run in a test
environment that has no network access. This attribute can also be
used to demonstrate code snippets that can cause Undefined Behavior.

```rust
/// ```no_run
/// loop {
///     println!("Hello, world");
/// }
/// ```
# fn foo() {}
```

`compile_fail` tells `rustdoc` that the compilation should fail. If it
compiles, then the test will fail. However, please note that code failing
with the current Rust release may work in a future release, as new features
are added.

```rust
/// ```compile_fail
/// let x = 5;
/// x += 2; // shouldn't compile!
/// ```
# fn foo() {}
```

`edition2015`, `edition2018`, `edition2021`, and `edition2024` tell `rustdoc`
that the code sample should be compiled using the respective edition of Rust.

```rust
/// Only runs on the 2018 edition.
///
/// ```edition2018
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "2".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// ```
# fn foo() {}
```

Starting in the 2024 edition[^edition-note], compatible doctests are merged as one before being
run. We combine doctests for performance reasons: the slowest part of doctests is to compile them.
Merging all of them into one file and compiling this new file, then running the doctests is much
faster. Whether doctests are merged or not, they are run in their own process.

An example of time spent when running doctests:

[sysinfo crate](https://crates.io/crates/sysinfo):

```text
wall-time duration: 4.59s
total compile time: 27.067s
total runtime: 3.969s
```

Rust core library:

```text
wall-time duration: 102s
total compile time: 775.204s
total runtime: 15.487s
```

[^edition-note]: This is based on the edition of the whole crate, not the edition of the individual
test case that may be specified in its code attribute.

In some cases, doctests cannot be merged. For example, if you have:

```rust
//! ```
//! let location = std::panic::Location::caller();
//! assert_eq!(location.line(), 4);
//! ```
```

The problem with this code is that, if you change any other doctests, it'll likely break when
running `rustdoc --test`, making it tricky to maintain.

This is where the `standalone_crate` attribute comes in: it tells `rustdoc` that a doctest
should not be merged with the others. So the previous code should use it:

```rust
//! ```standalone_crate
//! let location = std::panic::Location::caller();
//! assert_eq!(location.line(), 4);
//! ```
```

In this case, it means that the line information will not change if you add/remove other
doctests.

### Ignoring targets

Attributes starting with `ignore-` can be used to ignore doctests for specific
targets. For example, `ignore-x86_64` will avoid building doctests when the
target name contains `x86_64`.

```rust
/// ```ignore-x86_64
/// assert!(2 == 2);
/// ```
struct Foo;
```

This doctest will not be built for targets such as `x86_64-unknown-linux-gnu`.

Multiple ignore attributes can be specified to ignore multiple targets:

```rust
/// ```ignore-x86_64,ignore-windows
/// assert!(2 == 2);
/// ```
struct Foo;
```

If you want to preserve backwards compatibility for older versions of rustdoc,
you can specify both `ignore` and `ignore-`, such as:

```rust
/// ```ignore,ignore-x86_64
/// assert!(2 == 2);
/// ```
struct Foo;
```

In older versions, this will be ignored on all targets, but starting with
version 1.88.0, `ignore-x86_64` will override `ignore`.

### Custom CSS classes for code blocks

```rust
/// ```custom,{class=language-c}
/// int main(void) { return 0; }
/// ```
pub struct Bar;
```

The text `int main(void) { return 0; }` is rendered without highlighting in a code block
with the class `language-c`. This can be used to highlight other languages through JavaScript
libraries for example.

Without the `custom` attribute, it would be generated as a Rust code example with an additional
`language-C` CSS class. Therefore, if you specifically don't want it to be a Rust code example,
don't forget to add the `custom` attribute.

To be noted that you can replace `class=` with `.` to achieve the same result:

```rust
/// ```custom,{.language-c}
/// int main(void) { return 0; }
/// ```
pub struct Bar;
```

To be noted, `rust` and `.rust`/`class=rust` have different effects: `rust` indicates that this is
a Rust code block whereas the two others add a "rust" CSS class on the code block.

You can also use double quotes:

```rust
/// ```"not rust" {."hello everyone"}
/// int main(void) { return 0; }
/// ```
pub struct Bar;
```

## Syntax reference

The *exact* syntax for code blocks, including the edge cases, can be found
in the [Fenced Code Blocks](https://spec.commonmark.org/0.29/#fenced-code-blocks)
section of the CommonMark specification.

Rustdoc also accepts *indented* code blocks as an alternative to fenced
code blocks: instead of surrounding your code with three backticks, you
can indent each line by four or more spaces.

``````markdown
    let foo = "foo";
    assert_eq!(foo, "foo");
``````

These, too, are documented in the CommonMark specification, in the
[Indented Code Blocks](https://spec.commonmark.org/0.29/#indented-code-blocks)
section.

However, it's preferable to use fenced code blocks over indented code blocks.
Not only are fenced code blocks considered more idiomatic for Rust code,
but there is no way to use attributes such as `ignore` or `should_panic` with
indented code blocks.

### Include items only when collecting doctests

Rustdoc's documentation tests can do some things that regular unit tests can't, so it can
sometimes be useful to extend your doctests with samples that wouldn't otherwise need to be in
documentation. To this end, Rustdoc allows you to have certain items only appear when it's
collecting doctests, so you can utilize doctest functionality without forcing the test to appear in
docs, or to find an arbitrary private item to include it on.

When compiling a crate for use in doctests (with `--test` option), `rustdoc` will set `#[cfg(doctest)]`.
Note that they will still link against only the public items of your crate; if you need to test
private items, you need to write a unit test.

In this example, we're adding doctests that we know won't compile, to verify that our struct can
only take in valid data:

```rust
/// We have a struct here. Remember it doesn't accept negative numbers!
pub struct MyStruct(pub usize);

/// ```compile_fail
/// let x = my_crate::MyStruct(-5);
/// ```
#[cfg(doctest)]
pub struct MyStructOnlyTakesUsize;
```

Note that the struct `MyStructOnlyTakesUsize` here isn't actually part of your public crate
API. The use of `#[cfg(doctest)]` makes sure that this struct only exists while `rustdoc` is
collecting doctests. This means that its doctest is executed when `--test` is passed to rustdoc,
but is hidden from the public documentation.

Another possible use of `#[cfg(doctest)]` is to test doctests that are included in your README file
without including it in your main documentation. For example, you could write this into your
`lib.rs` to test your README as part of your doctests:

```rust,no_run
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
```

This will include your README as documentation on the hidden struct `ReadmeDoctests`, which will
then be tested alongside the rest of your doctests.

## Controlling the compilation and run directories

By default, `rustdoc --test` will compile and run documentation test examples
from the same working directory.
The compilation directory is being used for compiler diagnostics, the `file!()` macro and
the output of `rustdoc` test runner itself, whereas the run directory has an influence on file-system
operations within documentation test examples, such as `std::fs::read_to_string`.

The `--test-run-directory` flag allows controlling the run directory separately from the compilation directory.
This is particularly useful in workspaces, where compiler invocations and thus diagnostics should be
relative to the workspace directory, but documentation test examples should run relative to the crate directory.
