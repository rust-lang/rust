% Testing

> Program testing can be a very effective way to show the presence of bugs, but
> it is hopelessly inadequate for showing their absence.
>
> Edsger W. Dijkstra, "The Humble Programmer" (1972)

Let's talk about how to test Rust code. What we will not be talking about is
the right way to test Rust code. There are many schools of thought regarding
the right and wrong way to write tests. All of these approaches use the same
basic tools, and so we'll show you the syntax for using them.

# The `test` attribute

At its simplest, a test in Rust is a function that's annotated with the `test`
attribute. Let's make a new project with Cargo called `adder`:

```bash
$ cargo new adder
$ cd adder
```

Cargo will automatically generate a simple test when you make a new project.
Here's the contents of `src/lib.rs`:

```rust
#[test]
fn it_works() {
}
```

Note the `#[test]`. This attribute indicates that this is a test function. It
currently has no body. That's good enough to pass! We can run the tests with
`cargo test`:

```bash
$ cargo test
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

Cargo compiled and ran our tests. There are two sets of output here: one
for the test we wrote, and another for documentation tests. We'll talk about
those later. For now, see this line:

```text
test it_works ... ok
```

Note the `it_works`. This comes from the name of our function:

```rust
fn it_works() {
# }
```

We also get a summary line:

```text
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

So why does our do-nothing test pass? Any test which doesn't `panic!` passes,
and any test that does `panic!` fails. Let's make our test fail:

```rust
#[test]
fn it_works() {
    assert!(false);
}
```

`assert!` is a macro provided by Rust which takes one argument: if the argument
is `true`, nothing happens. If the argument is `false`, it `panic!`s. Let's run
our tests again:

```bash
$ cargo test
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test it_works ... FAILED

failures:

---- it_works stdout ----
        thread 'it_works' panicked at 'assertion failed: false', /home/steve/tmp/adder/src/lib.rs:3



failures:
    it_works

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured

thread '<main>' panicked at 'Some tests failed', /home/steve/src/rust/src/libtest/lib.rs:247
```

Rust indicates that our test failed:

```text
test it_works ... FAILED
```

And that's reflected in the summary line:

```text
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured
```

We also get a non-zero status code. We can use `$?` on OS X and Linux:

```bash
$ echo $?
101
```

On Windows, if you’re using `cmd`:

```bash
> echo %ERRORLEVEL%
```

And if you’re using PowerShell:

```bash
> echo $LASTEXITCODE # the code itself
> echo $? # a boolean, fail or succeed
```

This is useful if you want to integrate `cargo test` into other tooling.

We can invert our test's failure with another attribute: `should_panic`:

```rust
#[test]
#[should_panic]
fn it_works() {
    assert!(false);
}
```

This test will now succeed if we `panic!` and fail if we complete. Let's try it:

```bash
$ cargo test
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

Rust provides another macro, `assert_eq!`, that compares two arguments for
equality:

```rust
#[test]
#[should_panic]
fn it_works() {
    assert_eq!("Hello", "world");
}
```

Does this test pass or fail? Because of the `should_panic` attribute, it
passes:

```bash
$ cargo test
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

`should_panic` tests can be fragile, as it's hard to guarantee that the test
didn't fail for an unexpected reason. To help with this, an optional `expected`
parameter can be added to the `should_panic` attribute. The test harness will
make sure that the failure message contains the provided text. A safer version
of the example above would be:

```rust
#[test]
#[should_panic(expected = "assertion failed")]
fn it_works() {
    assert_eq!("Hello", "world");
}
```

That's all there is to the basics! Let's write one 'real' test:

```rust,ignore
pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[test]
fn it_works() {
    assert_eq!(4, add_two(2));
}
```

This is a very common use of `assert_eq!`: call some function with
some known arguments and compare it to the expected output.

# The `ignore` attribute

Sometimes a few specific tests can be very time-consuming to execute. These
can be disabled by default by using the `ignore` attribute:

```rust
#[test]
fn it_works() {
    assert_eq!(4, add_two(2));
}

#[test]
#[ignore]
fn expensive_test() {
    // code that takes an hour to run
}
```

Now we run our tests and see that `it_works` is run, but `expensive_test` is
not:

```bash
$ cargo test
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 2 tests
test expensive_test ... ignored
test it_works ... ok

test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

The expensive tests can be run explicitly using `cargo test -- --ignored`:

```bash
$ cargo test -- --ignored
     Running target/adder-91b3e234d4ed382a

running 1 test
test expensive_test ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

The `--ignored` argument is an argument to the test binary, and not to Cargo,
which is why the command is `cargo test -- --ignored`.

# The `tests` module

There is one way in which our existing example is not idiomatic: it's
missing the `tests` module. The idiomatic way of writing our example
looks like this:

```rust,ignore
pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[cfg(test)]
mod tests {
    use super::add_two;

    #[test]
    fn it_works() {
        assert_eq!(4, add_two(2));
    }
}
```

There's a few changes here. The first is the introduction of a `mod tests` with
a `cfg` attribute. The module allows us to group all of our tests together, and
to also define helper functions if needed, that don't become a part of the rest
of our crate. The `cfg` attribute only compiles our test code if we're
currently trying to run the tests. This can save compile time, and also ensures
that our tests are entirely left out of a normal build.

The second change is the `use` declaration. Because we're in an inner module,
we need to bring our test function into scope. This can be annoying if you have
a large module, and so this is a common use of globs. Let's change our
`src/lib.rs` to make use of it:

```rust,ignore
pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(4, add_two(2));
    }
}
```

Note the different `use` line. Now we run our tests:

```bash
$ cargo test
    Updating registry `https://github.com/rust-lang/crates.io-index`
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test tests::it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

It works!

The current convention is to use the `tests` module to hold your "unit-style"
tests. Anything that tests one small bit of functionality makes sense to
go here. But what about "integration-style" tests instead? For that, we have
the `tests` directory.

# The `tests` directory

To write an integration test, let's make a `tests` directory, and
put a `tests/lib.rs` file inside, with this as its contents:

```rust,ignore
extern crate adder;

#[test]
fn it_works() {
    assert_eq!(4, adder::add_two(2));
}
```

This looks similar to our previous tests, but slightly different. We now have
an `extern crate adder` at the top. This is because the tests in the `tests`
directory are an entirely separate crate, and so we need to import our library.
This is also why `tests` is a suitable place to write integration-style tests:
they use the library like any other consumer of it would.

Let's run them:

```bash
$ cargo test
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test tests::it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

     Running target/lib-c18e7d3494509e74

running 1 test
test it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

Now we have three sections: our previous test is also run, as well as our new
one.

That's all there is to the `tests` directory. The `tests` module isn't needed
here, since the whole thing is focused on tests.

Let's finally check out that third section: documentation tests.

# Documentation tests

Nothing is better than documentation with examples. Nothing is worse than
examples that don't actually work, because the code has changed since the
documentation has been written. To this end, Rust supports automatically
running examples in your documentation (**note:** this only works in library
crates, not binary crates). Here's a fleshed-out `src/lib.rs` with examples:

```rust,ignore
//! The `adder` crate provides functions that add numbers to other numbers.
//!
//! # Examples
//!
//! ```
//! assert_eq!(4, adder::add_two(2));
//! ```

/// This function adds two to its argument.
///
/// # Examples
///
/// ```
/// use adder::add_two;
///
/// assert_eq!(4, add_two(2));
/// ```
pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(4, add_two(2));
    }
}
```

Note the module-level documentation with `//!` and the function-level
documentation with `///`. Rust's documentation supports Markdown in comments,
and so triple graves mark code blocks. It is conventional to include the
`# Examples` section, exactly like that, with examples following.

Let's run the tests again:

```bash
$ cargo test
   Compiling adder v0.0.1 (file:///home/steve/tmp/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test tests::it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

     Running target/lib-c18e7d3494509e74

running 1 test
test it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 2 tests
test add_two_0 ... ok
test _0 ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured
```

Now we have all three kinds of tests running! Note the names of the
documentation tests: the `_0` is generated for the module test, and `add_two_0`
for the function test. These will auto increment with names like `add_two_1` as
you add more examples.

We haven’t covered all of the details with writing documentation tests. For more,
please see the [Documentation chapter](documentation.html).

One final note: documentation tests *cannot* be run on binary crates.
To see more on file arrangement see the [Crates and
Modules](crates-and-modules.html) section.
