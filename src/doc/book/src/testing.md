# Testing

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

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
```

For now, let's remove the `mod` bit, and focus on just the function:

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
#[test]
fn it_works() {
}
```

Note the `#[test]`. This attribute indicates that this is a test function. It
currently has no body. That's good enough to pass! We can run the tests with
`cargo test`:

```bash
$ cargo test
   Compiling adder v0.1.0 (file:///home/you/projects/adder)
    Finished debug [unoptimized + debuginfo] target(s) in 0.15 secs
     Running target/debug/deps/adder-941f01916ca4a642

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
# fn main() {
fn it_works() {
}
# }
```

We also get a summary line:

```text
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

So why does our do-nothing test pass? Any test which doesn't `panic!` passes,
and any test that does `panic!` fails. Let's make our test fail:

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
#[test]
fn it_works() {
    assert!(false);
}
```

`assert!` is a macro provided by Rust which takes one argument: if the argument
is `true`, nothing happens. If the argument is `false`, it will `panic!`. Let's
run our tests again:

```bash
$ cargo test
   Compiling adder v0.1.0 (file:///home/you/projects/adder)
    Finished debug [unoptimized + debuginfo] target(s) in 0.17 secs
     Running target/debug/deps/adder-941f01916ca4a642

running 1 test
test it_works ... FAILED

failures:

---- it_works stdout ----
        thread 'it_works' panicked at 'assertion failed: false', src/lib.rs:5
note: Run with `RUST_BACKTRACE=1` for a backtrace.


failures:
    it_works

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured

error: test failed
```

Rust indicates that our test failed:

```text
test it_works ... FAILED
```

And that's reflected in the summary line:

```text
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured
```

We also get a non-zero status code. We can use `$?` on macOS and Linux:

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

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
#[test]
#[should_panic]
fn it_works() {
    assert!(false);
}
```

This test will now succeed if we `panic!` and fail if we complete. Let's try it:

```bash
$ cargo test
   Compiling adder v0.1.0 (file:///home/you/projects/adder)
    Finished debug [unoptimized + debuginfo] target(s) in 0.17 secs
     Running target/debug/deps/adder-941f01916ca4a642

running 1 test
test it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

Rust provides another macro, `assert_eq!`, that compares two arguments for
equality:

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
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
   Compiling adder v0.1.0 (file:///home/you/projects/adder)
    Finished debug [unoptimized + debuginfo] target(s) in 0.21 secs
     Running target/debug/deps/adder-941f01916ca4a642

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

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
#[test]
#[should_panic(expected = "assertion failed")]
fn it_works() {
    assert_eq!("Hello", "world");
}
```

That's all there is to the basics! Let's write one 'real' test:

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
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

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[test]
fn it_works() {
    assert_eq!(4, add_two(2));
}

#[test]
#[ignore]
fn expensive_test() {
    // Code that takes an hour to run...
}
```

Now we run our tests and see that `it_works` is run, but `expensive_test` is
not:

```bash
$ cargo test
   Compiling adder v0.1.0 (file:///home/you/projects/adder)
    Finished debug [unoptimized + debuginfo] target(s) in 0.20 secs
     Running target/debug/deps/adder-941f01916ca4a642

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
    Finished debug [unoptimized + debuginfo] target(s) in 0.0 secs
     Running target/debug/deps/adder-941f01916ca4a642

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
missing the `tests` module. You might have noticed this test module was
present in the code that was initially generated with `cargo new` but
was missing from our last example. Let's explain what this does.

The idiomatic way of writing our example looks like this:

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
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
we need to bring the tested function into scope. This can be annoying if you have
a large module, and so this is a common use of globs. Let's change our
`src/lib.rs` to make use of it:

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
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
   Compiling adder v0.1.0 (file:///home/you/projects/adder)
     Running target/debug/deps/adder-91b3e234d4ed382a

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

Each file in `tests/*.rs` directory is treated as an individual crate.
To write an integration test, let's make a `tests` directory and
put a `tests/integration_test.rs` file inside with this as its contents:

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
# // Sadly, this code will not work in play.rust-lang.org, because we have no
# // crate adder to import. You'll need to try this part on your own machine.
extern crate adder;

#[test]
fn it_works() {
    assert_eq!(4, adder::add_two(2));
}
```

This looks similar to our previous tests, but slightly different. We now have
an `extern crate adder` at the top. This is because each test in the `tests`
directory is an entirely separate crate, and so we need to import our library.
This is also why `tests` is a suitable place to write integration-style tests:
they use the library like any other consumer of it would.

Let's run them:

```bash
$ cargo test
   Compiling adder v0.1.0 (file:///home/you/projects/adder)
     Running target/debug/deps/adder-91b3e234d4ed382a

running 1 test
test tests::it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

     Running target/debug/integration_test-68064b69521c828a

running 1 test
test it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

Now we have three sections: our previous test is also run, as well as our new
one.

Cargo will ignore files in subdirectories of the `tests/` directory.
Therefore shared modules in integrations tests are possible.
For example `tests/common/mod.rs` is not separately compiled by cargo but can
be imported in every test with `mod common;`

That's all there is to the `tests` directory. The `tests` module isn't needed
here, since the whole thing is focused on tests.

Note, when building integration tests, cargo will not pass the `test` attribute
to the compiler. It means that all parts in `cfg(test)` won't be included in
the build used in your integration tests.

Let's finally check out that third section: documentation tests.

# Documentation tests

Nothing is better than documentation with examples. Nothing is worse than
examples that don't actually work, because the code has changed since the
documentation has been written. To this end, Rust supports automatically
running examples in your documentation (**note:** this only works in library
crates, not binary crates). Here's a fleshed-out `src/lib.rs` with examples:

```rust,ignore
# // The next line exists to trick play.rust-lang.org into running our code as a
# // test:
# // fn main
#
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
   Compiling adder v0.1.0. (file:///home/you/projects/adder)
     Running target/debug/deps/adder-91b3e234d4ed382a

running 1 test
test tests::it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

     Running target/debug/integration_test-68064b69521c828a

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

# Testing and concurrency

It is important to note that tests are run concurrently using threads. For this
reason, care should be taken to ensure your tests do not depend on each-other,
or on any shared state. "Shared state" can also include the environment, such
as the current working directory, or environment variables.

If this is an issue it is possible to control this concurrency, either by
setting the environment variable `RUST_TEST_THREADS`, or by passing the argument
`--test-threads` to the tests:

```bash
$ RUST_TEST_THREADS=1 cargo test   # Run tests with no concurrency
...
$ cargo test -- --test-threads=1   # Same as above
...
```

# Test output

By default Rust's test library captures and discards output to standard
out/error, e.g. output from `println!()`. This too can be controlled using the
environment or a switch:


```bash
$ RUST_TEST_NOCAPTURE=1 cargo test   # Preserve stdout/stderr
...
$ cargo test -- --nocapture          # Same as above
...
```

However a better method avoiding capture is to use logging rather than raw
output. Rust has a [standard logging API][log], which provides a frontend to
multiple logging implementations. This can be used in conjunction with the
default [env_logger] to output any debugging information in a manner that can be
controlled at runtime.

[log]: https://crates.io/crates/log
[env_logger]: https://crates.io/crates/env_logger
