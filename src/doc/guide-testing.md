% The Rust Testing Guide

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
is `true`, nothing happens. If the argument is false, it `panic!`s. Let's run
our tests again:

```bash
$ cargo test
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test it_works ... FAILED

failures:

---- it_works stdout ----
        task 'it_works' panicked at 'assertion failed: false', /home/steve/tmp/adder/src/lib.rs:3



failures:
    it_works

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured

task '<main>' panicked at 'Some tests failed', /home/steve/src/rust/src/libtest/lib.rs:247
```

Rust indicates that our test failed:

```text
test it_works ... FAILED
```

And that's reflected in the summary line:

```text
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured
```

We also get a non-zero status code:

```bash
$ echo $?
101
```

This is useful if you want to integrate `cargo test` into other tooling.

We can invert our test's failure with another attribute: `should_fail`:

```rust
#[test]
#[should_fail]
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
#[should_fail]
fn it_works() {
    assert_eq!("Hello", "world");
}
```

Does this test pass or fail? Because of the `should_fail` attribute, it
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

`should_fail` tests can be fragile, as it's hard to guarantee that the test
didn't fail for an unexpected reason. To help with this, an optional `expected`
parameter can be added to the `should_fail` attribute. The test harness will
make sure that the failure message contains the provided text. A safer version
of the example above would be:

```
#[test]
#[should_fail(expected = "assertion failed")]
fn it_works() {
    assert_eq!("Hello", "world");
}
```

That's all there is to the basics! Let's write one 'real' test:

```{rust,ignore}
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

# The `test` module

There is one way in which our existing example is not idiomatic: it's
missing the test module. The idiomatic way of writing our example
looks like this:

```{rust,ignore}
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
a large module, and so this is a common use of the `glob` feature. Let's change
our `src/lib.rs` to make use of it:

```{rust,ignore}
#![feature(globs)]

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

Note the `feature` attribute, as well as the different `use` line. Now we run
our tests:

```bash
$ cargo test
    Updating registry `https://github.com/rust-lang/crates.io-index`
   Compiling adder v0.0.1 (file:///home/you/projects/adder)
     Running target/adder-91b3e234d4ed382a

running 1 test
test test::it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

It works!

The current convention is to use the `test` module to hold your "unit"-style
tests. Anything that just tests one small bit of functionality makes sense to
go here. But what about "integration"-style tests instead? For that, we have
the `tests` directory

# The `tests` directory

To write an integration test, let's make a `tests` directory, and
put a `tests/lib.rs` file inside, with this as its contents:

```{rust,ignore}
extern crate adder;

#[test]
fn it_works() {
    assert_eq(4, adder::add_two(2));
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
test test::it_works ... ok

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

That's all there is to the `tests` directory. The `test` module isn't needed
here, since the whole thing is focused on tests.

Let's finally check out that third section: documentation tests.

# Documentation tests

Nothing is better than documentation with examples. Nothing is worse than
examples that don't actually work, because the code has changed since the
documentation has been written. To this end, Rust supports automatically
running examples in your documentation. Here's a fleshed-out `src/lib.rs`
with examples:

```{rust,ignore}
//! The `adder` crate provides functions that add numbers to other numbers.
//!
//! # Examples
//!
//! ```
//! assert_eq!(4, adder::add_two(2));
//! ```

#![feature(globs)]

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
test test::it_works ... ok

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

# Benchmark tests

Rust also supports benchmark tests, which can test the performance of your
code. Let's make our `src/lib.rs` look like this (comments elided):

```{rust,ignore}
#![feature(globs)]

extern crate test;

pub fn add_two(a: i32) -> i32 {
    a + 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn it_works() {
        assert_eq!(4, add_two(2));
    }

    #[bench]
    fn bench_add_two(b: &mut Bencher) {
        b.iter(|| add_two(2));
    }
}
```

We've imported the `test` crate, which contains our benchmarking support.
We have a new function as well, with the `bench` attribute. Unlike regular
tests, which take no arguments, benchmark tests take a `&mut Bencher`. This
`Bencher` provides an `iter` method, which takes a closure. This closure
contains the code we'd like to benchmark.

We can run benchmark tests with `cargo bench`:

```bash
$ cargo bench
   Compiling adder v0.0.1 (file:///home/steve/tmp/adder)
     Running target/release/adder-91b3e234d4ed382a

running 2 tests
test tests::it_works ... ignored
test tests::bench_add_two ... bench:         1 ns/iter (+/- 0)

test result: ok. 0 passed; 0 failed; 1 ignored; 1 measured
```

Our non-benchmark test was ignored. You may have noticed that `cargo bench`
takes a bit longer than `cargo test`. This is because Rust runs our benchmark
a number of times, and then takes the average. Because we're doing so little
work in this example, we have a `1 ns/iter (+/- 0)`, but this would show
the variance if there was one.

Advice on writing benchmarks:


* Move setup code outside the `iter` loop; only put the part you want to measure inside
* Make the code do "the same thing" on each iteration; do not accumulate or change state
* Make the outer function idempotent too; the benchmark runner is likely to run
  it many times
*  Make the inner `iter` loop short and fast so benchmark runs are fast and the
   calibrator can adjust the run-length at fine resolution
* Make the code in the `iter` loop do something simple, to assist in pinpointing
  performance improvements (or regressions)

## Gotcha: optimizations

There's another tricky part to writing benchmarks: benchmarks compiled with
optimizations activated can be dramatically changed by the optimizer so that
the benchmark is no longer benchmarking what one expects. For example, the
compiler might recognize that some calculation has no external effects and
remove it entirely.

```{rust,ignore}
extern crate test;
use test::Bencher;

#[bench]
fn bench_xor_1000_ints(b: &mut Bencher) {
    b.iter(|| {
        range(0u, 1000).fold(0, |old, new| old ^ new);
    });
}
```

gives the following results

```text
running 1 test
test bench_xor_1000_ints ... bench:         0 ns/iter (+/- 0)

test result: ok. 0 passed; 0 failed; 0 ignored; 1 measured
```

The benchmarking runner offers two ways to avoid this. Either, the closure that
the `iter` method receives can return an arbitrary value which forces the
optimizer to consider the result used and ensures it cannot remove the
computation entirely. This could be done for the example above by adjusting the
`b.iter` call to

```rust
# struct X;
# impl X { fn iter<T, F>(&self, _: F) where F: FnMut() -> T {} } let b = X;
b.iter(|| {
    // note lack of `;` (could also use an explicit `return`).
    range(0u, 1000).fold(0, |old, new| old ^ new)
});
```

Or, the other option is to call the generic `test::black_box` function, which
is an opaque "black box" to the optimizer and so forces it to consider any
argument as used.

```rust
extern crate test;

# fn main() {
# struct X;
# impl X { fn iter<T, F>(&self, _: F) where F: FnMut() -> T {} } let b = X;
b.iter(|| {
    let mut n = 1000_u32;

    test::black_box(&mut n); // pretend to modify `n`

    range(0, n).fold(0, |a, b| a ^ b)
})
# }
```

Neither of these read or modify the value, and are very cheap for small values.
Larger values can be passed indirectly to reduce overhead (e.g.
`black_box(&huge_struct)`).

Performing either of the above changes gives the following benchmarking results

```text
running 1 test
test bench_xor_1000_ints ... bench:       1 ns/iter (+/- 0)

test result: ok. 0 passed; 0 failed; 0 ignored; 1 measured
```

However, the optimizer can still modify a testcase in an undesirable manner
even when using either of the above.
