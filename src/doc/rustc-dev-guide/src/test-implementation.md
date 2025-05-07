# The `#[test]` attribute

<!-- toc -->



Many Rust programmers rely on a built-in attribute called `#[test]`. All
you have to do is mark a function and include some asserts like so:


```rust,ignore
#[test]
fn my_test() {
    assert!(2+2 == 4);
}
```

When this program is compiled using `rustc --test` or `cargo test`, it will
produce an executable that can run this, and any other test function. This
method of testing allows tests to live alongside code in an organic way. You
can even put tests inside private modules:

```rust,ignore
mod my_priv_mod {
    fn my_priv_func() -> bool {}

    #[test]
    fn test_priv_func() {
        assert!(my_priv_func());
    }
}
```

Private items can thus be easily tested without worrying about how to expose
them to any sort of external testing apparatus. This is key to the
ergonomics of testing in Rust. Semantically, however, it's rather odd.
How does any sort of `main` function invoke these tests if they're not visible?
What exactly is `rustc --test` doing?

`#[test]` is implemented as a syntactic transformation inside the compiler's
[`rustc_ast`][rustc_ast]. Essentially, it's a fancy [`macro`] that
rewrites the crate in 3 steps:

## Step 1: Re-Exporting

As mentioned earlier, tests can exist inside private modules, so we need a
way of exposing them to the main function, without breaking any existing
code. To that end, [`rustc_ast`][rustc_ast] will create local modules called
`__test_reexports` that recursively reexport tests. This expansion translates
the above example into:

```rust,ignore
mod my_priv_mod {
    fn my_priv_func() -> bool {}

    pub fn test_priv_func() {
        assert!(my_priv_func());
    }

    pub mod __test_reexports {
        pub use super::test_priv_func;
    }
}
```

Now, our test can be accessed as
`my_priv_mod::__test_reexports::test_priv_func`. For deeper module
structures, `__test_reexports` will reexport modules that contain tests, so a
test at `a::b::my_test` becomes
`a::__test_reexports::b::__test_reexports::my_test`. While this process seems
pretty safe, what happens if there is an existing `__test_reexports` module?
The answer: nothing.

To explain, we need to understand how Rust's [Abstract Syntax Tree][ast]
represents [identifiers][Ident]. The name of every function, variable, module,
etc. is not stored as a string, but rather as an opaque [Symbol][Symbol] which
is essentially an ID number for each identifier. The compiler keeps a separate
hashtable that allows us to recover the human-readable name of a Symbol when
necessary (such as when printing a syntax error). When the compiler generates
the `__test_reexports` module, it generates a new [Symbol][Symbol] for the
identifier, so while the compiler-generated `__test_reexports` may share a name
with your hand-written one, it will not share a [Symbol][Symbol]. This
technique prevents name collision during code generation and is the foundation
of Rust's [`macro`] hygiene.

## Step 2: Harness generation

Now that our tests are accessible from the root of our crate, we need to do
something with them using [`rustc_ast`][ast] generates a module like so:

```rust,ignore
#[main]
pub fn main() {
    extern crate test;
    test::test_main_static(&[&path::to::test1, /*...*/]);
}
```

Here `path::to::test1` is a constant of type [`test::TestDescAndFn`][tdaf].

While this transformation is simple, it gives us a lot of insight into how
tests are actually run. The tests are aggregated into an array and passed to
a test runner called `test_main_static`. We'll come back to exactly what
[`TestDescAndFn`][tdaf] is, but for now, the key takeaway is that there is a crate
called [`test`][test] that is part of Rust core, that implements all of the
runtime for testing. [`test`][test]'s interface is unstable, so the only stable way
to interact with it is through the `#[test]` macro.

## Step 3: Test object generation

If you've written tests in Rust before, you may be familiar with some of the
optional attributes available on test functions. For example, a test can be
annotated with `#[should_panic]` if we expect the test to cause a panic. It
looks something like this:

```rust,ignore
#[test]
#[should_panic]
fn foo() {
    panic!("intentional");
}
```

This means our tests are more than just simple functions, they have
configuration information as well. `test` encodes this configuration data into
a `struct` called [`TestDesc`]. For each test function in a crate,
[`rustc_ast`][rustc_ast] will parse its attributes and generate a [`TestDesc`]
instance. It then combines the [`TestDesc`] and test function into the
predictably named [`TestDescAndFn`][tdaf] `struct`, that [`test_main_static`]
operates on.
For a given test, the generated [`TestDescAndFn`][tdaf] instance looks like so:

```rust,ignore
self::test::TestDescAndFn{
  desc: self::test::TestDesc{
    name: self::test::StaticTestName("foo"),
    ignore: false,
    should_panic: self::test::ShouldPanic::Yes,
    allow_fail: false,
  },
  testfn: self::test::StaticTestFn(||
    self::test::assert_test_result(::crate::__test_reexports::foo())),
}
```

Once we've constructed an array of these test objects, they're passed to the
test runner via the harness generated in Step 2.

## Inspecting the generated code

On `nightly` `rustc`, there's an unstable flag called `unpretty` that you can use
to print out the module source after [`macro`] expansion:

```bash
$ rustc my_mod.rs -Z unpretty=hir
```

[`macro`]: ./macro-expansion.md
[`TestDesc`]: https://doc.rust-lang.org/test/struct.TestDesc.html
[ast]: ./ast-validation.md
[Ident]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/struct.Ident.html
[rustc_ast]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_ast
[Symbol]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/struct.Symbol.html
[test]: https://doc.rust-lang.org/test/index.html
[tdaf]: https://doc.rust-lang.org/test/struct.TestDescAndFn.html
[`test_main_static`]: https://doc.rust-lang.org/test/fn.test_main_static.html
