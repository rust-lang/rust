# Testing

Developing lints for Clippy is a Test-Driven Development (TDD) process because
our first task before implementing any logic for a new lint is to write some test cases.

## Develop Lints with Tests

When we develop Clippy, we enter a complex and chaotic realm full of
programmatic issues, stylistic errors, illogical code and non-adherence to convention.
Tests are the first layer of order we can leverage to define when and where
we want a new lint to trigger or not.

Moreover, writing tests first help Clippy developers to find a balance for
the first iteration of and further enhancements for a lint.
With test cases on our side, we will not have to worry about over-engineering
a lint on its first version nor missing out some obvious edge cases of the lint.
This approach empowers us to iteratively enhance each lint.

## Clippy UI Tests

We use **UI tests** for testing in Clippy. These UI tests check that the output
of Clippy is exactly as we expect it to be. Each test is just a plain Rust file
that contains the code we want to check.

The output of Clippy is compared against a `.stderr` file. Note that you don't
have to create this file yourself. We'll get to generating the `.stderr` files
with the command [`cargo bless`](#cargo-bless) (seen later on).

### Write Test Cases

Let us now think about some tests for our imaginary `foo_functions` lint. We
start by opening the test file `tests/ui/foo_functions.rs` that was created by
`cargo dev new_lint`.

Update the file with some examples to get started:

```rust
#![warn(clippy::foo_functions)] // < Add this, so the lint is guaranteed to be enabled in this file

// Impl methods
struct A;
impl A {
    pub fn fo(&self) {}
    pub fn foo(&self) {}
    //~^ foo_functions
    pub fn food(&self) {}
}

// Default trait methods
trait B {
    fn fo(&self) {}
    fn foo(&self) {}
    //~^ foo_functions
    fn food(&self) {}
}

// Plain functions
fn fo() {}
fn foo() {}
//~^ foo_functions
fn food() {}

fn main() {
    // We also don't want to lint method calls
    foo();
    let a = A;
    a.foo();
}
```

Without actual lint logic to emit the lint when we see a `foo` function name,
this test will fail, because we expect errors at lines marked with
`//~^ foo_functions`. However, we can now run the test with the following command:

```sh
$ TESTNAME=foo_functions cargo uitest
```

Clippy will compile and it will fail complaining it didn't receive any errors:

```
...Clippy warnings and test outputs...
error: diagnostic code `clippy::foo_functions` not found on line 8
 --> tests/ui/foo_functions.rs:9:10
  |
9 |     //~^ foo_functions
  |          ^^^^^^^^^^^^^ expected because of this pattern
  |

error: diagnostic code `clippy::foo_functions` not found on line 16
  --> tests/ui/foo_functions.rs:17:10
   |
17 |     //~^ foo_functions
   |          ^^^^^^^^^^^^^ expected because of this pattern
   |

error: diagnostic code `clippy::foo_functions` not found on line 23
  --> tests/ui/foo_functions.rs:24:6
   |
24 | //~^ foo_functions
   |      ^^^^^^^^^^^^^ expected because of this pattern
   |

```

This is normal. After all, we wrote a bunch of Rust code but we haven't really
implemented any logic for Clippy to detect `foo` functions and emit a lint.

As we gradually implement our lint logic, we will keep running this UI test command.
Clippy will begin outputting information that allows us to check if the output is
turning into what we want it to be.

### Example output

As our `foo_functions` lint is tested, the output would look something like this:

```
failures:
---- compile_test stdout ----
normalized stderr:
error: function called "foo"
  --> tests/ui/foo_functions.rs:6:12
   |
LL |     pub fn foo(&self) {}
   |            ^^^
   |
   = note: `-D clippy::foo-functions` implied by `-D warnings`
error: function called "foo"
  --> tests/ui/foo_functions.rs:13:8
   |
LL |     fn foo(&self) {}
   |        ^^^
error: function called "foo"
  --> tests/ui/foo_functions.rs:19:4
   |
LL | fn foo() {}
   |    ^^^
error: aborting due to 3 previous errors
```

Note the *failures* label at the top of the fragment, we'll get rid of it
(saving this output) in the next section.

> _Note:_ You can run multiple test files by specifying a comma separated list:
> `TESTNAME=foo_functions,bar_methods,baz_structs`.

### `cargo bless`

Once we are satisfied with the output, we need to run this command to
generate or update the `.stderr` file for our lint:

```sh
$ TESTNAME=foo_functions cargo uibless
```

This writes the emitted lint suggestions and fixes to the `.stderr` file, with
the reason for the lint, suggested fixes, and line numbers, etc.

Running `TESTNAME=foo_functions cargo uitest` should pass then. When we commit
our lint, we need to commit the generated `.stderr` files, too.

In general, you should only commit files changed by `cargo bless` for the
specific lint you are creating/editing.

> _Note:_ If the generated `.stderr`, and `.fixed` files are empty,
> they should be removed.

## `toml` Tests

Some lints can be configured through a `clippy.toml` file. Those configuration
values are tested in `tests/ui-toml`.

To add a new test there, create a new directory and add the files:

- `clippy.toml`: Put here the configuration value you want to test.
- `lint_name.rs`: A test file where you put the testing code, that should see a
  different lint behavior according to the configuration set in the
  `clippy.toml` file.

The potential `.stderr` and `.fixed` files can again be generated with `cargo
bless`.

## Cargo Lints

The process of testing is different for Cargo lints in that now we are
interested in the `Cargo.toml` manifest file. In this case, we also need a
minimal crate associated with that manifest. Those tests are generated in
`tests/ui-cargo`.

Imagine we have a new example lint that is named `foo_categories`, we can run:

```sh
$ cargo dev new_lint --name=foo_categories --pass=late --category=cargo
```

After running `cargo dev new_lint` we will find by default two new crates,
each with its manifest file:

* `tests/ui-cargo/foo_categories/fail/Cargo.toml`: this file should cause the
  new lint to raise an error.
* `tests/ui-cargo/foo_categories/pass/Cargo.toml`: this file should not trigger
  the lint.

If you need more cases, you can copy one of those crates (under
`foo_categories`) and rename it.

The process of generating the `.stderr` file is the same as for other lints
and prepending the `TESTNAME` variable to `cargo uitest` works for Cargo lints too.

## Rustfix Tests

If the lint you are working on is making use of structured suggestions,
[`rustfix`] will apply the suggestions from the lint to the test file code and
compare that to the contents of a `.fixed` file.

Structured suggestions tell a user how to fix or re-write certain code that has
been linted with [`span_lint_and_sugg`].

Should `span_lint_and_sugg` be used to generate a suggestion, but not all
suggestions lead to valid code, you can use the `//@no-rustfix` comment on top
of the test file, to not run `rustfix` on that file.

We'll talk about suggestions more in depth in a [later chapter](emitting_lints.md).

Use `cargo bless` to automatically generate the `.fixed` file after running
the tests.

[`rustfix`]: https://github.com/rust-lang/cargo/tree/master/crates/rustfix
[`span_lint_and_sugg`]: https://doc.rust-lang.org/beta/nightly-rustc/clippy_utils/diagnostics/fn.span_lint_and_sugg.html

## Testing Manually

Manually testing against an example file can be useful if you have added some
`println!`s and the test suite output becomes unreadable.

To try Clippy with your local modifications, run from the working copy root.

```sh
$ cargo dev lint input.rs
```
