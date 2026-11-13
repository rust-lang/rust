rust-analyzer's testing is based on *snapshot tests*: a test is a piece of input text, usually Rust code, and some output text. There is then some testing helper that runs the feature on the input text and compares the result to the output text.

rust-analyzer uses a combination of the crate [`expect-test`](https://docs.rs/expect-test) and a custom testing framework.

This all may sound too abstract, so let's demonstrate with an example.

Type inference tests are located at `crates/hir-ty/src/tests`. There are various test helpers you can use. One of the simplest is `check_no_mismatches()`: it is given a piece of Rust code (we'll talk more about Rust code in tests later) and asserts that there are no type mismatches in it, that is, one type was expected but another was found (for example, `let x: () = 1` is a type mismatch). Note that we determine type mismatches via rust-analyzer's own analysis, not via the compiler (this is what we are testing, after all), which means there are often missed mismatches and sometimes bogus ones as well.

For example, the following test will fail:
```rust
#[test]
fn this_will_fail() {
    check_no_mismatches(
        r#"
fn main() {
    let x: () = 1;
}
    "#,
    );
}
```

Sometimes we want to check more than that there are no type mismatches. For that we use other helpers. For example, often we want to assert that the type of some expression is some specific type. For that we use the `check_types()` function. It takes a Rust code string with custom annotation, which are common in our test suite. The general scheme of annotation is:

 - `$0` marks a position. What to do with it is determined by the testing helper. Commonly it denotes the cursor position in IDE tests (for example, hover).
 - `$0...$0` marks a range, commonly a selection in IDE tests.
 - `^...^`, commonly seen in a comment (`// ^^^^`), labels the line above. For example, the following will attach the label `hey` to the range of the variable name `cool`:

    ```rust
    let cool;
     // ^^^^ hey
    ```

`check_types()` uses labels to assert types: when you attach a label to a range, `check_types()` asserts that the type of this range will be what is written in the label.

It's all too abstract without an example:
```rust
#[test]
fn my_test() {
    check_types(
        r#"
fn main() {
    let x = 1;
     // ^ i32
}
    "#,
    );
}
```
Here, we assert that the type of the variable `x` is `i32`. Which is true, of course, so the test will pass.

Oftentimes it is convenient to assert the types of all of the expressions at once, and that brings us to the last kind of test. It uses `expect-test` to match an output text:
```rust
#[test]
fn my_test() {
    check_infer(
        r#"
fn main() {
    let x = 1;
}
    "#,
        expect![[r#"
            10..28 '{     ...= 1; }': ()
            20..21 'x': i32
            24..25 '1': i32
        "#]],
    );
}
```
The text inside the `expect![[]]` is determined by the helper, `check_infer()` in this case. For `check_infer()`, each line is a range in the source code (the range is counted in bytes and the source is trimmed, so indentation is stripped); next to it there is the text in that range, or some part of it with `...` if it's too long, and finally comes the type of that range.

The important feature of `expect-test` is that it allows easy update of the expectation. Say you changed something in the code, maybe fixed a bug, and the output in `expect![[]]` needs to change. Or maybe you are writing it from scratch. Writing it by hand is very tedious and prone to mistakes. But `expect-trait` has some magic. You can set the environment variable `UPDATE_EXPECT=1`, then run the test, and it will update automatically! Some editors (e.g. VSCode) make it even more convenient: on them, at the top of every test that uses `expect-test`, next to the usual `Run | Debug` buttons, rust-analyzer also shows an `Update Expect` button. Clicking it will run that test in updating mode.

## Rust code in the tests

The first thing that you probably already noticed is that the Rust code in the tests is syntax highlighted! In fact, it even uses semantic highlighting. rust-analyzer highlights strings "as if" they contain Rust code if they are passed to a parameter marked `#[rust_analyzer::rust_fixture]`, and rust-analyzer test helpers do that (in fact, this was designed for them).

The syntax highlighting is very important, not just because it's nice to the eye: it's very easy to make mistakes in test code, and debugging that can be very hard. Often the test will just fail, printing an `{unknown}` type, and you'll have no clue what's going wrong. The syntax is the clue; if something isn't highlighted correctly, that probably means there is an error (there is one exception to this, which we'll discuss later). You can even set the semantic highlighting tag `unresolved_reference` to e.g. red, so you will see such things clearly.

Still, often you won't know what's going wrong. Why you can't fix the test, or worse, you expect it to fail but it doesn't. You can try the code on a real IDE to be sure it works. Later we'll give some tips for fixing the test.

### The fixture

The Rust code in a test is not, a fact, a single Rust file. It uses a mini-language that allows you to express multiple files, multiple crates, different configs, and more. All options are documented in `crates/test-utils/src/fixture.rs`, but here are some of the common ones:

 - `//- minicore: flag1, flag2, ...`. This is by far the most common option. Tests in rust-analyzer don't have access by default to any other type - not `Option`, not `Iterator`, not even `Sized`. This option allows you to include parts of the `crates/test-utils/src/minicore.rs` file, which mimics `core`. All possible flags are listed at the top of `minicore` along with the flags they imply, then later you can see by `// region:flag` and `// endregion:flag` what code each flag enables.
 - `// /path/to/file.rs crate:crate deps:dep_a,dep_b`. The first component is the filename of the code that follows (until the next file). It is required, but only if you supply this line. Other components in this line are optional. They include `crate:crate_name`, to start a new crate, or `deps:dep_a,dep_b`, to declare dependencies between crates. You can also declare modules as usual in Rust - just name your paths `/foo.rs` or `/foo/mod.rs`, declare `mod foo` and that's it!

So the following snippet:
```rust
//- minicore: sized, fn
// /lib.rs crate:foo
pub mod bar;
// /bar.rs
pub struct Bar;
// /main.rs crate:main deps:foo
use foo::Bar;
```
declares two crates `foo` and `main`, where `main` depends on `foo`, with dependencies on the `Sized` and `FnX` traits from `core`, and a module of `foo` called `bar`.

And as promised, here are some tips for making your test work:

 - If you use some type/trait, you must *always* include it in `minicore`. Note - not all types from core/std are available there, but you can add new ones (under flags) if you need. And import them if they are not in the prelude.
 - If you use unsized types (`dyn Trait`/slices), you may want to include some or all of the following `minicore` flags: `sized`, `unsize`, `coerce_unsized`, `dispatch_from_dyn`.
 - If you use closures, consider including the `fn` minicore flag. Async closures need the `async_fn` flag.
 - `sized` is commonly needed, so consider adding it if you're stuck.
