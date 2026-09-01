# Walkthrough: adding a new test

This chapter gives an example of a small change that you could make as your first contribution to Rust,
using [rust#59333] and [rust#161442] as an example.

[rust#59333]: https://github.com/rust-lang/rust/issues/59333
[rust#161442]: https://github.com/rust-lang/rust/issues/161442

## Find an `E-needs-test` issue to work on

See ["What should I work on?"](../getting-started.md#What-should-I-work-on) for a list of possible tasks you could try out.
Here, we've chosen an `E-needs-test` issue: 
[rust#59333](https://github.com/rust-lang/rust/issues/59333).
This is an especially good fit because it has an example
[directly in the issue](https://github.com/rust-lang/rust/issues/59333#issuecomment-555973113),
without needing additional work from you to minimize the bug.
Of course, it's always very helpful for you to take `E-needs-test` that *doesn't* have a minimal example and create one.

## Reproduce the issue

The example here was posted in 2019, fully 7 years ago (at time of writing).
Quite a lot of things change in the compiler in that period of time.
To make sure the test is still accurate, *reproduce it with the most recent compiler*.
An easy way to do this is with [play.rust-lang.org](https://play.rust-lang.org/?version=nightly) on the `nightly` branch,
or [rust.godbolt.org](https://rust.godbolt.org/) with `rustc nightly`.
You can also use `rustc +nightly` locally if you need complicated setup that isn't possible on Playground.

## Convert the issue to a test

Check out and set up the `rust-lang/rust` repo, as documented in [Quickstart](../building/quickstart.md):

```console
$ git clone https://github.com/rust-lang/rust
$ cd rust
$ ./x setup compiler
$ ./x build library
```

Here, we use `compiler` as the default profile, since the bug we're fixing is related to the compiler.
If you're adding a unit test to the standard library, you'd use `./x setup library`.

This will also suggest setting up a [`.git/hooks/pre-push` check][pre-push].
This is optional, but recommended.

[pre-push]: ../building/suggested.md#installing-a-pre-push-hook

We also started a build in the background with `./x build library`.
Rust unfortunately takes quite a while to build,
so starting a build early lets it run in the background while you're working on other things.

See [UI tests](../tests/ui.md) for a guide on adding new tests.
In rare cases, you may need a [run-make](../tests/compiletest.md#run-make-tests) or even more specialized kind of test.
See [Compiletest](../tests/compiletest.md) for more information.

In our case, our test is fairly simple:

```rust
// Save this file to `tests/ui/lint/dead-code/type-alias-used-in-impl-59333.rs`.

//@ check-pass
//! Regression test for <https://github.com/rust-lang/rust/issues/59333>.
//! A type alias used only as (part of) the self type of an impl was
//! incorrectly flagged as dead code.

#![deny(dead_code)]

struct Runner;

type RuntimeImpl = Runner;

trait Runtime {
    fn run(&mut self);
}

impl Runtime for &mut RuntimeImpl {
    fn run(&mut self) {}
}

struct Walker;

type WalkerImpl = Walker;

trait Walk {
    fn walk(&self) {}
}

impl Walk for WalkerImpl {}

fn main() {
    let mut runner = Runner;
    (&mut runner).run();
    Walker.walk();
}
```

Most of the details here don't matter too much, but note the `//@ check-pass` and `#![deny(dead_code)]` at the top.
Together, those ensure that the compiler doesn't emit a `dead_code` lint when compiling this file.

Also note the "Regression test for ..." comment.
This is *very* helpful for your reviewer, since it helps them understand what the test is doing and whether there's a simpler way to test the behavior.
Please do your best to write a complete description for the test.

Run your test.
[rust#161442] named its test `tests/ui/lint/dead-code/type-alias-used-in-impl-59333.rs`,
so you could run:

```
./x test tests/ui/lint/dead-code/type-alias-used-in-impl-59333.rs
```

If that passes, your test is ready.

## Open a PR

Follow the instructions in [Using Git](../git.md):

First, run the pre-push check if you didn't set it up earlier:

```sh
./x test tidy
```

Then, [open the PR](https://guides.github.com/activities/forking/#making-a-pull-request):

```
git switch --create issue-59333-test
git add tests/ui/lint/dead-code/type-alias-used-in-impl-59333.rs
git commit
git remote add personal https://github.com/YOUR_USERNAME_HERE/rust.git
git push --set-upstream personal issue-59333-test
```

## Review and feedback

See [Opening a PR](../git.md#opening-a-pr).
