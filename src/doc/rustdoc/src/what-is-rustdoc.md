# What is rustdoc?

The standard Rust distribution ships with a tool called `rustdoc`. Its job is
to generate documentation for Rust projects. On a fundamental level, Rustdoc
takes as an argument either a crate root or a Markdown file, and produces HTML,
CSS, and JavaScript.

## Basic usage

Let's give it a try! Let's create a new project with Cargo:

```bash
$ cargo new docs
$ cd docs
```

In `src/lib.rs`, you'll find that Cargo has generated some sample code. Delete
it and replace it with this:

```rust
/// foo is a function
fn foo() {}
```

Let's run `rustdoc` on our code. To do so, we can call it with the path to
our crate root like this:

```bash
$ rustdoc src/lib.rs
```

This will create a new directory, `doc`, with a website inside! In our case,
the main page is located in `doc/lib/index.html`. If you open that up in
a web browser, you'll see a page with a search bar, and "Crate lib" at the
top, with no contents. There's two problems with this: first, why does it
think that our package is named "lib"? Second, why does it not have any
contents?

The first problem is due to `rustdoc` trying to be helpful; like `rustc`,
it assumes that our crate's name is the name of the file for the crate
root. To fix this, we can pass in a command-line flag:

```bash
$ rustdoc src/lib.rs --crate-name docs
```

Now, `doc/docs/index.html` will be generated, and the page says "Crate docs."

For the second issue, it's because our function `foo` is not public; `rustdoc`
defaults to generating documentation for only public functions. If we change
our code...

```rust
/// foo is a function
pub fn foo() {}
```

... and then re-run `rustdoc`:

```bash
$ rustdoc src/lib.rs --crate-name docs
```

We'll have some generated documentation. Open up `doc/docs/index.html` and
check it out! It should show a link to the `foo` function's page, which
is located at `doc/docs/fn.foo.html`. On that page, you'll see the "foo is
a function" we put inside the documentation comment in our crate.

## Using rustdoc with Cargo

Cargo also has integration with `rustdoc` to make it easier to generate
docs. Instead of the `rustdoc` command, we could have done this:

```bash
$ cargo doc
```

Internally, this calls out to `rustdoc` like this:

```bash
$ rustdoc --crate-name docs srclib.rs -o <path>\docs\target\doc -L
dependency=<path>docs\target\debug\deps
```

You can see this with `cargo doc --verbose`.

It generates the correct `--crate-name` for us, as well as pointing to
`src/lib.rs` But what about those other arguments? `-o` controls the
*o*utput of our docs. Instead of a top-level `doc` directory, you'll
notice that Cargo puts generated documentation under `target`. That's
the idiomatic place for generated files in Cargo projects. Also, it
passes `-L`, a flag that helps rustdoc find the dependencies
your code relies on. If our project used dependencies, we'd get
documentation for them as well!

## Using standalone Markdown files

`rustdoc` can also generate HTML from standalone Markdown files. Let's
give it a try: create a `README.md` file with these contents:

````text
# Docs

This is a project to test out `rustdoc`.

[Here is a link!](https://www.rust-lang.org)

## Subheading

```rust
fn foo() -> i32 {
    1 + 1
}
```
````

And call `rustdoc` on it:

```bash
$ rustdoc README.md
```

You'll find an HTML file in `docs/doc/README.html` generated from its
Markdown contents.

Cargo currently does not understand standalone Markdown files, unfortunately.

## Summary

This covers the simplest use-cases of `rustdoc`. The rest of this book will
explain all of the options that `rustdoc` has, and how to use them.