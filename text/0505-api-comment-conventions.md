- Start Date: 2014-12-08
- RFC PR: [rust-lang/rfcs#505](https://github.com/rust-lang/rfcs/pull/505)
- Rust Issue: N/A

# Summary

This is a conventions RFC, providing guidance on providing API documentation
for Rust projects, including the Rust language itself.

# Motivation

Documentation is an extremely important part of any project. It's important
that we have consistency in our documentation.

For the most part, the RFC proposes guidelines that are already followed today,
but it tries to motivate and clarify them.

# Detailed design

There are a number of individual guidelines. Most of these guidelines are for
any Rust project, but some are specific to documenting `rustc` itself and the
standard library. These are called out specifically in the text itself.

## Use line comments

Avoid block comments. Use line comments instead:

```rust
// Wait for the main task to return, and set the process error code
// appropriately.
```

Instead of:

```rust
/*
 * Wait for the main task to return, and set the process error code
 * appropriately.
 */
```

Only use inner doc comments //! to write crate and module-level documentation,
nothing else. When using `mod` blocks, prefer `///` outside of the block:

```rust
/// This module contains tests
mod test {
    // ...
}
```

over

```rust
mod test {
    //! This module contains tests

    // ...
}
```

## Formatting

The first line in any doc comment should be a single-line short sentence
providing a summary of the code. This line is used as a summary description
throughout Rustdoc's output, so it's a good idea to keep it short.

All doc comments, including the summary line, should be properly punctuated.
Prefer full sentences to fragments.

The summary line should be written in third person singular present indicative
form. Basically, this means write "Returns" instead of "Return".

## Using Markdown

Within doc comments, use Markdown to format your documentation.

Use top level headings # to indicate sections within your comment. Common headings:

* Examples
* Panics
* Failure

Even if you only include one example, use the plural form: "Examples" rather
than "Example". Future tooling is easier this way.

Use graves (`) to denote a code fragment within a sentence.

Use triple graves (```) to write longer examples, like this:

    This code does something cool.

    ```rust
    let x = foo();
    x.bar();
    ```

When appropriate, make use of Rustdoc's modifiers. Annotate triple grave blocks with
the appropriate formatting directive. While they default to Rust in Rustdoc, prefer
being explicit, so that it highlights syntax in places that do not, like GitHub.

    ```rust
    println!("Hello, world!");
    ```

    ```ruby
    puts "Hello"
    ```

Rustdoc is able to test all Rust examples embedded inside of documentation, so
it's important to mark what is not Rust so your tests don't fail.

References and citation should be linked 'reference style.' Prefer

```
[Rust website][1]

[1]: http://www.rust-lang.org
```

to

```
[Rust website](http://www.rust-lang.org)
```

## English

This section applies to `rustc` and the standard library.

All documentation is standardized on American English, with regards to
spelling, grammar, and punctuation conventions. Language changes over time,
so this doesn't mean that there is always a correct answer to every grammar
question, but there is often some kind of formal consensus.

# Drawbacks

None.

# Alternatives

Not having documentation guidelines.

# Unresolved questions

None.
