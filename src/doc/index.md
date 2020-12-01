% Rust Documentation

<style>
nav {
    display: none;
}
#search-input {
    width: calc(100% - 58px);
}
#search-but {
    cursor: pointer;
}
#search-but, #search-input {
    padding: 4px;
    border: 1px solid #ccc;
    border-radius: 3px;
    outline: none;
    font-size: 0.7em;
    background-color: #fff;
}
#search-but:hover, #search-input:focus {
    border-color: #55a9ff;
}
h2 {
    font-size: 18px;
}
</style>

Welcome to an overview of the documentation provided by the [Rust project].
All of these projects are managed by the Docs Team; there are other
unofficial documentation resources as well!

Many of these resources take the form of "books"; we collectively call these
"The Rust Bookshelf." Some are large, some are small.

# Learn Rust

If you'd like to learn Rust, this is the spot for you! All of these resources
assume that you have programmed before, but not in any specific language:

## The Rust Programming Language

Affectionately nicknamed "the book," [The Rust Programming
Language](book/index.html) will give you an overview of the language from
first principles. You'll build a few projects along the way, and by the end,
you'll have a solid grasp of the language.

## Rust By Example

If reading multiple hundreds of pages about a language isn't your style, then
[Rust By Example](rust-by-example/index.html) has you covered. While the book talks about code with
a lot of words, RBE shows off a bunch of code, and keeps the talking to a
minimum. It also includes exercises!

## Rustlings

[Rustlings](https://github.com/rust-lang/rustlings) guides you through downloading and setting up the Rust toolchain,
and teaches you the basics of reading and writing Rust syntax. It's an
alternative to Rust by Example that works with your own environment.

# Use Rust

Once you've gotten familiar with the language, these resources can help you
when you're actually using it day-to-day.

## The Standard Library

Rust's standard library has [extensive API documentation](std/index.html),
with explanations of how to use various things, as well as example code for
accomplishing various tasks.

<div>
  <form action="std/index.html" method="get">
    <input id="search-input" type="search" name="search"
           placeholder="Search through the standard library"/>
    <button id="search-but">Search</button>
  </form>
</div>

## The Edition Guide

[The Edition Guide](edition-guide/index.html) describes the Rust editions.

## The Rustc Book

[The Rustc Book](rustc/index.html) describes the Rust compiler, `rustc`.

## The Cargo Book

[The Cargo Book](cargo/index.html) is a guide to Cargo, Rust's build tool and dependency manager.

## The Rustdoc Book

[The Rustdoc Book](rustdoc/index.html) describes our documentation tool, `rustdoc`.

## Extended Error Listing

Many of Rust's errors come with error codes, and you can request extended
diagnostics from the compiler on those errors. You can also [read them
here](error-index.html), if you prefer to read them that way.

# Master Rust

Once you're quite familiar with the language, you may find these advanced
resources useful.

## The Reference

[The Reference](reference/index.html) is not a formal spec, but is more detailed and
comprehensive than the book.

## The Rustonomicon

[The Rustonomicon](nomicon/index.html) is your guidebook to the dark arts of unsafe
Rust. It's also sometimes called "the 'nomicon."

## The Unstable Book

[The Unstable Book](unstable-book/index.html) has documentation for unstable features.

## The `rustc` Contribution Guide

[The `rustc` Guide](https://rustc-dev-guide.rust-lang.org/) documents how
the compiler works and how to contribute to it. This is useful if you want to build
or modify the Rust compiler from source (e.g. to target something non-standard).

# Specialize Rust

When using Rust in specific domain areas, consider using the following resources tailored to each domain.

## Embedded Systems

When developing for Bare Metal or Embedded Linux systems, you may find these resources maintained by the [Embedded Working Group] useful.

[Embedded Working Group]: https://github.com/rust-embedded

### The Embedded Rust Book

[The Embedded Rust Book] is targeted at developers familiar with embedded development and familiar with Rust, but have not used Rust for embedded development.

[The Embedded Rust Book]: embedded-book/index.html
[Rust project]: https://www.rust-lang.org
