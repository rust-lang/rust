% Rust Documentation

<style>
nav {
    display: none;
}
</style>

This page is an overview of the documentation included with your Rust install.
Other unofficial documentation may exist elsewhere; for example, the [Rust
Learning] project collects documentation from the community, and [Docs.rs]
builds documentation for individual Rust packages.

## API Documentation

Rust provides a standard library with a number of features; [we host its
documentation here][api].

## Reference Documentation

Rust does not yet have a formal specification, but we have [a reference document
][ref]. It is guaranteed to be accurate, but not complete. We now have a
policy that all new features must be included in the reference before
stabilization; however, we are still back-filling things that landed before
then. That work is being tracked [here][38643].

## Extended Error Documentation

Many of Rust's errors come with error codes, and you can request extended
diagnostics from the compiler on those errors. We also [have the text of those
extended errors on the web][err], if you prefer to read them that way.

## The Rust Bookshelf

Rust provides a number of book-length sets of documentation, collectively
nicknamed 'The Rust Bookshelf.'

* [The Rust Programming Language][book] teaches you how to program in Rust.
* [The Rustonomicon][nomicon] is your guidebook to the dark arts of unsafe Rust.

[Rust Learning]: https://github.com/ctjhoa/rust-learning
[Docs.rs]: https://docs.rs/
[api]: std/index.html
[ref]: reference.html
[38643]: https://github.com/rust-lang/rust/issues/38643
[err]: error-index.html
[book]: book/index.html
[nomicon]: nomicon/index.html

