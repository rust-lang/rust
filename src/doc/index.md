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

# API Documentation

Rust provides a standard library with a number of features; [we host its
documentation here][api].

# Extended Error Documentation

Many of Rust's errors come with error codes, and you can request extended
diagnostics from the compiler on those errors. We also [have the text of those
extended errors on the web][err], if you prefer to read them that way.

# The Rust Bookshelf

Rust provides a number of book-length sets of documentation, collectively
nicknamed 'The Rust Bookshelf.'

* [The Rust Programming Language][book] teaches you how to program in Rust.
* [The Cargo Book][cargo-book] is a guide to Cargo, Rust's build tool and dependency manager.
* [The Unstable Book][unstable-book] has documentation for unstable features.
* [The Rustonomicon][nomicon] is your guidebook to the dark arts of unsafe Rust.
* [The Reference][ref] is not a formal spec, but is more detailed and comprehensive than the book.
* [The Rustdoc Book][rustdoc-book] describes our documentation tool, `rustdoc`.

Initially, documentation lands in the Unstable Book, and then, as part of the
stabilization process, is moved into the Book, Nomicon, or Reference.

Another few words about the reference: it is guaranteed to be accurate, but not
complete. We have a policy that features must have documentation to be stabilized,
but we did not always have this policy, and so there are some stable things that
are not yet in the reference. We're working on back-filling things that landed
before this policy was put into place. That work is being tracked
[here][refchecklist].

[Rust Learning]: https://github.com/ctjhoa/rust-learning
[Docs.rs]: https://docs.rs/
[api]: std/index.html
[ref]: reference/index.html
[refchecklist]: https://github.com/rust-lang-nursery/reference/issues/9
[err]: error-index.html
[book]: book/index.html
[nomicon]: nomicon/index.html
[unstable-book]: unstable-book/index.html
[rustdoc-book]: rustdoc/index.html
[cargo-book]: cargo/index.html

