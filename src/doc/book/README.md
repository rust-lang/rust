% The Rust Programming Language

Welcome! This book will teach you about the [Rust Programming Language][rust].
Rust is a systems programming language focused on three goals: safety, speed,
and concurrency. It maintains these goals without having a garbage collector,
making it a useful language for a number of use cases other languages aren’t
good at: embedding in other languages, programs with specific space and time
requirements, and writing low-level code, like device drivers and operating
systems. It improves on current languages targeting this space by having a
number of compile-time safety checks that produce no runtime overhead, while
eliminating all data races. Rust also aims to achieve ‘zero-cost abstractions’
even though some of these abstractions feel like those of a high-level language.
Even then, Rust still allows precise control like a low-level language would.

[rust]: https://www.rust-lang.org

“The Rust Programming Language” is split into chapters. This introduction
is the first. After this:

* [Getting started][gs] - Set up your computer for Rust development.
* [Tutorial: Guessing Game][gg] - Learn some Rust with a small project.
* [Syntax and Semantics][ss] - Each bit of Rust, broken down into small chunks.
* [Effective Rust][er] - Higher-level concepts for writing excellent Rust code.
* [Nightly Rust][nr] - Cutting-edge features that aren’t in stable builds yet.
* [Glossary][gl] - A reference of terms used in the book.
* [Bibliography][bi] - Background on Rust's influences, papers about Rust.

[gs]: getting-started.html
[gg]: guessing-game.html
[er]: effective-rust.html
[ss]: syntax-and-semantics.html
[nr]: nightly-rust.html
[gl]: glossary.html
[bi]: bibliography.html

### Contributing

The source files from which this book is generated can be found on
[GitHub][book].

[book]: https://github.com/rust-lang/rust/tree/master/src/doc/book
