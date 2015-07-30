% The Advanced Rust Programming Language

# NOTE: This is a draft document, and may contain serious errors

So you've played around with Rust a bit. You've written a few simple programs
and you think you grok the basics. Maybe you've even read through *[The Rust
Programming Language][trpl]* (TRPL). Now you want to get neck-deep in all the
nitty-gritty details of the language. You want to know those weird corner-cases.
You want to know what the heck `unsafe` really means, and how to properly use
it. This is the book for you.

To be clear, this book goes into *serious* detail. We're going to dig into
exception-safety and pointer aliasing. We're going to talk about memory
models. We're even going to do some type-theory. This is stuff that you
absolutely *don't* need to know to write fast and safe Rust programs.
You could probably close this book *right now* and still have a productive
and happy career in Rust.

However if you intend to write unsafe code -- or just *really* want to dig into
the guts of the language -- this book contains *invaluable* information.

Unlike TRPL we will be assuming considerable prior knowledge. In particular, you
should be comfortable with basic systems programming and basic Rust. If you
don't feel comfortable with these topics, you should consider [reading
TRPL][trpl], though we will not be assuming that you have. You can skip
straight to this book if you want; just know that we won't be explaining
everything from the ground up.

Due to the nature of advanced Rust programming, we will be spending a lot of
time talking about *safety* and *guarantees*. In particular, a significant
portion of the book will be dedicated to correctly writing and understanding
Unsafe Rust.

[trpl]: ../book/
