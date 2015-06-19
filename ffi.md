% Interfacing with other Languages (FFI)

*Obviously* we'd all love to live in a **glorious** world where everything is
written in Rust, Rust, and More Rust. Tragically, programs have been written
in Not Rust for over 50 years. Crufty enterprises are doomed to
support ancient code bases, and greybeard programmers stuck in their ways
*insist* on writing programs in other languages, even to this day!

In all seriousness, there's a myriad of reasons for your codebase to be a
hybrid of different languages, and Rust is well-designed to interface with
all of them as painlessly as possible. It does this through the tried and
true strategy of all languages: pretend to be C, and understand C.

Thanks to Rust's minimal runtime and C-like semantics, this is about as
painless as FFI with C++. Obviously, most of Rust's features are completely
incompatible with other languages: tagged unions, zero-sized-types, dynamically-
sized types, destructors, methods, traits, references, and lifetimes are all
concepts that you won't be able to expose or accept in your foreign function
interface.

All mapping through C will give you is functions, structs, globals, raw pointers,
and C-like enums. That's it. Rust's default data layouts are also incompatible
with the C layout. See [the section on data layout][data.html] for details.
Long story short: mark FFI structs and enums with `#[repr(C)]`, mark FFI
functions as `extern`.

## Runtime

Rust's runtime is sufficiently minimal that it requires *no* special handling.
You don't need to set anything up. You don't need to tear anything down.
Awesome.

The only runtime detail you *really* need to worry about is unwinding. Rust's
unwinding model is not defined to be incompatible with any particular language.
That means that if you call Rust from another language and it unwinds into the
calling language, this will cause Undefined Behaviour. Similarly, if another
language unwinds into Rust, it will also cause Undefined Behaviour.

Rust can't really do anything about other languages unwinding into it (FFI is unsafe
for a reason!), but you can be a good FFI citizen by catching panics in any
FFI functions you export. Rust provides `thread::catch_panic` for exactly this.
Unfortunately, this API is still unstable.

## libc

