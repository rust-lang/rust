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

“The Rust Programming Language” is split into eight sections. This introduction
is the first. After this:

* [Getting started][gs] - Set up your computer for Rust development.
* [Learn Rust][lr] - Learn Rust programming through small projects.
* [Effective Rust][er] - Higher-level concepts for writing excellent Rust code.
* [Syntax and Semantics][ss] - Each bit of Rust, broken down into small chunks.
* [Nightly Rust][nr] - Cutting-edge features that aren’t in stable builds yet.
* [Glossary][gl] - A reference of terms used in the book.
* [Bibliography][bi] - Background on Rust's influences, papers about Rust.

[gs]: getting-started.html
[lr]: learn-rust.html
[er]: effective-rust.html
[ss]: syntax-and-semantics.html
[nr]: nightly-rust.html
[gl]: glossary.html
[bi]: bibliography.html

After reading this introduction, you’ll want to dive into either ‘Learn Rust’ or
‘Syntax and Semantics’, depending on your preference: ‘Learn Rust’ if you want
to dive in with a project, or ‘Syntax and Semantics’ if you prefer to start
small, and learn a single concept thoroughly before moving onto the next.
Copious cross-linking connects these parts together.

### Contributing

The source files from which this book is generated can be found on GitHub:
[github.com/rust-lang/rust/tree/master/src/doc/trpl](https://github.com/rust-lang/rust/tree/master/src/doc/trpl)

## A brief introduction to Rust

Is Rust a language you might be interested in? Let’s examine a few small code
samples to show off a few of its strengths.

The main concept that makes Rust unique is called ‘ownership’. Consider this
small example:

```rust
fn main() {
    let mut x = vec!["Hello", "world"];
}
```

This program makes a [variable binding][var] named `x`. The value of this
binding is a `Vec<T>`, a ‘vector’, that we create through a [macro][macro]
defined in the standard library. This macro is called `vec`, and we invoke
macros with a `!`. This follows a general principle of Rust: make things
explicit. Macros can do significantly more complicated things than function
calls, and so they’re visually distinct. The `!` also helps with parsing,
making tooling easier to write, which is also important.

We used `mut` to make `x` mutable: bindings are immutable by default in Rust.
We’ll be mutating this vector later in the example.

It’s also worth noting that we didn’t need a type annotation here: while Rust
is statically typed, we didn’t need to explicitly annotate the type. Rust has
type inference to balance out the power of static typing with the verbosity of
annotating types.

Rust prefers stack allocation to heap allocation: `x` is placed directly on the
stack. However, the `Vec<T>` type allocates space for the elements of the vector
on the heap. If you’re not familiar with this distinction, you can ignore it for
now, or check out [‘The Stack and the Heap’][heap]. As a systems programming
language, Rust gives us the ability to control how our memory is allocated, but
when we’re getting started, it’s less of a big deal.

[var]: variable-bindings.html
[macro]: macros.html
[heap]: the-stack-and-the-heap.html

Earlier, we mentioned that ‘ownership’ is the key new concept in Rust. In Rust
parlance, `x` is said to ‘own’ the vector. This means that when `x` goes out of
scope, the vector’s memory will be de-allocated. This is done deterministically
by the Rust compiler, rather than through a mechanism such as a garbage
collector. In other words, in Rust, we don’t call functions like `malloc` and
`free` ourselves: the compiler statically determines when we need to allocate or
deallocate memory, and inserts those calls itself. To err is to be human, but
compilers never forget.

Let’s add another line to our example:

```rust
fn main() {
    let mut x = vec!["Hello", "world"];

    let y = &x[0];
}
```

We’ve introduced another binding, `y`. In this case, `y` is a ‘reference’ to the
first element of the vector. Rust’s references are similar to pointers in other
languages, but with additional compile-time safety checks. References interact
with the ownership system by [‘borrowing’][borrowing] what they point to, rather
than owning it. The difference is, when the reference goes out of scope, it
won't deallocate the underlying memory. If it did, we’d de-allocate twice, which
is bad!

[borrowing]: references-and-borrowing.html

Let’s add a third line. It looks innocent enough, but causes a compiler error:

```rust,ignore
fn main() {
    let mut x = vec!["Hello", "world"];

    let y = &x[0];

    x.push("foo");
}
```

`push` is a method on vectors that appends another element to the end of the
vector. When we try to compile this program, we get an error:

```text
error: cannot borrow `x` as mutable because it is also borrowed as immutable
    x.push("foo");
    ^
note: previous borrow of `x` occurs here; the immutable borrow prevents
subsequent moves or mutable borrows of `x` until the borrow ends
    let y = &x[0];
             ^
note: previous borrow ends here
fn main() {

}
^
```

Whew! The Rust compiler gives quite detailed errors at times, and this is one
of those times. As the error explains, while we made our binding mutable, we
still can't call `push`. This is because we already have a reference to an
element of the vector, `y`. Mutating something while another reference exists
is dangerous, because we may invalidate the reference. In this specific case,
when we create the vector, we may have only allocated space for two elements.
Adding a third would mean allocating a new chunk of memory for all those elements,
copying the old values over, and updating the internal pointer to that memory.
That all works just fine. The problem is that `y` wouldn’t get updated, and so
we’d have a ‘dangling pointer’. That’s bad. Any use of `y` would be an error in
this case, and so the compiler has caught this for us.

So how do we solve this problem? There are two approaches we can take. The first
is making a copy rather than using a reference:

```rust
fn main() {
    let mut x = vec!["Hello", "world"];

    let y = x[0].clone();

    x.push("foo");
}
```

Rust has [move semantics][move] by default, so if we want to make a copy of some
data, we call the `clone()` method. In this example, `y` is no longer a reference
to the vector stored in `x`, but a copy of its first element, `"Hello"`. Now
that we don’t have a reference, our `push()` works just fine.

[move]: ownership.html#move-semantics

If we truly want a reference, we need the other option: ensure that our reference
goes out of scope before we try to do the mutation. That looks like this:

```rust
fn main() {
    let mut x = vec!["Hello", "world"];

    {
        let y = &x[0];
    }

    x.push("foo");
}
```

We created an inner scope with an additional set of curly braces. `y` will go out of
scope before we call `push()`, and so we’re all good.

This concept of ownership isn’t just good for preventing dangling pointers, but an
entire set of related problems, like iterator invalidation, concurrency, and more.
