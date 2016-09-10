% Ownership and Lifetimes

Ownership is the breakout feature of Rust. It allows Rust to be completely
memory-safe and efficient, while avoiding garbage collection. Before getting
into the ownership system in detail, we will consider the motivation of this
design.

We will assume that you accept that garbage collection (GC) is not always an
optimal solution, and that it is desirable to manually manage memory in some
contexts. If you do not accept this, might I interest you in a different
language?

Regardless of your feelings on GC, it is pretty clearly a *massive* boon to
making code safe. You never have to worry about things going away *too soon*
(although whether you still wanted to be pointing at that thing is a different
issue...). This is a pervasive problem that C and C++ programs need to deal
with. Consider this simple mistake that all of us who have used a non-GC'd
language have made at one point:

```rust,ignore
fn as_str(data: &u32) -> &str {
    // compute the string
    let s = format!("{}", data);

    // OH NO! We returned a reference to something that
    // exists only in this function!
    // Dangling pointer! Use after free! Alas!
    // (this does not compile in Rust)
    &s
}
```

This is exactly what Rust's ownership system was built to solve.
Rust knows the scope in which the `&s` lives, and as such can prevent it from
escaping. However this is a simple case that even a C compiler could plausibly
catch. Things get more complicated as code gets bigger and pointers get fed through
various functions. Eventually, a C compiler will fall down and won't be able to
perform sufficient escape analysis to prove your code unsound. It will consequently
be forced to accept your program on the assumption that it is correct.

This will never happen to Rust. It's up to the programmer to prove to the
compiler that everything is sound.

Of course, Rust's story around ownership is much more complicated than just
verifying that references don't escape the scope of their referent. That's
because ensuring pointers are always valid is much more complicated than this.
For instance in this code,

```rust,ignore
let mut data = vec![1, 2, 3];
// get an internal reference
let x = &data[0];

// OH NO! `push` causes the backing storage of `data` to be reallocated.
// Dangling pointer! Use after free! Alas!
// (this does not compile in Rust)
data.push(4);

println!("{}", x);
```

naive scope analysis would be insufficient to prevent this bug, because `data`
does in fact live as long as we needed. However it was *changed* while we had
a reference into it. This is why Rust requires any references to freeze the
referent and its owners.

