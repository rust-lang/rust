% Meet Safe and Unsafe

Programmers in safe "high-level" languages face a fundamental dilemma. On one
hand, it would be *really* great to just say what you want and not worry about
how it's done. On the other hand, that can lead to unacceptably poor
performance. It may be necessary to drop down to less clear or idiomatic
practices to get the performance characteristics you want. Or maybe you just
throw up your hands in disgust and decide to shell out to an implementation in
a less sugary-wonderful *unsafe* language.

Worse, when you want to talk directly to the operating system, you *have* to
talk to an unsafe language: *C*. C is ever-present and unavoidable. It's the
lingua-franca of the programming world.
Even other safe languages generally expose C interfaces for the world at large!
Regardless of why you're doing it, as soon as your program starts talking to
C it stops being safe.

With that said, Rust is *totally* a safe programming language.

Well, Rust *has* a safe programming language. Let's step back a bit.

Rust can be thought of as being composed of two programming languages: *Safe
Rust* and *Unsafe Rust*. Safe Rust is For Reals  Totally Safe. Unsafe Rust,
unsurprisingly, is *not* For Reals Totally Safe.  In fact, Unsafe Rust lets you
do some really crazy unsafe things.

Safe Rust is the *true* Rust programming language. If all you do is write Safe
Rust, you will never have to worry about type-safety or memory-safety. You will
never endure a null or dangling pointer, or any of that Undefined Behavior
nonsense.

*That's totally awesome.*

The standard library also gives you enough utilities out-of-the-box that you'll
be able to write awesome high-performance applications and libraries in pure
idiomatic Safe Rust.

But maybe you want to talk to another language. Maybe you're writing a
low-level abstraction not exposed by the standard library. Maybe you're
*writing* the standard library (which is written entirely in Rust). Maybe you
need to do something the type-system doesn't understand and just *frob some dang
bits*. Maybe you need Unsafe Rust.

Unsafe Rust is exactly like Safe Rust with all the same rules and semantics.
However Unsafe Rust lets you do some *extra* things that are Definitely Not Safe.

The only things that are different in Unsafe Rust are that you can:

* Dereference raw pointers
* Call `unsafe` functions (including C functions, intrinsics, and the raw allocator)
* Implement `unsafe` traits
* Mutate statics

That's it. The reason these operations are relegated to Unsafe is that misusing
any of these things will cause the ever dreaded Undefined Behavior. Invoking
Undefined Behavior gives the compiler full rights to do arbitrarily bad things
to your program. You definitely *should not* invoke Undefined Behavior.

Unlike C, Undefined Behavior is pretty limited in scope in Rust. All the core
language cares about is preventing the following things:

* Dereferencing null or dangling pointers
* Reading [uninitialized memory]
* Breaking the [pointer aliasing rules]
* Producing invalid primitive values:
    * dangling/null references
    * a `bool` that isn't 0 or 1
    * an undefined `enum` discriminant
    * a `char` outside the ranges [0x0, 0xD7FF] and [0xE000, 0x10FFFF]
    * A non-utf8 `str`
* Unwinding into another language
* Causing a [data race][race]

That's it. That's all the causes of Undefined Behavior baked into Rust. Of
course, unsafe functions and traits are free to declare arbitrary other
constraints that a program must maintain to avoid Undefined Behavior. However,
generally violations of these constraints will just transitively lead to one of
the above problems. Some additional constraints may also derive from compiler
intrinsics that make special assumptions about how code can be optimized.

Rust is otherwise quite permissive with respect to other dubious operations.
Rust considers it "safe" to:

* Deadlock
* Have a [race condition][race]
* Leak memory
* Fail to call destructors
* Overflow integers
* Abort the program
* Delete the production database

However any program that actually manages to do such a thing is *probably*
incorrect. Rust provides lots of tools to make these things rare, but
these problems are considered impractical to categorically prevent.

[pointer aliasing rules]: references.html
[uninitialized memory]: uninitialized.html
[race]: races.html
