% const and static

Rust has a way of defining constants with the `const` keyword:

```rust
const N: i32 = 5;
```

Unlike [`let`][let] bindings, you must annotate the type of a `const`.

[let]: variable-bindings.html

Constants live for the entire lifetime of a program. More specifically,
constants in Rust have no fixed address in memory. This is because they’re
effectively inlined to each place that they’re used. References to the same
constant are not necessarily guaranteed to refer to the same memory address for
this reason.

# `static`

Rust provides a ‘global variable’ sort of facility in static items. They’re
similar to constants, but static items aren’t inlined upon use. This means that
there is only one instance for each value, and it’s at a fixed location in
memory.

Here’s an example:

```rust
static N: i32 = 5;
```

Unlike [`let`][let] bindings, you must annotate the type of a `static`.

Statics live for the entire lifetime of a program, and therefore any
reference stored in a constant has a [`'static` lifetime][lifetimes]:

```rust
static NAME: &'static str = "Steve";
```

[lifetimes]: lifetimes.html

## Mutability

You can introduce mutability with the `mut` keyword:

```rust
static mut N: i32 = 5;
```

Because this is mutable, one thread could be updating `N` while another is
reading it, causing memory unsafety. As such both accessing and mutating a
`static mut` is [`unsafe`][unsafe], and so must be done in an `unsafe` block:

```rust
# static mut N: i32 = 5;

unsafe {
    N += 1;

    println!("N: {}", N);
}
```

[unsafe]: unsafe.html

Furthermore, any type stored in a `static` must be `Sync`, and must not have
a [`Drop`][drop] implementation.

[drop]: drop.html

# Initializing

Both `const` and `static` have requirements for giving them a value. They must
be given a value that’s a constant expression. In other words, you cannot use
the result of a function call or anything similarly complex or at runtime.

# Which construct should I use?

Almost always, if you can choose between the two, choose `const`. It’s pretty
rare that you actually want a memory location associated with your constant,
and using a `const` allows for optimizations like constant propagation not only
in your crate but downstream crates.
