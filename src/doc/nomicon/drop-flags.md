% Drop Flags

The examples in the previous section introduce an interesting problem for Rust.
We have seen that it's possible to conditionally initialize, deinitialize, and
reinitialize locations of memory totally safely. For Copy types, this isn't
particularly notable since they're just a random pile of bits. However types
with destructors are a different story: Rust needs to know whether to call a
destructor whenever a variable is assigned to, or a variable goes out of scope.
How can it do this with conditional initialization?

Note that this is not a problem that all assignments need worry about. In
particular, assigning through a dereference unconditionally drops, and assigning
in a `let` unconditionally doesn't drop:

```
let mut x = Box::new(0); // let makes a fresh variable, so never need to drop
let y = &mut x;
*y = Box::new(1); // Deref assumes the referent is initialized, so always drops
```

This is only a problem when overwriting a previously initialized variable or
one of its subfields.

It turns out that Rust actually tracks whether a type should be dropped or not
*at runtime*. As a variable becomes initialized and uninitialized, a *drop flag*
for that variable is toggled. When a variable might need to be dropped, this
flag is evaluated to determine if it should be dropped.

Of course, it is often the case that a value's initialization state can be
statically known at every point in the program. If this is the case, then the
compiler can theoretically generate more efficient code! For instance, straight-
line code has such *static drop semantics*:

```rust
let mut x = Box::new(0); // x was uninit; just overwrite.
let mut y = x;           // y was uninit; just overwrite and make x uninit.
x = Box::new(0);         // x was uninit; just overwrite.
y = x;                   // y was init; Drop y, overwrite it, and make x uninit!
                         // y goes out of scope; y was init; Drop y!
                         // x goes out of scope; x was uninit; do nothing.
```

Similarly, branched code where all branches have the same behavior with respect
to initialization has static drop semantics:

```rust
# let condition = true;
let mut x = Box::new(0);    // x was uninit; just overwrite.
if condition {
    drop(x)                 // x gets moved out; make x uninit.
} else {
    println!("{}", x);
    drop(x)                 // x gets moved out; make x uninit.
}
x = Box::new(0);            // x was uninit; just overwrite.
                            // x goes out of scope; x was init; Drop x!
```

However code like this *requires* runtime information to correctly Drop:

```rust
# let condition = true;
let x;
if condition {
    x = Box::new(0);        // x was uninit; just overwrite.
    println!("{}", x);
}
                            // x goes out of scope; x might be uninit;
                            // check the flag!
```

Of course, in this case it's trivial to retrieve static drop semantics:

```rust
# let condition = true;
if condition {
    let x = Box::new(0);
    println!("{}", x);
}
```

As of Rust 1.0, the drop flags are actually not-so-secretly stashed in a hidden
field of any type that implements Drop. Rust sets the drop flag by overwriting
the entire value with a particular bit pattern. This is pretty obviously Not
The Fastest and causes a bunch of trouble with optimizing code. It's legacy from
a time when you could do much more complex conditional initialization.

As such work is currently under way to move the flags out onto the stack frame
where they more reasonably belong. Unfortunately, this work will take some time
as it requires fairly substantial changes to the compiler.

Regardless, Rust programs don't need to worry about uninitialized values on
the stack for correctness. Although they might care for performance. Thankfully,
Rust makes it easy to take control here! Uninitialized values are there, and
you can work with them in Safe Rust, but you're never in danger.
