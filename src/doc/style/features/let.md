% Let binding

### Always separately bind RAII guards. [FIXME: needs RFC]

Prefer

```rust
fn use_mutex(m: sync::mutex::Mutex<int>) {
    let guard = m.lock();
    do_work(guard);
    drop(guard); // unlock the lock
    // do other work
}
```

over

```rust
fn use_mutex(m: sync::mutex::Mutex<int>) {
    do_work(m.lock());
    // do other work
}
```

As explained in the [RAII guide](../ownership/raii.md), RAII guards are values
that represent ownership of some resource and whose destructor releases the
resource. Because the lifetime of guards are significant, they should always be
explicitly `let`-bound to make the lifetime clear. Consider using an explicit
`drop` to release the resource early.

### Prefer conditional expressions to deferred initialization. [FIXME: needs RFC]

Prefer

```rust
let foo = match bar {
    Baz  => 0,
    Quux => 1
};
```

over

```rust
let foo;
match bar {
    Baz  => {
        foo = 0;
    }
    Quux => {
        foo = 1;
    }
}
```

unless the conditions for initialization are too complex to fit into a simple
conditional expression.

### Use type annotations for clarification; prefer explicit generics when inference fails. [FIXME: needs RFC]

Prefer

```rust
s.iter().map(|x| x * 2)
        .collect::<Vec<_>>()
```

over

```rust
let v: Vec<_> = s.iter().map(|x| x * 2)
                        .collect();
```

When the type of a value might be unclear to the _reader_ of the code, consider
explicitly annotating it in a `let`.

On the other hand, when the type is unclear to the _compiler_, prefer to specify
the type by explicit generics instantiation, which is usually more clear.

### Shadowing [FIXME]

> **[FIXME]** Repeatedly shadowing a binding is somewhat common in Rust code. We
> need to articulate a guideline on when it is appropriate/useful and when not.

### Prefer immutable bindings. [FIXME: needs RFC]

Use `mut` bindings to signal the span during which a value is mutated:

```rust
let mut v = Vec::new();
// push things onto v
let v = v;
// use v immutably henceforth
```

### Prefer to bind all `struct` or tuple fields. [FIXME: needs RFC]

When consuming a `struct` or tuple via a `let`, bind all of the fields rather
than using `..` to elide the ones you don't need. The benefit is that when
fields are added, the compiler will pinpoint all of the places where that type
of value was consumed, which will often need to be adjusted to take the new
field properly into account.
